"""Stage 0 image operations: stain normalisation, the hematoxylin channel, the
annotation-free boundary target, and tissue-region detection.

The boundary target is the part worth understanding. Hematoxylin binds to
nuclei, so its concentration changes sharply wherever one cell meets the next.
Thresholding the gradient of that channel marks object edges without any
pathologist having annotated anything.
"""
from __future__ import annotations

import cv2
import numpy as np

from config import cfg

# Reference optical-density matrix for H&E (Ruifrok & Johnston). Columns are the
# hematoxylin, eosin and residual stain vectors.
_HE_REFERENCE = np.array([[0.6443, 0.0928],
                          [0.7166, 0.9541],
                          [0.2668, 0.2831]], dtype=np.float64)


def optical_density(rgb_uint8: np.ndarray, i0: float = 255.0) -> np.ndarray:
    """Beer-Lambert optical density: OD = -log10(I / I0)."""
    rgb = rgb_uint8.astype(np.float64)
    rgb = np.maximum(rgb, 1.0)              # avoid log(0)
    return -np.log10(rgb / i0)


def _stain_matrix_macenko(od: np.ndarray, beta: float = 0.15,
                          alpha: float = 1.0) -> np.ndarray:
    """Estimate the 3x2 stain matrix by Macenko's method.

    Projects the optical-density cloud onto the plane of its two leading
    singular vectors and takes robust angular extremes as the stain vectors.
    """
    od_flat = od.reshape(-1, 3)
    od_flat = od_flat[np.all(od_flat > beta, axis=1)]   # drop near-white pixels
    if od_flat.shape[0] < 32:
        return _HE_REFERENCE.copy()

    _, _, vt = np.linalg.svd(od_flat - od_flat.mean(0), full_matrices=False)
    plane = vt[:2].T                                    # 3x2
    proj = od_flat @ plane
    angles = np.arctan2(proj[:, 1], proj[:, 0])
    lo, hi = np.percentile(angles, alpha), np.percentile(angles, 100 - alpha)

    v1 = plane @ np.array([np.cos(lo), np.sin(lo)])
    v2 = plane @ np.array([np.cos(hi), np.sin(hi)])
    stains = np.stack([v1, v2], axis=1)
    stains /= (np.linalg.norm(stains, axis=0, keepdims=True) + 1e-8)
    stains = np.abs(stains)

    # hematoxylin absorbs most in red, so order by the red component
    if stains[0, 0] < stains[0, 1]:
        stains = stains[:, ::-1]
    return stains


def concentrations(rgb_uint8: np.ndarray, stain_matrix: np.ndarray) -> np.ndarray:
    """Unmix an image into per-stain concentrations, shape (H, W, 2)."""
    h, w = rgb_uint8.shape[:2]
    od = optical_density(rgb_uint8).reshape(-1, 3).T        # 3 x N
    conc = np.linalg.pinv(stain_matrix) @ od                # 2 x N
    return conc.T.reshape(h, w, 2)


class MacenkoNormalizer:
    """Fit once on a reference tile, then normalise every image to match it.

    Slides scanned in different laboratories differ markedly in colour. Fixing
    a single reference means the rest of the pipeline sees one consistent
    colour distribution regardless of where an image came from.
    """

    def __init__(self) -> None:
        self.stain_matrix = _HE_REFERENCE.copy()
        self.max_conc = np.array([1.9705, 1.0308])
        self._fitted = False

    def fit(self, reference_uint8: np.ndarray) -> "MacenkoNormalizer":
        self.stain_matrix = _stain_matrix_macenko(optical_density(reference_uint8))
        conc = concentrations(reference_uint8, self.stain_matrix).reshape(-1, 2)
        self.max_conc = np.percentile(conc, 99, axis=0) + 1e-8
        self._fitted = True
        return self

    def normalize(self, rgb_uint8: np.ndarray):
        """Return (normalised RGB uint8, hematoxylin channel in [0, 1])."""
        h, w = rgb_uint8.shape[:2]
        src_matrix = _stain_matrix_macenko(optical_density(rgb_uint8))
        conc = concentrations(rgb_uint8, src_matrix).reshape(-1, 2).T   # 2 x N

        src_max = np.percentile(conc, 99, axis=1)[:, None] + 1e-8
        conc = conc * (self.max_conc[:, None] / src_max)

        od = self.stain_matrix @ conc
        rgb = np.clip(255.0 * np.power(10.0, -od), 0, 255)
        rgb = rgb.T.reshape(h, w, 3).astype(np.uint8)

        hema = conc[0].reshape(h, w)
        hema = hema / (np.percentile(hema, 99) + 1e-8)
        return rgb, np.clip(hema, 0.0, 1.0).astype(np.float32)


_NORMALIZER: MacenkoNormalizer | None = None


def get_normalizer(reference_uint8: np.ndarray | None = None) -> MacenkoNormalizer:
    """Process-wide normaliser, fitted once on a *training* reference tile.

    Fitting on a training image matters: fitting on test data would leak test
    colour statistics into training.
    """
    global _NORMALIZER
    if _NORMALIZER is None:
        _NORMALIZER = MacenkoNormalizer()
        if reference_uint8 is not None:
            _NORMALIZER.fit(reference_uint8)
    return _NORMALIZER


def reset_normalizer() -> None:
    global _NORMALIZER
    _NORMALIZER = None


def boundary_target(hema: np.ndarray, grid: int | None = None,
                    tau: float | None = None,
                    occupancy: float | None = None) -> np.ndarray:
    """Annotation-free boundary target at token resolution.

    Three steps: mark pixels whose hematoxylin gradient exceeds ``tau``, average
    those marks down to the token grid, then keep a token as boundary when at
    least ``occupancy`` of it was marked.
    """
    grid = grid or cfg.grid
    tau = cfg.edge_tau if tau is None else tau
    occupancy = cfg.edge_occupancy if occupancy is None else occupancy

    gx = cv2.Sobel(hema, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(hema, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx * gx + gy * gy)

    edges = (magnitude >= tau).astype(np.float32)
    pooled = cv2.resize(edges, (grid, grid), interpolation=cv2.INTER_AREA)
    return (pooled >= occupancy).astype(np.float32)


def tissue_mask(rgb_uint8: np.ndarray, grid: int | None = None) -> np.ndarray:
    """Per-token tissue mask; True where the token is more than half tissue.

    A whole-slide tile is mostly blank glass. Without this the background would
    dominate every average and the boundary head would learn very little.
    """
    grid = grid or cfg.grid
    gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)
    _, tissue = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    tissue = cv2.morphologyEx(tissue, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    # saturation fallback catches fatty or necrotic tissue that Otsu can miss
    hsv = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2HSV)
    saturated = (hsv[..., 1] > 30).astype(np.uint8) * 255
    tissue = np.clip(tissue.astype(np.int32) + saturated, 0, 255).astype(np.float32) / 255.0

    pooled = cv2.resize(tissue, (grid, grid), interpolation=cv2.INTER_AREA)
    return (pooled > 0.5).astype(np.float32)


def stain_jitter(rgb_uint8: np.ndarray, sigma: float = 0.05) -> np.ndarray:
    """Randomly perturb stain concentrations — augmentation for training."""
    src = _stain_matrix_macenko(optical_density(rgb_uint8))
    h, w = rgb_uint8.shape[:2]
    conc = concentrations(rgb_uint8, src).reshape(-1, 2).T
    conc = conc * (1.0 + np.random.randn(2, 1) * sigma)
    od = src @ conc
    rgb = np.clip(255.0 * np.power(10.0, -od), 0, 255)
    return rgb.T.reshape(h, w, 3).astype(np.uint8)
