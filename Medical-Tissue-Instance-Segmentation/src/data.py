"""Dataset loading, tiling and the training/inference data pipeline.

Two annotated datasets are supported here: GlaS (colorectal glands, large
objects, small dataset) and PanNuke (nuclei, many tissue types, large dataset).
Both are loaded at native resolution and tiled on the fly.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from config import cfg
from metrics import hv_targets
from stain import get_normalizer, stain_jitter, tissue_mask

# ImageNet statistics, which the pretrained backbone expects
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def to_tensor(rgb_uint8: np.ndarray) -> torch.Tensor:
    x = rgb_uint8.astype(np.float32) / 255.0
    x = (x - _MEAN) / _STD
    return torch.from_numpy(x.transpose(2, 0, 1))


@dataclass
class Sample:
    """One annotated image: a path pair, loaded lazily."""
    image_path: str
    label_path: str
    dataset: str

    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        rgb = cv2.cvtColor(cv2.imread(self.image_path, cv2.IMREAD_COLOR),
                           cv2.COLOR_BGR2RGB)
        inst = _load_instances(self.label_path)
        return rgb, inst


def _load_instances(path: str) -> np.ndarray:
    """Read an instance map. GlaS stores one integer per gland in a bitmap."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path).astype(np.int32)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3:
        img = img[..., 0]
    return img.astype(np.int32)


# --------------------------------------------------------------------------- #
# GlaS                                                                         #
# --------------------------------------------------------------------------- #
def load_glas(root: str | None = None) -> Tuple[List[Sample], List[Sample]]:
    """Warwick-QU layout: ``<name>.bmp`` beside ``<name>_anno.bmp``."""
    root = root or cfg.glas_root
    if not root or not os.path.isdir(root):
        raise FileNotFoundError(
            "GlaS not found. Set BJEPA_GLAS_ROOT to the folder holding "
            "train_1.bmp, train_1_anno.bmp, ...")

    train, test = [], []
    for anno in sorted(glob.glob(os.path.join(root, "**", "*_anno.*"), recursive=True)):
        stem = anno.replace("_anno", "")
        if not os.path.exists(stem):
            continue
        target = test if os.path.basename(stem).lower().startswith("test") else train
        target.append(Sample(stem, anno, "glas"))

    if not train:
        raise RuntimeError(f"No GlaS image/annotation pairs found under {root}")
    return train, test


# --------------------------------------------------------------------------- #
# PanNuke                                                                      #
# --------------------------------------------------------------------------- #
def pannuke_masks_to_instances(mask: np.ndarray) -> np.ndarray:
    """PanNuke ships one channel per nucleus type plus a background channel.

    Instance ids restart within each channel, so they are offset before merging
    to keep every nucleus distinct.
    """
    out = np.zeros(mask.shape[:2], np.int32)
    offset = 0
    for c in range(mask.shape[2] - 1):          # last channel is background
        layer = mask[..., c].astype(np.int32)
        ids = [i for i in np.unique(layer) if i != 0]
        for i in ids:
            offset += 1
            out[layer == i] = offset
    return out


def load_pannuke_fold(root: str, fold: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Load one PanNuke fold as (image, instance map) pairs."""
    images = os.path.join(root, f"Fold {fold}", "images", f"fold{fold}", "images.npy")
    masks = os.path.join(root, f"Fold {fold}", "masks", f"fold{fold}", "masks.npy")
    if not (os.path.exists(images) and os.path.exists(masks)):
        raise FileNotFoundError(
            f"PanNuke fold {fold} not found under {root}. Expected {images}")

    imgs = np.load(images, mmap_mode="r")
    msks = np.load(masks, mmap_mode="r")
    return [(np.asarray(imgs[i], np.uint8), pannuke_masks_to_instances(np.asarray(msks[i])))
            for i in range(len(imgs))]


def pannuke_rotation(k: int) -> Tuple[int, int, int]:
    """Three-fold rotation: fold k trains, k+1 validates, k+2 tests."""
    return (k % 3) + 1, ((k + 1) % 3) + 1, ((k + 2) % 3) + 1


# --------------------------------------------------------------------------- #
# Datasets                                                                     #
# --------------------------------------------------------------------------- #
class TiledDataset(Dataset):
    """Random tiles from annotated images, with the targets each head needs.

    Every tile yields a semantic mask, a contour mask and — on nuclei — the
    distance maps. Contours come from the annotation itself, unlike Stage 1
    where they came from the staining.
    """

    def __init__(self, samples: List[Sample], kind: str = "gland",
                 tiles_per_image: int = 4, augment: bool = True,
                 use_hv: bool | None = None) -> None:
        self.samples = samples
        self.kind = kind
        self.tiles_per_image = tiles_per_image
        self.augment = augment
        self.use_hv = (kind == "nuclei") if use_hv is None else use_hv
        self.tile = cfg.tile_size
        self._cache: dict = {}

    def __len__(self) -> int:
        return len(self.samples) * self.tiles_per_image

    def _get(self, idx: int):
        if idx not in self._cache:
            rgb, inst = self.samples[idx].load()
            rgb, _ = get_normalizer().normalize(rgb)
            self._cache[idx] = (rgb, inst)
        return self._cache[idx]

    def __getitem__(self, i: int):
        rgb, inst = self._get(i // self.tiles_per_image)
        h, w = inst.shape
        t = self.tile

        if h < t or w < t:                      # pad small images
            rgb = cv2.copyMakeBorder(rgb, 0, max(0, t - h), 0, max(0, t - w),
                                     cv2.BORDER_REFLECT)
            inst = cv2.copyMakeBorder(inst, 0, max(0, t - h), 0, max(0, t - w),
                                      cv2.BORDER_CONSTANT, value=0)
            h, w = inst.shape

        y = np.random.randint(0, h - t + 1)
        x = np.random.randint(0, w - t + 1)
        rgb_t = rgb[y:y + t, x:x + t]
        inst_t = inst[y:y + t, x:x + t]

        if self.augment:
            if np.random.rand() < 0.5:
                rgb_t, inst_t = rgb_t[:, ::-1], inst_t[:, ::-1]
            if np.random.rand() < 0.5:
                rgb_t, inst_t = rgb_t[::-1], inst_t[::-1]
            k = np.random.randint(4)
            rgb_t, inst_t = np.rot90(rgb_t, k), np.rot90(inst_t, k)
            if np.random.rand() < 0.3:
                rgb_t = stain_jitter(np.ascontiguousarray(rgb_t))

        rgb_t = np.ascontiguousarray(rgb_t)
        inst_t = np.ascontiguousarray(inst_t)

        sem = (inst_t > 0).astype(np.float32)
        contour = _contours_from_instances(inst_t)

        out = {"image": to_tensor(rgb_t),
               "sem": torch.from_numpy(sem)[None],
               "bnd": torch.from_numpy(contour)[None]}
        if self.use_hv:
            out["hv"] = torch.from_numpy(hv_targets(inst_t))
        return out


def _contours_from_instances(inst: np.ndarray, width: int = 2) -> np.ndarray:
    """One-pixel-wide object outlines, dilated slightly so they are learnable."""
    contour = np.zeros(inst.shape, np.float32)
    for lab in np.unique(inst):
        if lab == 0:
            continue
        m = (inst == lab).astype(np.uint8)
        eroded = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=1)
        contour[(m - eroded) > 0] = 1.0
    if width > 1:
        contour = cv2.dilate(contour, np.ones((width, width), np.uint8))
    return contour


class UnlabelledTiles(Dataset):
    """Tiles with no annotation, for Stage 1 and for semi-supervised training.

    Returns the boundary and tissue targets computed from the staining, which is
    what lets Stage 1 train without any expert input.
    """

    def __init__(self, images: List[np.ndarray], n_tiles: int = 8) -> None:
        self.images = images
        self.n_tiles = n_tiles
        self.tile = cfg.tile_size

    def __len__(self) -> int:
        return len(self.images) * self.n_tiles

    def __getitem__(self, i: int):
        from stain import boundary_target

        rgb = self.images[i // self.n_tiles]
        h, w = rgb.shape[:2]
        t = self.tile
        y = np.random.randint(0, max(1, h - t + 1))
        x = np.random.randint(0, max(1, w - t + 1))
        crop = np.ascontiguousarray(rgb[y:y + t, x:x + t])
        if crop.shape[0] < t or crop.shape[1] < t:
            crop = cv2.copyMakeBorder(crop, 0, t - crop.shape[0], 0, t - crop.shape[1],
                                      cv2.BORDER_REFLECT)

        norm, hema = get_normalizer().normalize(crop)
        return {"image": to_tensor(norm),
                "boundary": torch.from_numpy(boundary_target(hema).reshape(-1)),
                "tissue": torch.from_numpy(tissue_mask(norm).reshape(-1))}


def split_train_val(samples: List[Sample], seed: int = 0,
                    val_fraction: float = 0.2) -> Tuple[List[Sample], List[Sample]]:
    """Carve a validation split deterministically from the training set."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(samples))
    n_val = max(1, int(len(samples) * val_fraction))
    val_idx = set(perm[:n_val].tolist())
    train = [s for i, s in enumerate(samples) if i not in val_idx]
    val = [s for i, s in enumerate(samples) if i in val_idx]
    return train, val


def prime_normalizer(train_samples: List[Sample]) -> None:
    """Fit the stain normaliser on the first *training* image.

    This must be a training image. Fitting on test data would leak test colour
    statistics into training, which quietly inflates every result.
    """
    from stain import reset_normalizer
    reset_normalizer()
    reference, _ = train_samples[0].load()
    get_normalizer(reference_uint8=reference)
