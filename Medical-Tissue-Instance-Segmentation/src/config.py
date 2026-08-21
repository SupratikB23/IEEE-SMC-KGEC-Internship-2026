"""Central configuration.

Every setting is overridable from the environment, so no source file needs
editing between experiments. Defaults are the values used for the results in
the internship report.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Tuple

import torch


def _f(key: str, default: float) -> float:
    return float(os.environ.get(key, default))


def _i(key: str, default: int) -> int:
    return int(os.environ.get(key, default))


def _s(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _list_s(key: str, default: str) -> Tuple[str, ...]:
    return tuple(x.strip() for x in os.environ.get(key, default).split(",") if x.strip())


def _list_i(key: str, default: str) -> Tuple[int, ...]:
    return tuple(int(x) for x in _list_s(key, default))


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Config:
    # ---- backbone ---------------------------------------------------------
    # Phikon-v2 is a ViT-L/16 pathology foundation model: 24 layers, 1024-d
    # tokens, 307M parameters. It stays frozen at all times.
    backbone: str = _s("BJEPA_BACKBONE", "owkin/phikon-v2")
    tile_size: int = 224
    patch_size: int = 16
    grid: int = 14                      # 224 / 16
    n_tokens: int = 196                 # 14 * 14
    embed_dim: int = 1024
    tap_layers: Tuple[int, ...] = (6, 12, 18, 24)   # multi-scale feature taps

    # ---- adapters ---------------------------------------------------------
    # Rank 8 over the 96 attention projections (4 per block x 24 blocks) gives
    # ~1.67M trainable parameters, about 0.54% of the backbone.
    adapter_rank: int = _i("BJEPA_LORA_RANK", 8)
    adapter_alpha: float = _f("BJEPA_LORA_ALPHA", 16.0)

    # ---- stage 1: self-supervised adaptation ------------------------------
    mu_boundary: float = _f("BJEPA_MU", 0.3)        # weight on the boundary term
    mask_ratio: float = _f("BJEPA_MASK_RATIO", 0.5)
    ssl_epochs: int = _i("BJEPA_SSL_EPOCHS", 30)
    ssl_batch: int = _i("BJEPA_SSL_BATCH", 32)
    ssl_lr: float = _f("BJEPA_SSL_LR", 1e-4)
    edge_tau: float = _f("BJEPA_EDGE_TAU", 0.10)    # hematoxylin gradient threshold
    edge_occupancy: float = _f("BJEPA_EDGE_OCC", 0.10)  # fraction of a token that must be edge
    boundary_pos_weight: float = _f("BJEPA_BND_POS_W", 10.0)

    # ---- stage 2: supervised read-out -------------------------------------
    decoder: str = _s("BJEPA_DECODER", "unet")      # unet | segformer
    ft_epochs: int = _i("BJEPA_FT_EPOCHS", 80)
    batch_size: int = _i("BJEPA_BATCH", 8)
    adapter_lr: float = _f("BJEPA_ADAPTER_LR", 1e-4)
    decoder_lr: float = _f("BJEPA_DECODER_LR", 1e-3)

    # supervised loss weights (Eq. in the report's methodology section)
    w_dice: float = _f("BJEPA_W_DICE", 1.0)
    w_tversky: float = _f("BJEPA_W_TVERSKY", 0.5)
    w_boundary: float = _f("BJEPA_W_BOUNDARY", 0.3)
    tversky_alpha_gland: float = _f("BJEPA_TVERSKY_A_GLAND", 0.5)
    tversky_alpha_nuclei: float = _f("BJEPA_TVERSKY_A_NUCLEI", 0.3)

    # ---- distance-map instance head (nuclei only) -------------------------
    use_hv: bool = _s("BJEPA_USE_HV", "1") == "1"
    hv_grad_weight: float = _f("BJEPA_HV_GRAD_W", 1.0)
    hv_marker_thr: float = _f("BJEPA_HV_MARKER_THR", 0.40)
    hv_min_size: int = _i("BJEPA_HV_MIN", 10)

    # ---- semi-supervised co-training --------------------------------------
    ema_decay: float = _f("BJEPA_EMA", 0.99)
    lambda_cons: float = _f("BJEPA_LAMBDA_CONS", 1.0)
    noise_std: float = _f("BJEPA_NOISE_STD", 0.1)

    # ---- inference --------------------------------------------------------
    stride: int = _i("BJEPA_STRIDE", 168)           # 224 tile, 56 px overlap
    stitch_sigma: float = _f("BJEPA_STITCH_SIGMA", 0.30)  # fraction of tile side
    sem_thr_gland: float = _f("BJEPA_SEM_THR_GLAND", 0.50)
    bnd_thr_gland: float = _f("BJEPA_BND_THR_GLAND", 0.60)
    min_size_gland: int = _i("BJEPA_MIN_GLAND", 200)

    # ---- experiment grid --------------------------------------------------
    datasets: Tuple[str, ...] = field(default_factory=lambda: _list_s("BJEPA_DATASETS", "glas"))
    seeds: Tuple[int, ...] = field(default_factory=lambda: _list_i("BJEPA_SEEDS", "0"))

    # ---- paths ------------------------------------------------------------
    state_root: str = _s("BJEPA_STATE_ROOT", "./bjepa_state")
    glas_root: str = _s("BJEPA_GLAS_ROOT", "")
    pannuke_root: str = _s("BJEPA_PANNUKE_ROOT", "")

    def kind(self, dataset: str) -> str:
        """Glands and nuclei need different instance read-outs."""
        return "gland" if dataset.lower() == "glas" else "nuclei"

    def tversky_alpha(self, dataset: str) -> float:
        return (self.tversky_alpha_nuclei if self.kind(dataset) == "nuclei"
                else self.tversky_alpha_gland)

    @property
    def cache_dir(self) -> str:
        return os.path.join(self.state_root, "stage0_cache")

    @property
    def ssl_dir(self) -> str:
        return os.path.join(self.state_root, "stage1_ssl")

    @property
    def results_dir(self) -> str:
        return os.path.join(self.state_root, "stage2_results")


cfg = Config()


def seed_everything(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def describe() -> str:
    return (f"backbone={cfg.backbone} device={DEVICE} decoder={cfg.decoder} "
            f"rank={cfg.adapter_rank} datasets={cfg.datasets} seeds={cfg.seeds}")
