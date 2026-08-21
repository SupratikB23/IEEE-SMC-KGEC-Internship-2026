"""Stage 0 — preparation.

Stain-normalise every image, compute the annotation-free boundary target and
the tissue mask, and cache the results so later stages never recompute them.

Run:  python phase0_prep.py
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from config import cfg
from data import load_glas, prime_normalizer
from stain import boundary_target, get_normalizer, tissue_mask


def prepare_dataset(dataset: str, limit: int | None = None) -> dict:
    """Normalise, derive Stage-1 targets, and cache them to disk."""
    print(f"[stage 0] preparing {dataset}")
    os.makedirs(cfg.cache_dir, exist_ok=True)

    if dataset == "glas":
        train, test = load_glas()
    else:
        raise ValueError(f"{dataset} preparation is not part of this release")

    # the reference must come from training data, never from test
    prime_normalizer(train)
    print(f"[stage 0] stain reference fitted on {os.path.basename(train[0].image_path)}")

    samples = train if limit is None else train[:limit]
    boundaries, tissues = [], []
    started = time.time()

    for i, sample in enumerate(samples):
        rgb, _ = sample.load()
        norm, hema = get_normalizer().normalize(rgb)
        boundaries.append(boundary_target(hema))
        tissues.append(tissue_mask(norm))
        if (i + 1) % 10 == 0 or i + 1 == len(samples):
            print(f"  {i + 1}/{len(samples)} images", flush=True)

    boundaries = np.stack(boundaries).astype(np.float32)
    tissues = np.stack(tissues).astype(np.float32)
    np.save(os.path.join(cfg.cache_dir, f"{dataset}_boundary.npy"), boundaries)
    np.save(os.path.join(cfg.cache_dir, f"{dataset}_tissue.npy"), tissues)

    meta = {
        "dataset": dataset,
        "n_train": len(train),
        "n_test": len(test),
        "n_cached": len(samples),
        "grid": cfg.grid,
        "edge_tau": cfg.edge_tau,
        "edge_occupancy": cfg.edge_occupancy,
        "boundary_token_fraction": float(boundaries.mean()),
        "tissue_token_fraction": float(tissues.mean()),
        "seconds": round(time.time() - started, 1),
    }
    with open(os.path.join(cfg.cache_dir, f"{dataset}_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"[stage 0] cached {len(samples)} images to {cfg.cache_dir}")
    print(f"[stage 0] {meta['boundary_token_fraction']:.1%} of tokens are boundary, "
          f"{meta['tissue_token_fraction']:.1%} are tissue")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 0: preprocessing and caching")
    ap.add_argument("--dataset", default=cfg.datasets[0])
    ap.add_argument("--limit", type=int, default=None,
                    help="cache only the first N images (for a quick check)")
    args = ap.parse_args()
    prepare_dataset(args.dataset, args.limit)


if __name__ == "__main__":
    main()
