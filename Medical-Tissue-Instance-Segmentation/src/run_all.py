"""Run the three stages in order, for every dataset and seed in the config.

Run:  python run_all.py
      python run_all.py --skip-ssl        (supervised only)
      python run_all.py --smoke           (tiny run, checks the wiring)
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from config import DEVICE, cfg, describe


def main() -> None:
    ap = argparse.ArgumentParser(description="Full pipeline")
    ap.add_argument("--skip-prep", action="store_true")
    ap.add_argument("--skip-ssl", action="store_true",
                    help="train adapters from scratch in Stage 2 instead")
    ap.add_argument("--smoke", action="store_true",
                    help="a few images and epochs, to verify the pipeline runs")
    args = ap.parse_args()

    print("=" * 66)
    print("Deep Learning for Medical Tissue Image Analysis")
    print("Automatic Segmentation of Glands and Nuclei in Microscope Images")
    print("=" * 66)
    print(describe())
    print()

    started = time.time()
    limit = 4 if args.smoke else None
    epochs = 2 if args.smoke else None

    # ---- Stage 0 ---------------------------------------------------------
    if not args.skip_prep:
        from phase0_prep import prepare_dataset
        for dataset in cfg.datasets:
            prepare_dataset(dataset, limit=limit)
        print()

    # ---- Stage 1 ---------------------------------------------------------
    adapter_path = os.path.join(cfg.ssl_dir, "adapter.pt")
    if not args.skip_ssl:
        from data import load_glas
        from phase1_ssl import train as train_ssl

        train_samples, _ = load_glas()
        if limit:
            train_samples = train_samples[:limit]
        images = [s.load()[0] for s in train_samples]
        adapter_path = train_ssl(images, epochs=epochs, seed=cfg.seeds[0])
        print()

    # ---- Stage 2 ---------------------------------------------------------
    from phase2_train import train_one

    all_results = []
    for dataset in cfg.datasets:
        for seed in cfg.seeds:
            all_results.append(train_one(
                dataset, seed, epochs=epochs,
                adapter_path=None if args.skip_ssl else adapter_path))
            print()

    # ---- summary ---------------------------------------------------------
    print("=" * 66)
    print("SUMMARY")
    print("=" * 66)
    for dataset in cfg.datasets:
        rows = [r for r in all_results if r["dataset"] == dataset]
        if not rows:
            continue
        print(f"\n{dataset}  ({len(rows)} seed(s))")
        for metric in ("dice", "object_dice", "object_f1", "aji", "pq"):
            values = [r[metric] for r in rows]
            spread = f" +- {np.std(values, ddof=1):.4f}" if len(values) > 1 else ""
            print(f"  {metric:<14} {np.mean(values):.4f}{spread}")

    out = os.path.join(cfg.results_dir, "summary.json")
    os.makedirs(cfg.results_dir, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"\nAll results written to {out}")
    print(f"Total time: {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
