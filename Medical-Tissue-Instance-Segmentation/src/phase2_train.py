"""Stage 2 — supervised read-out and evaluation.

Trains the decoder (and the adapters) on annotated images, then evaluates at the
image's native resolution. This is the only stage that uses annotations.

Run:  python phase2_train.py --dataset glas --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import DEVICE, cfg, seed_everything
from data import (Sample, TiledDataset, load_glas, prime_normalizer,
                  split_train_val, to_tensor)
from losses import SupervisedLoss, distance_loss
from metrics import (aggregate, evaluate, instances_from_contour,
                     instances_from_distance)
from model import SegModel, count_parameters
from stain import get_normalizer


# --------------------------------------------------------------------------- #
# Native-resolution inference                                                  #
# --------------------------------------------------------------------------- #
def _gaussian_window(size: int, sigma_fraction: float) -> np.ndarray:
    """Weights peaking at the tile centre, so overlapping tiles blend smoothly
    and no seam appears where two tiles meet."""
    axis = np.arange(size, dtype=np.float32) - (size - 1) / 2.0
    sigma = max(1.0, sigma_fraction * size)
    g = np.exp(-(axis ** 2) / (2.0 * sigma ** 2))
    return np.maximum(np.outer(g, g), 1e-3).astype(np.float32)


@torch.no_grad()
def predict_native(model: SegModel, rgb: np.ndarray) -> Dict[str, np.ndarray]:
    """Tile a full-resolution image, predict, and blend the overlaps.

    A stride of 168 against a 224 tile gives every pixel at least two votes.
    """
    model.eval()
    tile, stride = cfg.tile_size, cfg.stride
    h, w = rgb.shape[:2]

    pad_h, pad_w = max(0, tile - h), max(0, tile - w)
    if pad_h or pad_w:
        rgb = np.pad(rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        h, w = rgb.shape[:2]

    window = _gaussian_window(tile, cfg.stitch_sigma)
    sem = np.zeros((h, w), np.float32)
    bnd = np.zeros((h, w), np.float32)
    hv = np.zeros((2, h, w), np.float32) if model.use_hv else None
    norm = np.zeros((h, w), np.float32)

    ys = list(range(0, max(1, h - tile + 1), stride))
    xs = list(range(0, max(1, w - tile + 1), stride))
    if ys[-1] != h - tile:
        ys.append(h - tile)
    if xs[-1] != w - tile:
        xs.append(w - tile)

    for y in ys:
        for x in xs:
            crop = rgb[y:y + tile, x:x + tile]
            batch = to_tensor(crop)[None].to(DEVICE)
            s, b, v = model(batch)
            s = torch.sigmoid(s)[0, 0].cpu().numpy()
            b = torch.sigmoid(b)[0, 0].cpu().numpy()

            sem[y:y + tile, x:x + tile] += s * window
            bnd[y:y + tile, x:x + tile] += b * window
            norm[y:y + tile, x:x + tile] += window
            if hv is not None and v is not None:
                hv[:, y:y + tile, x:x + tile] += v[0].cpu().numpy() * window

    norm = np.maximum(norm, 1e-6)
    out = {"sem": sem / norm, "bnd": bnd / norm}
    if hv is not None:
        out["hv"] = hv / norm
    return out


def to_instances(pred: Dict[str, np.ndarray], kind: str) -> np.ndarray:
    if kind == "nuclei" and "hv" in pred:
        return instances_from_distance(pred["sem"], pred["hv"])
    return instances_from_contour(pred["sem"], pred["bnd"])


def evaluate_split(model: SegModel, samples: List[Sample], kind: str) -> Dict[str, float]:
    rows = []
    for sample in samples:
        rgb, gt = sample.load()
        rgb, _ = get_normalizer().normalize(rgb)
        pred = to_instances(predict_native(model, rgb), kind)
        pred = pred[:gt.shape[0], :gt.shape[1]]
        rows.append(evaluate(gt, pred))
    return aggregate(rows)


# --------------------------------------------------------------------------- #
# Training                                                                     #
# --------------------------------------------------------------------------- #
def train_one(dataset: str = "glas", seed: int = 0, epochs: int | None = None,
              adapter_path: str | None = None) -> Dict[str, float]:
    epochs = epochs or cfg.ft_epochs
    kind = cfg.kind(dataset)
    seed_everything(seed)

    train_all, test = load_glas()
    prime_normalizer(train_all)
    train_samples, val_samples = split_train_val(train_all, seed)
    print(f"[stage 2] {dataset} seed={seed} kind={kind} "
          f"train={len(train_samples)} val={len(val_samples)} test={len(test)}")

    model = SegModel(kind=kind).to(DEVICE)
    if adapter_path and os.path.exists(adapter_path):
        blob = torch.load(adapter_path, map_location="cpu")
        model.backbone.load_state_dict(blob["adapter"], strict=False)
        print(f"[stage 2] initialised adapters from {adapter_path}")

    total, trainable = count_parameters(model)
    print(f"[stage 2] {total / 1e6:.1f}M parameters, {trainable / 1e6:.2f}M trainable "
          f"({100 * trainable / total:.2f}%)")

    # adapters and decoder learn at different rates
    adapter_params = [p for n, p in model.named_parameters()
                      if p.requires_grad and "backbone" in n]
    decoder_params = [p for n, p in model.named_parameters()
                      if p.requires_grad and "backbone" not in n]
    opt = torch.optim.AdamW(
        [{"params": adapter_params, "lr": cfg.adapter_lr},
         {"params": decoder_params, "lr": cfg.decoder_lr}], weight_decay=1e-4)

    criterion = SupervisedLoss(tversky_alpha=cfg.tversky_alpha(dataset)).to(DEVICE)
    loader = DataLoader(TiledDataset(train_samples, kind=kind),
                        batch_size=cfg.batch_size, shuffle=True,
                        num_workers=0, drop_last=True)

    best_metric, best_state = -1.0, None
    for epoch in range(epochs):
        model.train()
        running, seen, started = 0.0, 0, time.time()

        for batch in loader:
            image = batch["image"].to(DEVICE)
            sem_gt = batch["sem"].to(DEVICE)
            bnd_gt = batch["bnd"].to(DEVICE)

            sem, bnd, hv = model(image)
            loss = criterion(sem, bnd, sem_gt, bnd_gt)
            if hv is not None and "hv" in batch:
                loss = loss + distance_loss(hv, batch["hv"].to(DEVICE), sem_gt)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 5.0)
            opt.step()
            running += float(loss.detach())
            seen += 1

        line = f"[stage 2] epoch {epoch + 1}/{epochs} loss={running / max(1, seen):.4f}"

        # select on the metric being reported, not on pixel accuracy: a
        # high-Dice checkpoint can still be merging every touching object
        if val_samples and ((epoch + 1) % 5 == 0 or epoch + 1 == epochs):
            val = evaluate_split(model, val_samples, kind)
            line += f"  val_object_dice={val['object_dice']:.4f} val_aji={val['aji']:.4f}"
            if val["object_dice"] > best_metric:
                best_metric = val["object_dice"]
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
        print(line + f"  ({time.time() - started:.0f}s)", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[stage 2] restored best checkpoint (val object Dice {best_metric:.4f})")

    results = evaluate_split(model, test, kind)
    results.update({"dataset": dataset, "seed": seed, "kind": kind,
                    "decoder": cfg.decoder, "epochs": epochs,
                    "trainable_M": round(trainable / 1e6, 3)})

    os.makedirs(cfg.results_dir, exist_ok=True)
    path = os.path.join(cfg.results_dir, f"{dataset}_seed{seed}.json")
    with open(path, "w") as fh:
        json.dump(results, fh, indent=2)

    print(f"[stage 2] TEST  dice={results['dice']:.4f}  "
          f"object_dice={results['object_dice']:.4f}  aji={results['aji']:.4f}  "
          f"pq={results['pq']:.4f}")
    print(f"[stage 2] written to {path}")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 2: supervised read-out")
    ap.add_argument("--dataset", default=cfg.datasets[0])
    ap.add_argument("--seed", type=int, default=cfg.seeds[0])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--adapter", default=os.path.join(cfg.ssl_dir, "adapter.pt"),
                    help="Stage-1 adapter; omit to train from a fresh adapter")
    args = ap.parse_args()
    train_one(args.dataset, args.seed, args.epochs, args.adapter)


if __name__ == "__main__":
    main()
