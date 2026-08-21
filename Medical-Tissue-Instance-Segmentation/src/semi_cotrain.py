"""Semi-supervised co-training for the low-annotation regime.

Trains on K annotated images plus the rest of the dataset left unlabelled. A
slowly-updated teacher predicts on clean unlabelled images and the student must
agree on a perturbed copy — so unlabelled data contributes without ever being
annotated.

The unlabelled images come from the target dataset itself, which is what
separates this from the cross-domain pre-adaptation of Stage 1.

Run:  python semi_cotrain.py --budget 16
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import DEVICE, cfg, seed_everything
from data import TiledDataset, load_glas, prime_normalizer, split_train_val
from losses import (SupervisedLoss, consistency_loss, distance_loss, rampup,
                    update_teacher)
from metrics import aggregate
from model import SegModel
from phase2_train import evaluate_split


def train_semi(dataset: str = "glas", budget: int = 16, seed: int = 0,
               epochs: int | None = None) -> Dict[str, float]:
    epochs = epochs or cfg.ft_epochs
    kind = cfg.kind(dataset)
    seed_everything(seed)

    train_all, test = load_glas()
    prime_normalizer(train_all)
    pool, val_samples = split_train_val(train_all, seed)

    # K annotated images; everything else keeps its images but discards labels
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(pool))
    labelled = [pool[i] for i in order[:budget]]
    unlabelled = [pool[i] for i in order[budget:]]
    print(f"[semi] {dataset} seed={seed} K={budget} labelled, "
          f"{len(unlabelled)} unlabelled, val={len(val_samples)}")

    if not unlabelled:
        print("[semi] no unlabelled images left — this is the fully supervised control")

    student = SegModel(kind=kind).to(DEVICE)
    teacher = copy.deepcopy(student).to(DEVICE)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()

    params = student.trainable_parameters()
    opt = torch.optim.AdamW(params, lr=cfg.decoder_lr, weight_decay=1e-4)
    criterion = SupervisedLoss(tversky_alpha=cfg.tversky_alpha(dataset)).to(DEVICE)

    lab_loader = DataLoader(TiledDataset(labelled, kind=kind),
                            batch_size=cfg.batch_size, shuffle=True,
                            num_workers=0, drop_last=True)
    unlab_loader = (DataLoader(TiledDataset(unlabelled, kind=kind, augment=True),
                               batch_size=cfg.batch_size, shuffle=True,
                               num_workers=0, drop_last=True)
                    if unlabelled else None)

    ramp_length = max(1, epochs // 2)
    best_metric, best_state = -1.0, None

    for epoch in range(epochs):
        student.train()
        weight = cfg.lambda_cons * rampup(epoch, ramp_length)
        unlab_iter = iter(unlab_loader) if unlab_loader else None
        running, seen, started = 0.0, 0, time.time()

        for batch in lab_loader:
            image = batch["image"].to(DEVICE)
            sem_gt = batch["sem"].to(DEVICE)
            bnd_gt = batch["bnd"].to(DEVICE)

            sem, bnd, hv = student(image)
            loss = criterion(sem, bnd, sem_gt, bnd_gt)
            if hv is not None and "hv" in batch:
                loss = loss + distance_loss(hv, batch["hv"].to(DEVICE), sem_gt)

            if unlab_iter is not None and weight > 1e-4:
                try:
                    ub = next(unlab_iter)
                except StopIteration:
                    unlab_iter = iter(unlab_loader)
                    ub = next(unlab_iter)

                clean = ub["image"].to(DEVICE)
                noisy = clean + torch.randn_like(clean) * cfg.noise_std
                with torch.no_grad():
                    t_sem, _, _ = teacher(clean)
                s_sem, _, _ = student(noisy)
                loss = loss + weight * consistency_loss(s_sem, t_sem)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            update_teacher(teacher, student, cfg.ema_decay)

            running += float(loss.detach())
            seen += 1

        line = (f"[semi] epoch {epoch + 1}/{epochs} loss={running / max(1, seen):.4f} "
                f"cons_w={weight:.3f}")
        if val_samples and ((epoch + 1) % 5 == 0 or epoch + 1 == epochs):
            val = evaluate_split(student, val_samples, kind)
            line += f"  val_object_dice={val['object_dice']:.4f}"
            if val["object_dice"] > best_metric:
                best_metric = val["object_dice"]
                best_state = {k: v.detach().cpu().clone()
                              for k, v in student.state_dict().items()}
        print(line + f"  ({time.time() - started:.0f}s)", flush=True)

    if best_state is not None:
        student.load_state_dict(best_state)

    results = evaluate_split(student, test, kind)
    results.update({"dataset": dataset, "seed": seed, "budget": budget,
                    "method": "semi", "kind": kind})

    os.makedirs(cfg.results_dir, exist_ok=True)
    path = os.path.join(cfg.results_dir, f"{dataset}_semi_K{budget}_seed{seed}.json")
    with open(path, "w") as fh:
        json.dump(results, fh, indent=2)

    print(f"[semi] TEST  object_dice={results['object_dice']:.4f}  "
          f"aji={results['aji']:.4f}  pq={results['pq']:.4f}")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Semi-supervised co-training")
    ap.add_argument("--dataset", default=cfg.datasets[0])
    ap.add_argument("--budget", type=int, default=16,
                    help="number of annotated images (K)")
    ap.add_argument("--seed", type=int, default=cfg.seeds[0])
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    train_semi(args.dataset, args.budget, args.seed, args.epochs)


if __name__ == "__main__":
    main()
