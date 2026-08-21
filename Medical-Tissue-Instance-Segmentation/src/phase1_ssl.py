"""Stage 1 — self-supervised adaptation.

Trains the adapters on unlabelled tissue, before any annotation is seen. Two
signals drive it: a latent anchor keeping the adapted features close to what the
frozen model already encodes, and the hematoxylin boundary target teaching them
where objects end.

Only the adapters survive this stage. The predictor and the masking machinery
are discarded.

Run:  python phase1_ssl.py
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import DEVICE, cfg, seed_everything
from data import UnlabelledTiles, load_glas
from losses import ssl_loss
from model import FrozenBackbone, Predictor, count_parameters


def sample_mask(batch: int, n_tokens: int, ratio: float, device) -> torch.Tensor:
    """Boolean mask marking the token positions the loss is computed at.

    All tokens stay in the forward pass. A stock vision transformer builds
    positional embeddings for all 196 positions before masking could happen, so
    dropping tokens would change the positional code of every token that
    remains. Restricting the loss instead keeps positions consistent, at the
    cost of a weaker masking signal.
    """
    n_masked = max(1, int(n_tokens * ratio))
    mask = torch.zeros(batch, n_tokens, dtype=torch.bool, device=device)
    for b in range(batch):
        idx = torch.randperm(n_tokens, device=device)[:n_masked]
        mask[b, idx] = True
    return mask


def train(images, epochs: int | None = None, batch_size: int | None = None,
          lr: float | None = None, seed: int = 0) -> str:
    epochs = epochs or cfg.ssl_epochs
    batch_size = batch_size or cfg.ssl_batch
    lr = lr or cfg.ssl_lr
    seed_everything(seed)

    print(f"[stage 1] device={DEVICE} epochs={epochs} batch={batch_size}")

    # the student carries adapters; the teacher is the untouched frozen model
    student = FrozenBackbone(adapt=True).to(DEVICE)
    teacher = FrozenBackbone(adapt=False).to(DEVICE).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    predictor = Predictor(cfg.embed_dim, n_tokens=cfg.n_tokens).to(DEVICE)

    total, trainable = count_parameters(student)
    print(f"[stage 1] adapted {student.n_adapted} projections")
    print(f"[stage 1] backbone {total / 1e6:.1f}M parameters, "
          f"{trainable / 1e6:.2f}M trainable ({100 * trainable / total:.2f}%)")

    params = ([p for p in student.parameters() if p.requires_grad]
              + list(predictor.parameters()))
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

    loader = DataLoader(UnlabelledTiles(images), batch_size=batch_size,
                        shuffle=True, num_workers=0, drop_last=True)

    for epoch in range(epochs):
        student.train()
        predictor.train()
        running, seen, started = {}, 0, time.time()

        for batch in loader:
            image = batch["image"].to(DEVICE)
            boundary = batch["boundary"].to(DEVICE)
            tissue = batch["tissue"].to(DEVICE)

            with torch.no_grad():
                target = teacher(image)[-1]                       # B x C x g x g
                b, c, g, _ = target.shape
                target = target.flatten(2).transpose(1, 2)        # B x N x C

            feats = student(image)[-1].flatten(2).transpose(1, 2)
            pred_emb, pred_bnd = predictor(feats)

            mask = sample_mask(image.size(0), target.size(1), cfg.mask_ratio, DEVICE)
            weight = tissue * mask.float()                        # tissue AND masked

            loss, parts = ssl_loss(pred_emb, pred_bnd, target, boundary, weight)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()

            for k, v in parts.items():
                running[k] = running.get(k, 0.0) + v
            seen += 1

        msg = "  ".join(f"{k}={running[k] / max(1, seen):.4f}" for k in sorted(running))
        print(f"[stage 1] epoch {epoch + 1}/{epochs}  {msg}  "
              f"({time.time() - started:.0f}s)", flush=True)

    os.makedirs(cfg.ssl_dir, exist_ok=True)
    out = os.path.join(cfg.ssl_dir, "adapter.pt")
    adapter_state = {k: v.cpu() for k, v in student.state_dict().items()
                     if any(t in k for t in (".A", ".B", ".m"))}
    torch.save({"adapter": adapter_state, "rank": cfg.adapter_rank,
                "alpha": cfg.adapter_alpha, "backbone": cfg.backbone}, out)
    print(f"[stage 1] adapter saved to {out} "
          f"({os.path.getsize(out) / 1e6:.1f} MB)")
    return out


def load_adapter(model: FrozenBackbone, path: str) -> int:
    """Load Stage-1 adapter weights into a fresh backbone."""
    blob = torch.load(path, map_location="cpu")
    missing = model.load_state_dict(blob["adapter"], strict=False)
    return len(blob["adapter"])


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 1: self-supervised adaptation")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="use only the first N images as the unlabelled pool")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # any tissue images work here; annotations are never read
    train_samples, _ = load_glas()
    if args.limit:
        train_samples = train_samples[:args.limit]
    images = [s.load()[0] for s in train_samples]
    print(f"[stage 1] unlabelled pool: {len(images)} images")

    train(images, epochs=args.epochs, batch_size=args.batch_size, seed=args.seed)


if __name__ == "__main__":
    main()
