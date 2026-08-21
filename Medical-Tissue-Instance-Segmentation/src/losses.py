"""Loss functions for both training stages.

Stage 1 (self-supervised, no annotations):
    L = L_latent + mu * L_boundary

Stage 2 (supervised):
    L = w_dice * Dice + w_tversky * Tversky
        + w_boundary * (BCE_contour + Dice_contour)
      [ + distance-map loss on nuclei ]
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import cfg


# --------------------------------------------------------------------------- #
# Stage 1 — self-supervised                                                    #
# --------------------------------------------------------------------------- #
def latent_loss(pred: torch.Tensor, target: torch.Tensor,
                weight: torch.Tensor | None = None) -> torch.Tensor:
    """Smooth-L1 against the frozen teacher's layer-normalised tokens.

    The teacher is fixed, not an exponential average of the student. That is
    what stops the adapted features drifting away from the pretrained
    representation, and it removes the collapse-control machinery an averaged
    teacher would need.
    """
    target = F.layer_norm(target, (target.size(-1),))
    loss = F.smooth_l1_loss(pred, target, beta=1.0, reduction="none").mean(-1)
    if weight is None:
        return loss.mean()
    w = weight.clamp_min(1e-6)
    return (loss * w).sum() / w.sum()


def boundary_loss(pred: torch.Tensor, target: torch.Tensor,
                  weight: torch.Tensor | None = None,
                  pos_weight: float | None = None) -> torch.Tensor:
    """Weighted BCE against the hematoxylin boundary target.

    ``weight`` is the per-token tissue mask, so blank slide contributes nothing.
    ``pos_weight`` compensates for boundary tokens being the minority class.
    """
    pw = torch.tensor(cfg.boundary_pos_weight if pos_weight is None else pos_weight,
                      device=pred.device, dtype=pred.dtype)
    loss = F.binary_cross_entropy_with_logits(pred, target, pos_weight=pw,
                                              reduction="none")
    if weight is None:
        return loss.mean()
    w = weight.clamp_min(1e-6)
    return (loss * w).sum() / w.sum()


def ssl_loss(pred_emb, pred_bnd, target_emb, target_bnd, tissue_weight=None,
             mu: float | None = None):
    """Total Stage-1 objective. Returns (loss, components)."""
    mu = cfg.mu_boundary if mu is None else mu
    l_latent = latent_loss(pred_emb, target_emb, tissue_weight)
    l_bnd = boundary_loss(pred_bnd, target_bnd, tissue_weight)
    total = l_latent + mu * l_bnd
    return total, {"latent": float(l_latent.detach()),
                   "boundary": float(l_bnd.detach()),
                   "total": float(total.detach())}


# --------------------------------------------------------------------------- #
# Stage 2 — supervised                                                         #
# --------------------------------------------------------------------------- #
class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        num = 2.0 * (p * target).sum() + self.eps
        den = p.sum() + target.sum() + self.eps
        return 1.0 - num / den


class TverskyLoss(nn.Module):
    """Dice with the two error types weighted separately.

        TI = TP / (TP + alpha * FP + beta * FN)

    ``alpha`` charges false alarms, ``beta`` charges misses. On nuclei
    alpha = 0.3 makes a missed nucleus cost more than a spurious one, which is
    the right trade when an image holds several hundred small objects.
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.5, eps: float = 1e-6) -> None:
        super().__init__()
        self.alpha, self.beta, self.eps = alpha, beta, eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        tp = (p * target).sum()
        fp = (p * (1 - target)).sum()
        fn = ((1 - p) * target).sum()
        ti = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        return 1.0 - ti


class SupervisedLoss(nn.Module):
    """Region overlap, imbalance-aware overlap, and direct contour supervision."""

    def __init__(self, tversky_alpha: float = 0.5, w_dice: float | None = None,
                 w_tversky: float | None = None, w_boundary: float | None = None,
                 boundary_pos_weight: float | None = None) -> None:
        super().__init__()
        self.w_dice = cfg.w_dice if w_dice is None else w_dice
        self.w_tversky = cfg.w_tversky if w_tversky is None else w_tversky
        self.w_boundary = cfg.w_boundary if w_boundary is None else w_boundary

        self.dice = DiceLoss()
        self.tversky = TverskyLoss(tversky_alpha, 1.0 - tversky_alpha)
        self.bnd_dice = DiceLoss()
        pw = cfg.boundary_pos_weight if boundary_pos_weight is None else boundary_pos_weight
        self.register_buffer("pos_weight", torch.tensor(pw))

    def forward(self, sem_logits, bnd_logits, sem_gt, bnd_gt) -> torch.Tensor:
        loss = (self.w_dice * self.dice(sem_logits, sem_gt)
                + self.w_tversky * self.tversky(sem_logits, sem_gt))
        bce = F.binary_cross_entropy_with_logits(
            bnd_logits, bnd_gt, pos_weight=self.pos_weight.to(bnd_logits.device))
        loss = loss + self.w_boundary * (bce + self.bnd_dice(bnd_logits, bnd_gt))
        return loss


def distance_loss(pred_hv: torch.Tensor, target_hv: torch.Tensor,
                  foreground: torch.Tensor,
                  grad_weight: float | None = None) -> torch.Tensor:
    """Match the distance maps and their gradients.

    The gradient term is the one that matters. The maps reverse sign across the
    join between two nuclei, so their gradient spikes exactly on that join —
    charging the model to get the gradient right is what sharpens the ridge the
    watershed later floods from. It is restricted to the foreground, since the
    background carries no object to separate.
    """
    gw = cfg.hv_grad_weight if grad_weight is None else grad_weight
    mse = ((pred_hv - target_hv) ** 2).mean()

    def spatial_grad(t: torch.Tensor) -> torch.Tensor:
        gx = t[..., :, 1:] - t[..., :, :-1]
        gy = t[..., 1:, :] - t[..., :-1, :]
        return (F.pad(gx, (0, 1, 0, 0)), F.pad(gy, (0, 0, 0, 1)))

    pgx, pgy = spatial_grad(pred_hv)
    tgx, tgy = spatial_grad(target_hv)
    f = foreground.clamp_min(0.0)
    msge = ((f * (pgx - tgx) ** 2).sum() + (f * (pgy - tgy) ** 2).sum()) / (f.sum() + 1e-8)
    return mse + gw * msge


# --------------------------------------------------------------------------- #
# Semi-supervised consistency                                                  #
# --------------------------------------------------------------------------- #
def consistency_loss(student_logits: torch.Tensor,
                     teacher_logits: torch.Tensor) -> torch.Tensor:
    """Agreement between the student on a perturbed image and the teacher on the
    clean one, measured on probabilities rather than logits."""
    return F.mse_loss(torch.sigmoid(student_logits), torch.sigmoid(teacher_logits))


def rampup(epoch: int, length: int) -> float:
    """Gaussian ramp from 0 to 1 over ``length`` epochs.

    The consistency term stays near zero while the teacher is still unreliable,
    which prevents it from reinforcing its own early mistakes.
    """
    if length <= 0:
        return 1.0
    x = min(max(epoch / float(length), 0.0), 1.0)
    return float(torch.exp(torch.tensor(-5.0 * (1.0 - x) ** 2)))


@torch.no_grad()
def update_teacher(teacher: torch.nn.Module, student: torch.nn.Module,
                   decay: float | None = None) -> None:
    """Teacher weights follow the student as an exponential moving average."""
    d = cfg.ema_decay if decay is None else decay
    for tp, sp in zip(teacher.parameters(), student.parameters()):
        tp.mul_(d).add_(sp.detach(), alpha=1.0 - d)
    for tb, sb in zip(teacher.buffers(), student.buffers()):
        tb.copy_(sb)
