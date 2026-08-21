"""Instance extraction and evaluation.

Semantic metrics (Dice, IoU) say how well tissue pixels were found. Instance
metrics say whether individual objects were separated. A model can score well on
the first while merging every pair of touching objects, which is exactly the
failure this project set out to fix — so both are always reported.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import scipy.ndimage as ndi
from skimage.segmentation import watershed

from config import cfg


# --------------------------------------------------------------------------- #
# Instance extraction                                                          #
# --------------------------------------------------------------------------- #
def instances_from_contour(sem_prob: np.ndarray, bnd_prob: np.ndarray,
                           sem_thr: float | None = None, bnd_thr: float | None = None,
                           min_size: int | None = None) -> np.ndarray:
    """Glands: flood inward from foreground that is confidently away from an edge.

    Large objects separated by thin membranes, so a predicted contour carries
    enough information on its own.
    """
    sem_thr = cfg.sem_thr_gland if sem_thr is None else sem_thr
    bnd_thr = cfg.bnd_thr_gland if bnd_thr is None else bnd_thr
    min_size = cfg.min_size_gland if min_size is None else min_size

    foreground = sem_prob > sem_thr
    if not foreground.any():
        return np.zeros(sem_prob.shape, np.int32)

    interior = ndi.binary_opening(foreground & (bnd_prob < bnd_thr),
                                  structure=np.ones((3, 3)))
    markers, n = ndi.label(interior)
    if n == 0:
        markers, n = ndi.label(foreground)
        if n == 0:
            return np.zeros(sem_prob.shape, np.int32)

    inst = watershed(bnd_prob, markers=markers, mask=foreground)
    return _drop_small(inst.astype(np.int32), min_size)


def instances_from_distance(sem_prob: np.ndarray, hv: np.ndarray,
                            marker_thr: float | None = None,
                            min_size: int | None = None) -> np.ndarray:
    """Nuclei: split on the ridges of the distance-map gradient.

    Each map runs from -1 to +1 across an object, so it reverses sign at the
    join between neighbours and its gradient spikes there. Flooding
    ``1 - |grad|`` from confident interior seeds separates objects that a
    contour alone would fuse.
    """
    marker_thr = cfg.hv_marker_thr if marker_thr is None else marker_thr
    min_size = cfg.hv_min_size if min_size is None else min_size

    foreground = sem_prob > 0.5
    if not foreground.any():
        return np.zeros(sem_prob.shape, np.int32)

    gx = np.abs(np.gradient(hv[0], axis=1))
    gy = np.abs(np.gradient(hv[1], axis=0))
    magnitude = np.maximum(gx, gy)
    rng = magnitude.max() - magnitude.min()
    magnitude = (magnitude - magnitude.min()) / (rng + 1e-8)

    seeds = foreground & (magnitude < marker_thr)
    seeds = ndi.binary_opening(seeds, structure=np.ones((3, 3)))
    markers, n = ndi.label(seeds)
    if n == 0:
        markers, n = ndi.label(foreground)
        if n == 0:
            return np.zeros(sem_prob.shape, np.int32)

    inst = watershed(magnitude, markers=markers, mask=foreground)
    return _drop_small(inst.astype(np.int32), min_size)


def _drop_small(inst: np.ndarray, min_size: int) -> np.ndarray:
    """Remove specks and relabel consecutively."""
    out = np.zeros_like(inst)
    nxt = 1
    for lab in np.unique(inst):
        if lab == 0:
            continue
        m = inst == lab
        if m.sum() >= min_size:
            out[m] = nxt
            nxt += 1
    return out


def hv_targets(inst: np.ndarray) -> np.ndarray:
    """Ground-truth distance maps: for every pixel, how far it lies from its own
    object's centre in x and y, normalised to [-1, 1] within each object."""
    hx = np.zeros(inst.shape, np.float32)
    hy = np.zeros(inst.shape, np.float32)
    for lab in np.unique(inst):
        if lab == 0:
            continue
        ys, xs = np.nonzero(inst == lab)
        dx = xs.astype(np.float32) - xs.mean()
        dy = ys.astype(np.float32) - ys.mean()
        for d in (dx, dy):                      # scale each side independently
            neg, pos = d < 0, d > 0
            if neg.any():
                d[neg] /= -d[neg].min()
            if pos.any():
                d[pos] /= d[pos].max()
        hx[ys, xs] = dx
        hy[ys, xs] = dy
    return np.stack([hx, hy], axis=0)


# --------------------------------------------------------------------------- #
# Metrics                                                                      #
# --------------------------------------------------------------------------- #
def dice_score(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    p, g = pred.astype(bool), gt.astype(bool)
    return float((2.0 * (p & g).sum() + eps) / (p.sum() + g.sum() + eps))


def iou_score(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    p, g = pred.astype(bool), gt.astype(bool)
    return float(((p & g).sum() + eps) / ((p | g).sum() + eps))


def _pairwise(gt_inst: np.ndarray, pred_inst: np.ndarray):
    """Intersection and union between every ground-truth and predicted object."""
    gt_ids = [i for i in np.unique(gt_inst) if i != 0]
    pred_ids = [i for i in np.unique(pred_inst) if i != 0]
    gt_masks = {i: gt_inst == i for i in gt_ids}
    pred_masks = {j: pred_inst == j for j in pred_ids}

    inter = np.zeros((len(gt_ids), len(pred_ids)), np.float64)
    union = np.zeros_like(inter)
    for a, i in enumerate(gt_ids):
        gm = gt_masks[i]
        overlapping = np.unique(pred_inst[gm])
        for b, j in enumerate(pred_ids):
            if j not in overlapping:
                continue
            pm = pred_masks[j]
            v = float((gm & pm).sum())
            inter[a, b] = v
            union[a, b] = gm.sum() + pm.sum() - v
    return gt_ids, pred_ids, gt_masks, pred_masks, inter, union


def aggregated_jaccard(gt_inst: np.ndarray, pred_inst: np.ndarray) -> float:
    """Aggregated Jaccard Index.

    Each ground-truth object is matched to the prediction it overlaps most, and
    matched intersections and unions accumulate globally. Every predicted object
    that matched nothing is added to the denominator, so merging two objects and
    splitting one are both penalised.
    """
    if gt_inst.max() == 0:
        return 1.0 if pred_inst.max() == 0 else 0.0
    if pred_inst.max() == 0:
        return 0.0

    gt_ids, pred_ids, gt_masks, pred_masks, inter, union = _pairwise(gt_inst, pred_inst)
    if not pred_ids:
        return 0.0

    iou = inter / (union + 1e-6)
    best = np.argmax(iou, axis=1)
    matched = np.max(iou, axis=1) > 0.0

    total_inter = float(inter[np.arange(len(gt_ids))[matched], best[matched]].sum())
    total_union = float(union[np.arange(len(gt_ids))[matched], best[matched]].sum())

    for a, i in enumerate(gt_ids):              # ground truth with no overlap
        if not matched[a]:
            total_union += float(gt_masks[i].sum())
    used = set(best[matched].tolist())
    for b, j in enumerate(pred_ids):            # predictions matching nothing
        if b not in used:
            total_union += float(pred_masks[j].sum())

    return float(total_inter / (total_union + 1e-6))


def panoptic_quality(gt_inst: np.ndarray, pred_inst: np.ndarray,
                     match_iou: float = 0.5) -> Dict[str, float]:
    """Panoptic Quality, split into detection and segmentation quality.

        PQ = DQ x SQ,  DQ = TP / (TP + FP/2 + FN/2),  SQ = mean IoU over matches

    A match needs IoU above 0.5, which guarantees it is unique.
    """
    if gt_inst.max() == 0 and pred_inst.max() == 0:
        return {"dq": 1.0, "sq": 1.0, "pq": 1.0}
    if gt_inst.max() == 0 or pred_inst.max() == 0:
        return {"dq": 0.0, "sq": 0.0, "pq": 0.0}

    gt_ids, pred_ids, _, _, inter, union = _pairwise(gt_inst, pred_inst)
    iou = inter / (union + 1e-6)

    pairs = np.argwhere(iou > match_iou)
    matched_iou, used_g, used_p = [], set(), set()
    for k in np.argsort(-iou[pairs[:, 0], pairs[:, 1]]) if len(pairs) else []:
        g, p = pairs[k]
        if g in used_g or p in used_p:
            continue
        used_g.add(g)
        used_p.add(p)
        matched_iou.append(iou[g, p])

    tp = len(matched_iou)
    fp = len(pred_ids) - tp
    fn = len(gt_ids) - tp
    dq = tp / (tp + 0.5 * fp + 0.5 * fn + 1e-6)
    sq = float(np.sum(matched_iou) / (tp + 1e-6)) if tp else 0.0
    return {"dq": float(dq), "sq": sq, "pq": float(dq * sq)}


def object_dice(gt_inst: np.ndarray, pred_inst: np.ndarray) -> float:
    """Area-weighted Dice over matched objects, averaged in both directions."""
    def directed(a: np.ndarray, b: np.ndarray) -> float:
        ids = [i for i in np.unique(a) if i != 0]
        total = sum((a == i).sum() for i in ids)
        if total == 0:
            return 0.0
        acc = 0.0
        for i in ids:
            am = a == i
            overlapping = [j for j in np.unique(b[am]) if j != 0]
            best = max((dice_score(b == j, am) for j in overlapping), default=0.0)
            acc += (am.sum() / total) * best
        return acc
    return 0.5 * (directed(gt_inst, pred_inst) + directed(pred_inst, gt_inst))


def object_f1(gt_inst: np.ndarray, pred_inst: np.ndarray,
              match_iou: float = 0.5) -> float:
    """Detection F1: how many objects were found, ignoring outline quality."""
    pq = panoptic_quality(gt_inst, pred_inst, match_iou)
    return pq["dq"]


def evaluate(gt_inst: np.ndarray, pred_inst: np.ndarray) -> Dict[str, float]:
    """All metrics for one image."""
    gt_sem = gt_inst > 0
    pred_sem = pred_inst > 0
    pq = panoptic_quality(gt_inst, pred_inst)
    return {
        "dice": dice_score(pred_sem, gt_sem),
        "iou": iou_score(pred_sem, gt_sem),
        "object_dice": object_dice(gt_inst, pred_inst),
        "object_f1": object_f1(gt_inst, pred_inst),
        "aji": aggregated_jaccard(gt_inst, pred_inst),
        "pq": pq["pq"],
        "dq": pq["dq"],
        "sq": pq["sq"],
        "n_pred": int(len([i for i in np.unique(pred_inst) if i != 0])),
        "n_gt": int(len([i for i in np.unique(gt_inst) if i != 0])),
    }


def aggregate(rows) -> Dict[str, float]:
    """Mean of per-image metrics."""
    if not rows:
        return {}
    keys = rows[0].keys()
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}
