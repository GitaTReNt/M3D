"""Multi-task loss for Phase C.

Composition (see docs/phase_c_plan.md §5 / docs/bp_medsam_tuning.md §4):

    L_total = L_main                                 # Dice + BCE on pred_gmask
            + lam_ex  * L_existence                  # BCE
            + lam_sex * L_slice_exist                # BCE (per sample)
            + lam_bb  * (L_bbox_l1 + L_bbox_giou)    # smoothL1 + GIoU (masked)
            + lam_ct  * L_centroid                   # L2 (masked)
            + lam_zr  * L_zrange                     # smoothL1 (masked)
            + lam_dist * L_distill                   # optional scorer KL

The bbox / centroid / z_range terms are **masked** by the per-sample
``existence`` flag so we do not push the heads toward learning bbox
coordinates on cases where the target does not exist.

We deliberately keep ``L_main`` standalone and compute it on the raw
pred_gmask tensor produced by BP, so the main-line loss stays identical
to BP's original Dice+BCE even if we later swap out the rest.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


# ---------------------------------------------------------------------------
# segmentation loss (L_main)
# ---------------------------------------------------------------------------

def soft_dice_loss(
    pred_logit: torch.Tensor,
    gt: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Multi-class-safe binary soft Dice.

    pred_logit, gt: (B, H, W) or (B, 1, H, W). Returns scalar.
    """
    if pred_logit.dim() == 4:
        pred_logit = pred_logit.squeeze(1)
    if gt.dim() == 4:
        gt = gt.squeeze(1)
    prob = torch.sigmoid(pred_logit)
    num = 2 * (prob * gt).sum(dim=(-1, -2))
    den = prob.sum(dim=(-1, -2)) + gt.sum(dim=(-1, -2)) + eps
    dice = num / den
    return 1.0 - dice.mean()


def bce_loss(
    pred_logit: torch.Tensor,
    gt: torch.Tensor,
) -> torch.Tensor:
    if pred_logit.dim() == 4:
        pred_logit = pred_logit.squeeze(1)
    if gt.dim() == 4:
        gt = gt.squeeze(1)
    return F.binary_cross_entropy_with_logits(pred_logit, gt)


def main_seg_loss(pred_logit: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
    return 0.5 * soft_dice_loss(pred_logit, gt_mask) + 0.5 * bce_loss(pred_logit, gt_mask)


# ---------------------------------------------------------------------------
# bbox / GIoU (3D boxes, corner-normalised to [0,1])
# ---------------------------------------------------------------------------

def _giou_3d(
    pred_box: torch.Tensor,
    gt_box: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Generalised IoU for 3D axis-aligned boxes.

    Boxes stored as (z1, y1, x1, z2, y2, x2). Coordinates in [0,1].
    Returns (B,) 1 - GIoU loss.
    """
    pz1, py1, px1, pz2, py2, px2 = pred_box.unbind(-1)
    gz1, gy1, gx1, gz2, gy2, gx2 = gt_box.unbind(-1)

    # Intersection
    iz1 = torch.max(pz1, gz1); iy1 = torch.max(py1, gy1); ix1 = torch.max(px1, gx1)
    iz2 = torch.min(pz2, gz2); iy2 = torch.min(py2, gy2); ix2 = torch.min(px2, gx2)
    inter = (iz2 - iz1).clamp(min=0) * (iy2 - iy1).clamp(min=0) * (ix2 - ix1).clamp(min=0)

    pv = (pz2 - pz1).clamp(min=0) * (py2 - py1).clamp(min=0) * (px2 - px1).clamp(min=0)
    gv = (gz2 - gz1).clamp(min=0) * (gy2 - gy1).clamp(min=0) * (gx2 - gx1).clamp(min=0)
    union = pv + gv - inter + eps
    iou = inter / union

    # Enclosing box
    ez1 = torch.min(pz1, gz1); ey1 = torch.min(py1, gy1); ex1 = torch.min(px1, gx1)
    ez2 = torch.max(pz2, gz2); ey2 = torch.max(py2, gy2); ex2 = torch.max(px2, gx2)
    enc = (ez2 - ez1).clamp(min=0) * (ey2 - ey1).clamp(min=0) * (ex2 - ex1).clamp(min=0) + eps

    giou = iou - (enc - union) / enc
    return 1.0 - giou


def bbox_loss(
    pred_bbox: torch.Tensor,
    gt_bbox: torch.Tensor,
    mask: torch.Tensor,
    w_l1: float = 1.0,
    w_giou: float = 1.0,
) -> torch.Tensor:
    """Masked smooth-L1 + GIoU on (B, 6) bboxes.

    mask: (B,) float 0/1, applied per-sample.
    """
    if mask.sum() < 0.5:
        return pred_bbox.sum() * 0.0  # keep graph
    l1 = F.smooth_l1_loss(pred_bbox, gt_bbox, reduction="none").mean(dim=-1)
    giou = _giou_3d(pred_bbox, gt_bbox)
    per_sample = w_l1 * l1 + w_giou * giou
    return (per_sample * mask).sum() / mask.sum().clamp_min(1.0)


# ---------------------------------------------------------------------------
# composite phase-C loss
# ---------------------------------------------------------------------------

@dataclass
class PhaseCLossWeights:
    # Defaults tuned for the BP→MedSAM axis (docs/bp_medsam_tuning.md §4):
    # slice_exist and bbox are the top levers; centroid+zrange are regularisers.
    lam_ex: float = 0.3
    lam_sex: float = 0.5
    lam_bb: float = 0.3
    lam_ct: float = 0.1
    lam_zr: float = 0.2
    lam_dist: float = 0.0  # optional scorer KL — off unless enabled
    ramp_epochs: int = 2   # linear warmup on all aux λ's


class PhaseCLoss(nn.Module):
    def __init__(self, weights: PhaseCLossWeights | None = None):
        super().__init__()
        self.w = weights or PhaseCLossWeights()

    def _ramp(self, epoch: int) -> float:
        if self.w.ramp_epochs <= 0:
            return 1.0
        return min(1.0, (epoch + 1) / self.w.ramp_epochs)

    def forward(
        self,
        pred_seg_logit: torch.Tensor,
        gt_mask: torch.Tensor,
        aux_out: dict[str, torch.Tensor],
        aux_gt: dict[str, torch.Tensor],
        existence_mask: torch.Tensor,
        scorer_prob: torch.Tensor | None = None,
        epoch: int = 0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        logs: dict[str, float] = {}

        L_main = main_seg_loss(pred_seg_logit, gt_mask)
        logs["loss/main"] = float(L_main.detach())

        L_ex = F.binary_cross_entropy_with_logits(
            aux_out["existence_logit"], aux_gt["existence"]
        )
        L_sex = F.binary_cross_entropy_with_logits(
            aux_out["slice_exist_logit"], aux_gt["slice_exist"]
        )
        L_bb = bbox_loss(
            aux_out["bbox_3d"], aux_gt["bbox_3d"], existence_mask,
        )
        L_ct = F.mse_loss(
            aux_out["centroid_3d"], aux_gt["centroid_3d"], reduction="none",
        ).mean(dim=-1)
        L_ct = (L_ct * existence_mask).sum() / existence_mask.sum().clamp_min(1.0)
        L_zr = F.smooth_l1_loss(
            aux_out["z_range"], aux_gt["z_range"], reduction="none",
        ).mean(dim=-1)
        L_zr = (L_zr * existence_mask).sum() / existence_mask.sum().clamp_min(1.0)

        logs["loss/existence"] = float(L_ex.detach())
        logs["loss/slice_exist"] = float(L_sex.detach())
        logs["loss/bbox"] = float(L_bb.detach())
        logs["loss/centroid"] = float(L_ct.detach())
        logs["loss/zrange"] = float(L_zr.detach())

        ramp = self._ramp(epoch)
        L_total = (
            L_main
            + ramp * self.w.lam_ex * L_ex
            + ramp * self.w.lam_sex * L_sex
            + ramp * self.w.lam_bb * L_bb
            + ramp * self.w.lam_ct * L_ct
            + ramp * self.w.lam_zr * L_zr
        )

        if scorer_prob is not None and self.w.lam_dist > 0:
            # KL( stop_grad(scorer) || sigmoid(slice_exist_logit) )
            q = torch.sigmoid(aux_out["slice_exist_logit"]).clamp(1e-6, 1 - 1e-6)
            p = scorer_prob.detach().clamp(1e-6, 1 - 1e-6)
            L_dist = (p * (p.log() - q.log()) + (1 - p) * ((1 - p).log() - (1 - q).log())).mean()
            L_total = L_total + ramp * self.w.lam_dist * L_dist
            logs["loss/distill"] = float(L_dist.detach())

        logs["loss/total"] = float(L_total.detach())
        logs["loss/ramp"] = ramp
        return L_total, logs
