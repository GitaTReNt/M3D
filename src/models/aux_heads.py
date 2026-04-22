"""Auxiliary heads attached to BiomedParse for Phase C.

All heads take the **pooled per-sample feature** (typically the mean of
the pixel_decoder multi-scale features, optionally concatenated with the
text class embedding) and predict a case-level 3D target.

We attach them at per-slice samples (the training regime). The network
is asked to infer 3D extent from what it sees in a single slice +
prompt. At inference time the heads can also be averaged across slices
of the same case to get a more stable 3D prediction.
"""
from __future__ import annotations

import torch
from torch import nn


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0) -> nn.Sequential:
    layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


class PhaseCAuxHeads(nn.Module):
    """Container module for the five aux heads.

    Outputs are **raw logits/regressions**. The loss module applies the
    sigmoid/softplus where appropriate.

    Args:
        feat_dim: dimensionality of the pooled feature fed in.
        hidden_dim: MLP hidden width.
        dropout: dropout inside each MLP (default 0.1).
    """

    def __init__(
        self,
        feat_dim: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.existence = _mlp(feat_dim, hidden_dim, 1, dropout)
        self.slice_exist = _mlp(feat_dim, hidden_dim, 1, dropout)
        self.bbox_3d = _mlp(feat_dim, hidden_dim, 6, dropout)
        self.centroid_3d = _mlp(feat_dim, hidden_dim, 3, dropout)
        self.z_range = _mlp(feat_dim, hidden_dim, 2, dropout)

    def forward(self, feat: torch.Tensor) -> dict:
        """feat: (B, feat_dim)."""
        out = {
            "existence_logit": self.existence(feat).squeeze(-1),  # (B,)
            "slice_exist_logit": self.slice_exist(feat).squeeze(-1),  # (B,)
            "bbox_3d_raw": self.bbox_3d(feat),  # (B, 6)
            "centroid_3d_raw": self.centroid_3d(feat),  # (B, 3)
            "z_range_raw": self.z_range(feat),  # (B, 2)
        }
        # Sigmoid-bound the coordinate regressions to [0, 1]
        out["bbox_3d"] = torch.sigmoid(out["bbox_3d_raw"])
        out["centroid_3d"] = torch.sigmoid(out["centroid_3d_raw"])
        out["z_range"] = torch.sigmoid(out["z_range_raw"])
        return out


def pool_features(
    multi_scale_features: list[torch.Tensor],
    class_emb: torch.Tensor | None = None,
) -> torch.Tensor:
    """Global-average pool each scale, concat, optionally concat text embedding.

    Args:
        multi_scale_features: list of (B*P, C_i, H_i, W_i) tensors from BP's
            pixel_decoder.forward_features output.
        class_emb: (B*P, D_text) optional text class embedding.

    Returns:
        (B*P, sum_i C_i [+ D_text]) pooled feature.
    """
    pooled = [f.mean(dim=(-2, -1)) for f in multi_scale_features]
    feat = torch.cat(pooled, dim=-1)
    if class_emb is not None:
        feat = torch.cat([feat, class_emb.to(feat.dtype)], dim=-1)
    return feat
