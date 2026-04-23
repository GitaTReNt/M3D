"""Standard-definition LoRA for nn.MultiheadAttention.

PyTorch's ``nn.MultiheadAttention`` packs Q/K/V into a single parameter
``in_proj_weight`` and reads ``out_proj.weight`` directly inside
``F.multi_head_attention_forward``, so wrapping individual ``nn.Linear``
children with a LoRA adapter is a no-op on the attention math.

This module provides ``LoRAMultiheadAttention``, a drop-in replacement
that:

- splits the original packed weight into four independent
  ``nn.Linear`` projections (Q, K, V, O), each frozen,
- wraps the selected projections (default: Q and V, per Hu et al. 2021)
  in ``LoRALinear``,
- reimplements the attention forward via
  ``F.scaled_dot_product_attention`` so the base behaviour is
  numerically identical to the source MHA when LoRA deltas are zero
  (they are at init, because ``B`` is zero-initialised).

**Scope**: we only reproduce the subset of ``nn.MultiheadAttention``'s
interface actually used by BiomedParse, namely ``batch_first=False``,
same-dim K/V, no ``add_bias_kv`` / ``add_zero_attn``. Anything outside
that scope asserts out loud in ``from_mha``.

Callers typically use ``replace_mha_with_lora`` in ``bp_lora.py`` to
walk a subtree and swap instances.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from m3d.models.bp_lora import LoRALinear


class LoRAMultiheadAttention(nn.Module):
    """Drop-in ``nn.MultiheadAttention`` replacement with LoRA on Q/V.

    Args:
        embed_dim: total embedding dim (d_model).
        num_heads: number of attention heads.
        dropout: attention dropout probability (applied inside SDPA during
            training, matches ``nn.MultiheadAttention``'s ``dropout`` arg).
        bias: include bias on Q/K/V/O projections.
        lora_rank, lora_alpha: standard LoRA hyperparameters.
        lora_targets: which projections to LoRA. Subset of
            ``{"q", "k", "v", "o"}``. Default ``("q", "v")`` matches the
            original LoRA paper's recommendation for transformers.
        lora_dropout: dropout on the LoRA branch input (0 by default —
            with only 150 train cases we don't want more regularisation).

    The frozen base behaves **exactly** like an ``nn.MultiheadAttention``
    with the source weights because of how LoRA is initialised
    (``B`` starts at zero, so ``ΔW = B A = 0``).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_targets: tuple[str, ...] = ("q", "v"),
        lora_dropout: float = 0.0,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim {embed_dim} must divide num_heads {num_heads}")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout_p = float(dropout)

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Freeze all base projection params by default; LoRALinear flips
        # only the adapter params to requires_grad=True.
        for p in self.parameters():
            p.requires_grad_(False)

        proj_map = {"q": "q_proj", "k": "k_proj", "v": "v_proj", "o": "out_proj"}
        invalid = set(lora_targets) - proj_map.keys()
        if invalid:
            raise ValueError(f"unknown lora_targets: {invalid}")
        for t in lora_targets:
            attr = proj_map[t]
            base = getattr(self, attr)
            setattr(
                self,
                attr,
                LoRALinear(
                    base, rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout,
                ),
            )

    # --- factory from stock MHA ------------------------------------------

    @staticmethod
    def _base(proj) -> nn.Linear:
        """Return the frozen base Linear regardless of LoRA wrapping."""
        return proj.base if isinstance(proj, LoRALinear) else proj

    @classmethod
    def from_mha(
        cls,
        mha: nn.MultiheadAttention,
        rank: int = 16,
        alpha: int = 32,
        targets: tuple[str, ...] = ("q", "v"),
        lora_dropout: float = 0.0,
    ) -> "LoRAMultiheadAttention":
        """Build an instance and copy weights from an existing MHA."""
        assert mha._qkv_same_embed_dim, "kdim/vdim != embed_dim not supported"
        # `add_bias_kv` is exposed via bias_k/bias_v parameters rather than
        # a flag on recent torch; treat either set as unsupported.
        assert mha.bias_k is None and mha.bias_v is None, "add_bias_kv not supported"
        assert not mha.add_zero_attn, "add_zero_attn not supported"
        assert not mha.batch_first, "batch_first=True not supported (BP uses False)"

        d = mha.embed_dim
        nh = mha.num_heads
        dp = float(mha.dropout)
        bias = mha.in_proj_bias is not None

        new = cls(
            embed_dim=d, num_heads=nh, dropout=dp, bias=bias,
            lora_rank=rank, lora_alpha=alpha, lora_targets=targets,
            lora_dropout=lora_dropout,
        )
        new = new.to(device=mha.in_proj_weight.device, dtype=mha.in_proj_weight.dtype)

        with torch.no_grad():
            W = mha.in_proj_weight  # (3d, d)
            cls._base(new.q_proj).weight.copy_(W[:d])
            cls._base(new.k_proj).weight.copy_(W[d : 2 * d])
            cls._base(new.v_proj).weight.copy_(W[2 * d :])
            if bias:
                b = mha.in_proj_bias  # (3d,)
                cls._base(new.q_proj).bias.copy_(b[:d])
                cls._base(new.k_proj).bias.copy_(b[d : 2 * d])
                cls._base(new.v_proj).bias.copy_(b[2 * d :])
            cls._base(new.out_proj).weight.copy_(mha.out_proj.weight)
            if mha.out_proj.bias is not None and bias:
                cls._base(new.out_proj).bias.copy_(mha.out_proj.bias)
        return new

    # --- forward matching nn.MultiheadAttention's signature --------------

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> tuple[Tensor, Optional[Tensor]]:
        """See ``nn.MultiheadAttention.forward``.

        Shapes (batch_first=False):
            query: (L, N, E)  key/value: (S, N, E)
            attn_mask: (L, S) or (N*H, L, S), bool or float
            key_padding_mask: (N, S), bool where True = ignore
        Returns:
            (attn_output of shape (L, N, E), None)  — attention weights
            are not materialised (SDPA is fused and hides them); BP never
            consumes the weights downstream.
        """
        assert not is_causal, "is_causal not used by BP; disabled"
        L, N, E = query.shape
        S = key.shape[0]
        H = self.num_heads
        D = self.head_dim

        # Project + reshape to (N, H, L/S, D) for SDPA
        Q = self.q_proj(query).reshape(L, N, H, D).permute(1, 2, 0, 3)
        K = self.k_proj(key).reshape(S, N, H, D).permute(1, 2, 0, 3)
        V = self.v_proj(value).reshape(S, N, H, D).permute(1, 2, 0, 3)

        # Build a single additive-float bias combining attn_mask +
        # key_padding_mask, broadcastable to (N, H, L, S). SDPA's
        # ``attn_mask`` arg accepts bool or float; we normalise to float
        # so combining two masks is just addition.
        bias: Optional[Tensor] = None
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                am = torch.zeros_like(attn_mask, dtype=Q.dtype)
                am = am.masked_fill(attn_mask, float("-inf"))
            else:
                am = attn_mask.to(Q.dtype)
            if am.dim() == 2:           # (L, S)
                am = am.view(1, 1, L, S)
            elif am.dim() == 3:         # (N*H, L, S)
                am = am.view(N, H, L, S)
            bias = am
        if key_padding_mask is not None:
            kpm = key_padding_mask.view(N, 1, 1, S)
            kpm_f = torch.zeros_like(kpm, dtype=Q.dtype)
            kpm_f = kpm_f.masked_fill(kpm.to(torch.bool), float("-inf"))
            bias = kpm_f if bias is None else bias + kpm_f

        out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=bias,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        # (N, H, L, D) -> (L, N, E)
        out = out.permute(2, 0, 1, 3).reshape(L, N, E)
        out = self.out_proj(out)
        return out, None


def replace_mha_with_lora(
    root: nn.Module,
    target_patterns: tuple[str, ...],
    rank: int = 16,
    alpha: int = 32,
    targets: tuple[str, ...] = ("q", "v"),
    lora_dropout: float = 0.0,
    exclude_patterns: tuple[str, ...] = (),
) -> int:
    """Walk ``root``'s module tree and replace matching ``nn.MultiheadAttention``
    instances with ``LoRAMultiheadAttention`` initialised from the originals.

    ``target_patterns`` / ``exclude_patterns`` are plain substrings matched
    against the dotted module path. Returns the number of layers replaced.
    """
    replaced = 0
    for name, module in list(root.named_modules()):
        if not isinstance(module, nn.MultiheadAttention):
            continue
        if not any(p in name for p in target_patterns):
            continue
        if any(p in name for p in exclude_patterns):
            continue

        # Locate parent and attribute name for re-binding
        if "." in name:
            parent_path, attr = name.rsplit(".", 1)
            parent = root.get_submodule(parent_path)
        else:
            parent, attr = root, name
        new = LoRAMultiheadAttention.from_mha(
            module, rank=rank, alpha=alpha, targets=targets, lora_dropout=lora_dropout,
        )
        setattr(parent, attr, new)
        replaced += 1
    return replaced
