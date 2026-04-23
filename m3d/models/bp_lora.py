"""LoRA injection for BiomedParse v2.

BP v2's internal modules use vanilla ``nn.Linear`` inside custom
``CrossAttentionLayer`` / ``FFNLayer`` / ``MSDeformAttn`` classes. The
HF ``peft`` library can target module *names* but BP's layers don't
register cleanly as transformer blocks, so we use a small manual LoRA
wrapper instead.

Typical usage from the training driver:

    from m3d.models.bp_lora import inject_lora, freeze_all, trainable_params

    freeze_all(bp_model)
    n = inject_lora(
        bp_model.sem_seg_head.predictor,
        target_patterns=("transformer_cross_attention_layers",),
        rank=32, alpha=64,
    )
    n += inject_lora(
        bp_model.sem_seg_head.pixel_decoder,
        target_patterns=("transformer",),  # MSDeformAttn Linears
        rank=16, alpha=32,
    )
    print(f"injected {n} LoRA layers")
"""
from __future__ import annotations

import re
from typing import Iterable

import torch
from torch import nn


class LoRALinear(nn.Module):
    """Low-rank adapter wrapping a frozen ``nn.Linear``.

    Forward: ``y = base(x) + (dropout(x) @ A^T @ B^T) * (alpha / rank)``.
    The base weight is frozen; only ``A`` and ``B`` are trainable.
    """

    def __init__(
        self,
        base: nn.Linear,
        rank: int = 16,
        alpha: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)

        in_f, out_f = base.in_features, base.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        # Match base weight's device/dtype so injection after model.to(device)
        # doesn't leave lora_A/B on CPU.
        dev, dt = base.weight.device, base.weight.dtype
        self.lora_A = nn.Parameter(torch.zeros(rank, in_f, device=dev, dtype=dt))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank, device=dev, dtype=dt))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B)  # LoRA starts as identity
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    # Defensive: PyTorch's nn.MultiheadAttention reads out_proj.weight directly
    # (bypassing forward) via F.multi_head_attention_forward. If inject_lora
    # accidentally wraps an MHA-owned Linear, these properties keep the model
    # from crashing — LoRA will simply be a no-op on that layer. The real fix
    # is to add "out_proj"/"in_proj" to exclude_patterns so MHA-internal
    # Linears stay unwrapped.
    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        delta = self.drop(x) @ self.lora_A.T @ self.lora_B.T
        return out + delta * self.scaling


def _iter_linear_names(
    root: nn.Module,
    path_prefix: str = "",
) -> Iterable[tuple[str, nn.Module, str]]:
    """Yield (full_path, parent_module, attr_name) for every nn.Linear."""
    for name, child in root.named_children():
        full = f"{path_prefix}.{name}" if path_prefix else name
        if isinstance(child, nn.Linear) and not isinstance(child, LoRALinear):
            yield full, root, name
        else:
            yield from _iter_linear_names(child, full)


def inject_lora(
    root: nn.Module,
    target_patterns: tuple[str, ...],
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.0,
    exclude_patterns: tuple[str, ...] = (),
) -> int:
    """Wrap every ``nn.Linear`` whose path matches ``target_patterns`` in LoRA.

    ``target_patterns`` / ``exclude_patterns`` are plain substrings (not
    regexes) matched against the dotted path under ``root``.

    Returns the number of layers wrapped.
    """
    pairs = list(_iter_linear_names(root))
    n = 0
    for path, parent, attr in pairs:
        if not any(p in path for p in target_patterns):
            continue
        if any(p in path for p in exclude_patterns):
            continue
        base = getattr(parent, attr)
        setattr(parent, attr, LoRALinear(base, rank=rank, alpha=alpha, dropout=dropout))
        n += 1
    return n


def freeze_all(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad_(False)


def unfreeze_prefixes(module: nn.Module, prefixes: tuple[str, ...]) -> int:
    """Unfreeze every parameter whose name starts with any prefix."""
    n = 0
    for name, p in module.named_parameters():
        if any(name.startswith(pr) or pr in name for pr in prefixes):
            p.requires_grad_(True)
            n += 1
    return n


def trainable_params(module: nn.Module) -> list[nn.Parameter]:
    return [p for p in module.parameters() if p.requires_grad]


def summarize_trainable(module: nn.Module, tag: str = "") -> str:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    pct = 100.0 * trainable / max(1, total)
    return (
        f"[{tag}] trainable={trainable:,}  total={total:,}  frac={pct:.2f}%"
    )
