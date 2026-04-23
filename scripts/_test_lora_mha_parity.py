"""Parity test: LoRAMultiheadAttention at init must match nn.MultiheadAttention.

Because LoRA's B matrix is zero-initialised, ΔW = B @ A = 0, and the
LoRA-wrapped attention must produce **identical** outputs to the source
``nn.MultiheadAttention`` (up to fp32 rounding) before any training.

Run on the server:

    cd /root/autodl-tmp/m3d-finetune
    /root/miniconda3/envs/d2/bin/python scripts/_test_lora_mha_parity.py

This is a pure CPU test — no BP weights needed. If it doesn't pass,
``replace_mha_with_lora`` has a bug and cross-attn training would
silently corrupt the model.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from m3d.models.lora_mha import LoRAMultiheadAttention


def parity_once(embed_dim: int, num_heads: int, L: int, S: int, N: int,
                seed: int, with_masks: bool) -> tuple[float, float]:
    torch.manual_seed(seed)
    src = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0, bias=True)
    src.eval()
    new = LoRAMultiheadAttention.from_mha(
        src, rank=16, alpha=32, targets=("q", "v"),
    )
    new.eval()

    q = torch.randn(L, N, embed_dim)
    k = torch.randn(S, N, embed_dim)
    v = torch.randn(S, N, embed_dim)

    attn_mask = None
    kpm = None
    if with_masks:
        # boolean attn_mask covering some positions
        attn_mask = torch.zeros(L, S, dtype=torch.bool)
        attn_mask[:, -2:] = True  # mask out last 2 source positions
        # key padding mask ignoring one position
        kpm = torch.zeros(N, S, dtype=torch.bool)
        kpm[:, -1] = True

    with torch.no_grad():
        out_src, _ = src(q, k, v, key_padding_mask=kpm, attn_mask=attn_mask,
                         need_weights=False)
        out_new, _ = new(q, k, v, key_padding_mask=kpm, attn_mask=attn_mask,
                         need_weights=False)
    abs_err = (out_src - out_new).abs().max().item()
    rel_err = abs_err / (out_src.abs().max().item() + 1e-9)
    return abs_err, rel_err


def main():
    cases = [
        ("no_mask small", 64, 4, 5, 7, 2, 0, False),
        ("no_mask BP-like", 256, 8, 100, 77, 2, 1, False),
        ("masked small", 64, 4, 5, 7, 2, 2, True),
        ("masked BP-like", 256, 8, 100, 77, 2, 3, True),
    ]
    ok = True
    for label, d, h, L, S, N, seed, mask in cases:
        abs_e, rel_e = parity_once(d, h, L, S, N, seed, mask)
        # We allow small fp32 rounding; SDPA can re-associate sums.
        passed = abs_e < 1e-5 and rel_e < 1e-5
        ok &= passed
        print(f"[{'PASS' if passed else 'FAIL'}] {label}: abs_err={abs_e:.2e} rel_err={rel_e:.2e}")

    # Trainable-params sanity: only LoRA A/B should be trainable
    torch.manual_seed(0)
    src = nn.MultiheadAttention(256, 8, dropout=0.0)
    new = LoRAMultiheadAttention.from_mha(src, rank=16, alpha=32, targets=("q", "v"))
    tr = [(n, p.numel()) for n, p in new.named_parameters() if p.requires_grad]
    tr_total = sum(n for _, n in tr)
    expected = 2 * (16 * 256 + 256 * 16)   # Q-LoRA + V-LoRA
    print(f"[{'PASS' if tr_total == expected else 'FAIL'}] "
          f"trainable_params={tr_total}  expected={expected}  (Q,V LoRA only)")
    for n, sz in tr:
        print(f"    {n}  {sz}")

    # Post-training simulation: perturb LoRA B, outputs should *differ*
    torch.manual_seed(0)
    src = nn.MultiheadAttention(64, 4, dropout=0.0)
    new = LoRAMultiheadAttention.from_mha(src, rank=4, alpha=8, targets=("q", "v"))
    for n, p in new.named_parameters():
        if "lora_B" in n:
            p.data.normal_(0, 0.01)
    src.eval(); new.eval()
    q = torch.randn(5, 2, 64); k = torch.randn(7, 2, 64); v = torch.randn(7, 2, 64)
    with torch.no_grad():
        o_s, _ = src(q, k, v, need_weights=False)
        o_n, _ = new(q, k, v, need_weights=False)
    diff = (o_s - o_n).abs().mean().item()
    # Any reasonable perturbation should make them differ measurably.
    nonzero_passed = diff > 1e-4
    print(f"[{'PASS' if nonzero_passed else 'FAIL'}] "
          f"after-training simulation: mean_diff={diff:.4e} (should be > 1e-4)")
    ok &= nonzero_passed

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
