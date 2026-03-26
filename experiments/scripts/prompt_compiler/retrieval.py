"""
Structured retrieval: match on parsed slots, refine atlas prior with train data.
"""

import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional

from .compiler import PromptPacket, compile_text


@dataclass
class BankEntry:
    case_id: str
    mask_id: int
    text: str
    packet: PromptPacket
    # Ground-truth spatial info (normalized)
    z_min: float
    z_max: float
    y_min: float
    x_min: float
    y_max: float
    x_max: float
    gt_voxels: int
    D: int
    H: int
    W: int


def build_prompt_bank(npy_root: Path) -> List[BankEntry]:
    """Build prompt bank from all cases with structured parsing."""
    bank = []
    for case_dir in sorted(d for d in npy_root.iterdir() if d.is_dir()):
        case_id = case_dir.name
        mask_path = case_dir / "mask.npy"
        text_path = case_dir / "text.json"
        if not mask_path.exists() or not text_path.exists():
            continue

        mask_vol = np.load(str(mask_path))
        if mask_vol.ndim == 4 and mask_vol.shape[0] == 1:
            mask_vol = mask_vol[0]
        mask_vol = np.rint(mask_vol).astype(np.int32)
        D, H, W = mask_vol.shape

        with open(text_path, "r", encoding="utf-8") as f:
            text_map = json.load(f)

        for lid_str, desc in text_map.items():
            lid = int(lid_str)
            gt = (mask_vol == lid).astype(np.uint8)
            pkt = compile_text(desc)

            if gt.sum() == 0:
                bank.append(BankEntry(
                    case_id=case_id, mask_id=lid, text=desc, packet=pkt,
                    z_min=-1, z_max=-1, y_min=-1, x_min=-1, y_max=-1, x_max=-1,
                    gt_voxels=0, D=D, H=H, W=W,
                ))
                continue

            zs, ys, xs = np.where(gt > 0)
            bank.append(BankEntry(
                case_id=case_id, mask_id=lid, text=desc, packet=pkt,
                z_min=float(zs.min()) / D,
                z_max=float(zs.max()) / D,
                y_min=float(ys.min()) / H,
                x_min=float(xs.min()) / W,
                y_max=float(ys.max()) / H,
                x_max=float(xs.max()) / W,
                gt_voxels=int(gt.sum()),
                D=D, H=H, W=W,
            ))

    return bank


def structured_similarity(query: PromptPacket, entry: BankEntry) -> float:
    """Compute structured similarity between query packet and bank entry."""
    score = 0.0
    ep = entry.packet

    # Anatomy match (most important)
    if query.anatomy == ep.anatomy:
        score += 4.0
    elif query.anatomy in ("kidney", "liver", "lung") and ep.anatomy == query.anatomy:
        score += 4.0

    # Side match
    if query.side == ep.side:
        score += 2.0
    elif query.side == "bilateral" and ep.side == "bilateral":
        score += 2.0
    elif query.side != "none" and ep.side != "none" and query.side != ep.side:
        score -= 1.0  # penalize wrong side

    # Finding type match
    if query.finding_type == ep.finding_type:
        score += 2.0

    # Target form match
    if query.target_form == ep.target_form:
        score += 1.0

    # Level match (spine)
    if query.level and ep.level and query.level == ep.level:
        score += 2.0

    # Count match
    if query.count == ep.count:
        score += 0.5

    return score


def retrieve_prior(
    query_pkt: PromptPacket,
    query_case_id: str,
    bank: List[BankEntry],
    top_k: int = 5,
) -> Tuple[Optional[dict], float, List[BankEntry]]:
    """
    Retrieve spatial prior from bank using structured matching.
    Returns: (prior_dict, avg_score, matched_entries)
    """
    candidates = []
    for entry in bank:
        if entry.case_id == query_case_id:
            continue  # leave-one-case-out
        if entry.gt_voxels == 0:
            continue
        sim = structured_similarity(query_pkt, entry)
        candidates.append((sim, entry))

    candidates.sort(key=lambda x: -x[0])
    top = candidates[:top_k]

    if not top or all(s <= 0 for s, _ in top):
        return None, 0.0, []

    # Filter: only keep entries with positive score
    top = [(s, e) for s, e in top if s > 0]
    if not top:
        return None, 0.0, []

    # Weighted aggregation
    scores = np.array([s for s, _ in top])
    weights = scores / scores.sum()

    z_mins = np.array([e.z_min for _, e in top])
    z_maxs = np.array([e.z_max for _, e in top])
    y_mins = np.array([e.y_min for _, e in top])
    x_mins = np.array([e.x_min for _, e in top])
    y_maxs = np.array([e.y_max for _, e in top])
    x_maxs = np.array([e.x_max for _, e in top])

    # Use weighted median for robustness (approximate via weighted percentile)
    prior = {
        "z_min": float(np.dot(weights, z_mins)),
        "z_max": float(np.dot(weights, z_maxs)),
        "y_min": float(np.dot(weights, y_mins)),
        "y_max": float(np.dot(weights, y_maxs)),
        "x_min": float(np.dot(weights, x_mins)),
        "x_max": float(np.dot(weights, x_maxs)),
    }

    avg_score = float(scores.mean())
    return prior, avg_score, [e for _, e in top]


def merge_atlas_and_retrieval(
    pkt: PromptPacket,
    retrieval_prior: Optional[dict],
    retrieval_score: float,
    min_score_for_retrieval: float = 3.0,
) -> dict:
    """
    Merge atlas prior (from packet) with retrieval prior.
    High retrieval score → trust retrieval more.
    Low retrieval score → fall back to atlas.
    """
    atlas = {
        "z_min": pkt.z_range[0],
        "z_max": pkt.z_range[1],
        "y_min": pkt.box_prior[0],
        "x_min": pkt.box_prior[1],
        "y_max": pkt.box_prior[2],
        "x_max": pkt.box_prior[3],
    }

    if retrieval_prior is None or retrieval_score < min_score_for_retrieval:
        return atlas

    # Blend: higher score → more trust in retrieval
    # Score typically ranges 0-10, normalize to alpha in [0.3, 0.9]
    alpha = min(0.9, max(0.3, retrieval_score / 10.0))

    merged = {}
    for key in atlas:
        merged[key] = alpha * retrieval_prior[key] + (1 - alpha) * atlas[key]

    return merged
