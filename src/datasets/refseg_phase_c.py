"""M3D-RefSeg dataset for Phase C (per-slice sampling).

Each sample is a single (case, mask_id, slice_idx) triple.  The dataset
returns the CT slice as a 3-channel stack (prev / curr / next, BP v2
packing convention), the text prompt for mask_id, the GT mask for that
slice, and auxiliary targets (case-level 3D labels + per-slice flags).

Positive / negative slice ratio is controlled by the sampler wrapper.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


# ---------------------------------------------------------------------------
# index construction
# ---------------------------------------------------------------------------

@dataclass
class SliceSample:
    case_id: str
    mask_id: int
    slice_idx: int
    is_positive: int  # 1 if mask_id present on this slice


def build_slice_index(
    cases: list[str],
    npy_root: Path,
    aux_root: Path,
) -> list[SliceSample]:
    """Enumerate every (case, mask_id, slice) triple for the given cases."""
    items: list[SliceSample] = []
    for cid in cases:
        aux_path = aux_root / f"{cid}.json"
        if not aux_path.exists():
            raise FileNotFoundError(
                f"aux labels missing for {cid}; run scripts/build_aux_labels.py"
            )
        with open(aux_path, "r", encoding="utf-8") as f:
            aux = json.load(f)
        for mid_str, a in aux.items():
            mid = int(mid_str)
            slice_exist = a["slice_exist"]
            for z, flag in enumerate(slice_exist):
                items.append(SliceSample(cid, mid, z, int(flag)))
    return items


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------

class RefSegPhaseCDataset(Dataset):
    """Per-slice dataset.

    Returns a dict (torch tensors unless noted):
        image_3c     (3, H, W) float32 — prev/curr/next slices in [0,1]
        text         str
        gt_mask      (H, W) float32 binary for mask_id on this slice
        existence    float (case-level)
        slice_exist  float (per-slice)
        bbox_3d      (6,) float32 in [0,1]
        centroid_3d  (3,) float32 in [0,1]
        z_range      (2,) float32 in [0,1]
        meta         dict (not collated by default)
    """

    def __init__(
        self,
        cases: list[str],
        npy_root: str | Path = "data/M3D_RefSeg_npy",
        aux_root: str | Path = "data/M3D_RefSeg_aux",
        cache_arrays: bool = True,
    ):
        self.npy_root = Path(npy_root)
        self.aux_root = Path(aux_root)
        self.cases = cases
        self.items = build_slice_index(cases, self.npy_root, self.aux_root)
        self._array_cache: dict[str, tuple[np.ndarray, np.ndarray, dict, dict]] = {}
        self._cache_enabled = cache_arrays

    def __len__(self) -> int:
        return len(self.items)

    def _load_case(self, case_id: str):
        if case_id in self._array_cache:
            return self._array_cache[case_id]
        ct = np.load(self.npy_root / case_id / "ct.npy")  # (1,D,H,W) in [0,1]
        if ct.ndim == 4:
            ct = ct[0]
        mask = np.load(self.npy_root / case_id / "mask.npy")  # (1,D,H,W) int-valued float
        if mask.ndim == 4:
            mask = mask[0]
        mask = mask.astype(np.int32)
        with open(self.npy_root / case_id / "text.json", "r", encoding="utf-8") as f:
            text_map = json.load(f)
        with open(self.aux_root / f"{case_id}.json", "r", encoding="utf-8") as f:
            aux_map = json.load(f)
        pack = (ct.astype(np.float32), mask, text_map, aux_map)
        if self._cache_enabled:
            self._array_cache[case_id] = pack
        return pack

    def __getitem__(self, idx: int) -> dict:
        it = self.items[idx]
        ct, mask, text_map, aux_map = self._load_case(it.case_id)
        D, H, W = ct.shape
        z = it.slice_idx

        # BP v2 expects 3-channel stacking of neighbouring slices
        zm = max(0, z - 1)
        zp = min(D - 1, z + 1)
        image_3c = np.stack([ct[zm], ct[z], ct[zp]], axis=0)  # (3,H,W)

        gt_slice = (mask[z] == it.mask_id).astype(np.float32)
        aux = aux_map[str(it.mask_id)]
        text = text_map[str(it.mask_id)]

        out = {
            "image_3c": torch.from_numpy(image_3c).float(),
            "text": text,
            "gt_mask": torch.from_numpy(gt_slice).float(),
            "existence": torch.tensor(aux["existence"], dtype=torch.float32),
            "slice_exist": torch.tensor(it.is_positive, dtype=torch.float32),
            "bbox_3d": torch.tensor(aux["bbox_3d"], dtype=torch.float32),
            "centroid_3d": torch.tensor(aux["centroid_3d"], dtype=torch.float32),
            "z_range": torch.tensor(aux["z_range"], dtype=torch.float32),
            "meta": {
                "case_id": it.case_id,
                "mask_id": it.mask_id,
                "slice_idx": z,
                "D": D, "H": H, "W": W,
            },
        }
        return out


def collate_phase_c(batch: list[dict]) -> dict:
    out = {
        "image_3c": torch.stack([b["image_3c"] for b in batch], dim=0),
        "text": [b["text"] for b in batch],
        "gt_mask": torch.stack([b["gt_mask"] for b in batch], dim=0),
        "existence": torch.stack([b["existence"] for b in batch], dim=0),
        "slice_exist": torch.stack([b["slice_exist"] for b in batch], dim=0),
        "bbox_3d": torch.stack([b["bbox_3d"] for b in batch], dim=0),
        "centroid_3d": torch.stack([b["centroid_3d"] for b in batch], dim=0),
        "z_range": torch.stack([b["z_range"] for b in batch], dim=0),
        "meta": [b["meta"] for b in batch],
    }
    return out


# ---------------------------------------------------------------------------
# pos/neg ratio sampler
# ---------------------------------------------------------------------------

class PosNegRatioSampler(Sampler[int]):
    """Draws indices with a fixed positive/negative mix.

    Used to enforce ``pos_ratio=0.7`` as specified in the Phase C plan.
    Always returns ``num_samples`` indices per epoch.
    """

    def __init__(
        self,
        dataset: RefSegPhaseCDataset,
        pos_ratio: float = 0.7,
        num_samples: int | None = None,
        seed: int = 0,
    ):
        self.pos_indices = [i for i, it in enumerate(dataset.items) if it.is_positive]
        self.neg_indices = [i for i, it in enumerate(dataset.items) if not it.is_positive]
        if not self.pos_indices:
            raise ValueError("no positive slices in dataset")
        self.pos_ratio = pos_ratio
        self.num_samples = num_samples if num_samples is not None else len(dataset)
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        n_pos = int(round(self.num_samples * self.pos_ratio))
        n_neg = self.num_samples - n_pos
        pos = torch.tensor(self.pos_indices)
        neg = torch.tensor(self.neg_indices) if self.neg_indices else pos
        pos_sel = pos[torch.randint(0, len(pos), (n_pos,), generator=g)]
        neg_sel = neg[torch.randint(0, len(neg), (n_neg,), generator=g)] if self.neg_indices else torch.empty(0, dtype=torch.long)
        order = torch.cat([pos_sel, neg_sel])
        perm = torch.randperm(order.numel(), generator=g)
        yield from order[perm].tolist()

    def __len__(self) -> int:
        return self.num_samples


# ---------------------------------------------------------------------------
# case-level dataset for evaluation
# ---------------------------------------------------------------------------

class RefSegPhaseCCaseDataset(Dataset):
    """One sample per (case, mask_id) — returns the full 3D volume.

    Used for end-of-epoch dev eval and final test eval. Collate is *not*
    supported; use ``batch_size=1``.
    """

    def __init__(
        self,
        cases: list[str],
        npy_root: str | Path = "data/M3D_RefSeg_npy",
        aux_root: str | Path = "data/M3D_RefSeg_aux",
    ):
        self.npy_root = Path(npy_root)
        self.aux_root = Path(aux_root)
        self.items: list[tuple[str, int]] = []
        for cid in cases:
            with open(self.npy_root / cid / "text.json", "r", encoding="utf-8") as f:
                text = json.load(f)
            for mid_str in text.keys():
                self.items.append((cid, int(mid_str)))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        cid, mid = self.items[idx]
        ct = np.load(self.npy_root / cid / "ct.npy")
        if ct.ndim == 4:
            ct = ct[0]
        mask = np.load(self.npy_root / cid / "mask.npy")
        if mask.ndim == 4:
            mask = mask[0]
        with open(self.npy_root / cid / "text.json", "r", encoding="utf-8") as f:
            text = json.load(f)
        with open(self.aux_root / f"{cid}.json", "r", encoding="utf-8") as f:
            aux = json.load(f)
        gt = (mask.astype(np.int32) == mid).astype(np.uint8)
        return {
            "case_id": cid,
            "mask_id": mid,
            "ct": ct.astype(np.float32),  # (D,H,W) in [0,1]
            "gt_mask_3d": gt,             # (D,H,W) uint8
            "text": text[str(mid)],
            "aux": aux[str(mid)],
        }
