#!/usr/bin/env python3
"""
Train a lightweight slice-level text-image scorer for M3D-RefSeg.

Goal:
  Given (text, CT slice), predict whether the target lesion appears on this slice.

This is intended to replace heuristic CLS-similarity filtering in
inference_medclip_medsam_v3.py.

Labels:
  y = 1 if (mask_vol[z] == mask_id).any()
  y = 0 otherwise

Model:
  Frozen BiomedCLIP image/text features + small MLP head

Outputs:
  checkpoints/slice_scorer_best.pt
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from skimage import transform as sk_transform
from tqdm import tqdm
from open_clip import create_model_from_pretrained, get_tokenizer

# --------------------------
# Utils
# --------------------------

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_volume(path: Path) -> np.ndarray:
    arr = np.load(str(path))
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def prepare_slice_biomedclip(ct_slice: np.ndarray) -> torch.Tensor:
    """
    Match inference_medclip_medsam_v3.py preprocessing.
    Input: (H, W) float32 [0,1]
    Output: (3, 224, 224) float32
    """
    img_3c = np.repeat(ct_slice[:, :, None], 3, axis=-1)
    img_224 = sk_transform.resize(
        img_3c, (224, 224), order=3, preserve_range=True, anti_aliasing=True,
    ).astype(np.float32)

    vmin, vmax = img_224.min(), img_224.max()
    if vmax - vmin > 1e-8:
        img_224 = (img_224 - vmin) / (vmax - vmin)

    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    img_224 = (img_224 - mean) / std

    return torch.tensor(img_224, dtype=torch.float32).permute(2, 0, 1)


def build_case_list(npy_root: Path):
    case_dirs = sorted([d for d in npy_root.iterdir() if d.is_dir()])
    valid = []
    for case_dir in case_dirs:
        ct_path = case_dir / "ct.npy"
        mask_path = case_dir / "mask.npy"
        text_path = case_dir / "text.json"
        if ct_path.exists() and mask_path.exists() and text_path.exists():
            valid.append(case_dir)
    return valid


# --------------------------
# Dataset
# --------------------------

class SliceScorerDataset(Dataset):
    """
    Builds training samples of (ct_slice, text, y, meta).
    To avoid huge imbalance, we keep all positive slices and sample a ratio of negatives.
    """

    def __init__(self, case_dirs, neg_ratio=3, max_cases=0):
        self.samples = []

        if max_cases > 0:
            case_dirs = case_dirs[:max_cases]

        for case_dir in tqdm(case_dirs, desc="Building dataset"):
            case_id = case_dir.name
            ct_vol = load_volume(case_dir / "ct.npy").astype(np.float32)   # (D,H,W)
            mask_vol = np.rint(load_volume(case_dir / "mask.npy")).astype(np.int32)
            with open(case_dir / "text.json", "r", encoding="utf-8") as f:
                text_map = json.load(f)

            D, H, W = ct_vol.shape

            for lid_str, desc in text_map.items():
                lid = int(lid_str)
                gt_3d = (mask_vol == lid).astype(np.uint8)

                pos_slices = []
                neg_slices = []

                for z in range(D):
                    y = 1 if gt_3d[z].any() else 0
                    if y == 1:
                        pos_slices.append(z)
                    else:
                        neg_slices.append(z)

                # keep all positives
                for z in pos_slices:
                    self.samples.append({
                        "case_id": case_id,
                        "z": z,
                        "lid": lid,
                        "text": desc,
                        "y": 1,
                        "ct_path": str(case_dir / "ct.npy"),
                    })

                # sample negatives
                n_neg_keep = min(len(neg_slices), max(len(pos_slices) * neg_ratio, neg_ratio))
                if len(neg_slices) > 0 and n_neg_keep > 0:
                    chosen = random.sample(neg_slices, n_neg_keep)
                    for z in chosen:
                        self.samples.append({
                            "case_id": case_id,
                            "z": z,
                            "lid": lid,
                            "text": desc,
                            "y": 0,
                            "ct_path": str(case_dir / "ct.npy"),
                        })

        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        ct_vol = load_volume(Path(s["ct_path"])).astype(np.float32)
        ct_slice = ct_vol[s["z"]]
        img = prepare_slice_biomedclip(ct_slice)

        return {
            "image": img,
            "text": s["text"],
            "label": torch.tensor(float(s["y"]), dtype=torch.float32),
            "case_id": s["case_id"],
            "z": s["z"],
            "lid": s["lid"],
        }


def collate_fn(batch):
    images = torch.stack([b["image"] for b in batch], dim=0)
    texts = [b["text"] for b in batch]
    labels = torch.stack([b["label"] for b in batch], dim=0)
    metas = [{
        "case_id": b["case_id"],
        "z": b["z"],
        "lid": b["lid"],
    } for b in batch]
    return images, texts, labels, metas


# --------------------------
# Model
# --------------------------

class SliceScorerHead(nn.Module):
    """
    Input:
      img_feat:  (B, C)
      text_feat: (B, C)
    Features:
      [img, text, img*text, abs(img-text)] => 4C
    Output:
      logit (B,)
    """

    def __init__(self, feat_dim=512, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim * 4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, img_feat, text_feat):
        x = torch.cat([
            img_feat,
            text_feat,
            img_feat * text_feat,
            torch.abs(img_feat - text_feat)
        ], dim=-1)
        return self.net(x).squeeze(-1)


# --------------------------
# Eval
# --------------------------

@torch.no_grad()
def evaluate_epoch(clip_model, tokenizer, head, loader, device):
    clip_model.eval()
    head.eval()

    all_probs = []
    all_labels = []

    for images, texts, labels, metas in tqdm(loader, desc="Val", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        tok = tokenizer(texts, context_length=256).to(device)

        with torch.no_grad():
            img_feat, text_feat, logit_scale = clip_model(images, tok)

        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        logits = head(img_feat, text_feat)
        probs = torch.sigmoid(logits)

        all_probs.append(probs.detach().cpu())
        all_labels.append(labels.detach().cpu())

    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()

    preds = (all_probs >= 0.5).astype(np.float32)
    acc = float((preds == all_labels).mean())

    # simple positive recall / precision
    tp = ((preds == 1) & (all_labels == 1)).sum()
    fp = ((preds == 1) & (all_labels == 0)).sum()
    fn = ((preds == 0) & (all_labels == 1)).sum()

    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(tp + fn, 1))

    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
    }


# --------------------------
# Main
# --------------------------

def main():
    parser = argparse.ArgumentParser("Train slice-level scorer")
    parser.add_argument("--npy_root", required=True, help="Path to M3D_RefSeg_npy")
    parser.add_argument("--out_dir", default="./checkpoints_slice_scorer")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--neg_ratio", type=int, default=3)
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze_clip", action="store_true", default=True)
    args = parser.parse_args()

    seed_everything(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # load CLIP exactly like inference
    from open_clip import create_model_from_pretrained, get_tokenizer

    print("[+] Loading BiomedCLIP from HuggingFace...")

    clip_model, preprocess = create_model_from_pretrained(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )
    clip_model = clip_model.to(device)
    tokenizer = get_tokenizer(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )

    # freeze CLIP for first version
    if args.freeze_clip:
        for p in clip_model.parameters():
            p.requires_grad = False
        clip_model.eval()

    head = SliceScorerHead(feat_dim=512, hidden_dim=512, dropout=0.1).to(device)

    case_dirs = build_case_list(Path(args.npy_root))
    random.shuffle(case_dirs)
    if args.max_cases > 0:
        case_dirs = case_dirs[:args.max_cases]

    n_train = int(len(case_dirs) * args.train_frac)
    train_cases = case_dirs[:n_train]
    val_cases = case_dirs[n_train:]

    print(f"[+] Train cases: {len(train_cases)} | Val cases: {len(val_cases)}")

    train_ds = SliceScorerDataset(train_cases, neg_ratio=args.neg_ratio)
    val_ds = SliceScorerDataset(val_cases, neg_ratio=args.neg_ratio)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_fn
    )

    print(f"[+] Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)

    # compute rough pos_weight from train set
    ys = np.array([s["y"] for s in train_ds.samples], dtype=np.float32)
    n_pos = max((ys == 1).sum(), 1)
    n_neg = max((ys == 0).sum(), 1)
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print(f"[+] pos_weight = {pos_weight.item():.3f}")

    best_recall = -1.0

    for epoch in range(1, args.epochs + 1):
        head.train()
        running_loss = 0.0

        for images, texts, labels, metas in tqdm(train_loader, desc=f"Train {epoch}"):
            images = images.to(device)
            labels = labels.to(device)

            tok = tokenizer(texts, context_length=256).to(device)

            with torch.no_grad():
                with torch.no_grad():
                    img_feat, text_feat, logit_scale = clip_model(images, tok)

            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

            logits = head(img_feat, text_feat)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / max(len(train_ds), 1)
        metrics = evaluate_epoch(clip_model, tokenizer, head, val_loader, device)

        print(
            f"[Epoch {epoch}] "
            f"train_loss={train_loss:.4f} "
            f"val_acc={metrics['acc']:.4f} "
            f"val_prec={metrics['precision']:.4f} "
            f"val_recall={metrics['recall']:.4f}"
        )

        # prioritize recall because missing positive slices hurts downstream more
        if metrics["recall"] > best_recall:
            best_recall = metrics["recall"]
            ckpt_path = out_dir / "slice_scorer_best.pt"
            torch.save({
                "head_state_dict": head.state_dict(),
                "feat_dim": 512,
                "hidden_dim": 512,
                "dropout": 0.1,
                "best_recall": best_recall,
            }, ckpt_path)
            print(f"[+] Saved best checkpoint to {ckpt_path}")

    print("[+] Done.")


if __name__ == "__main__":
    main()