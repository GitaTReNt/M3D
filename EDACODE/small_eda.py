import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ===== 改成你自己的路径 =====
npy_root = Path(r"D:\M3D\M3D_RefSeg_npy")
feat_csv = Path(r".\eda_out\tables\regions_image_features.csv")
out_dir = Path(r".\eda_out\overlays_stratified")
out_dir.mkdir(parents=True, exist_ok=True)

feat = pd.read_csv(feat_csv)

# 只看非空 ROI
feat = feat.query("empty == 0").sort_values("voxels").reset_index(drop=True)
n = len(feat)
print("non-empty regions:", n)

# 分层抽样：小 / 中 / 大
k = 10  # 每层抽 k 个
small = feat.head(k)
mid = feat.iloc[n//2 : n//2 + k]
large = feat.tail(k)
pick = pd.concat([small, mid, large], ignore_index=True)

def load_3d(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    return arr

for _, row in pick.iterrows():
    case = row["case_id"]
    mid = int(row["Mask_ID"])
    vox = int(row["voxels"])

    ct_path = npy_root / case / "ct.npy"
    mask_path = npy_root / case / "mask.npy"
    text_path = npy_root / case / "text.json"

    if (not ct_path.exists()) or (not mask_path.exists()):
        continue

    ct = load_3d(ct_path).astype(np.float32)      # (D,H,W)
    mask = load_3d(mask_path)
    mask = np.rint(mask).astype(np.int32)

    roi = (mask == mid)
    if roi.sum() == 0:
        continue

    # 选择 ROI 面积最大的切片
    areas = roi.sum(axis=(1, 2))
    z = int(areas.argmax())
    area = int(areas[z])

    # 读取 label 描述（可选）
    desc = ""
    if text_path.exists():
        try:
            d = json.load(open(text_path, "r", encoding="utf-8"))
            desc = d.get(str(mid), "")
        except Exception:
            pass

    # 画图：可调窗口让对比更清晰
    img = ct[z]
    vmin = np.percentile(img, 1)
    vmax = np.percentile(img, 99)

    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
    plt.contour(roi[z].astype(np.uint8), levels=[0.5], linewidths=1)
    plt.title(f"{case} id={mid} vox={vox} z={z} area={area}\n{desc[:90]}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / f"{case}__{mid}_vox{vox}.png", dpi=220)
    plt.close()

print("Saved to:", out_dir.resolve())
