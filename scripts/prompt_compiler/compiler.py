"""
Text-to-Prompt Compiler: parse radiology text into structured Prompt Packets.

Pipeline: free text → structured slots → atlas prior → retrieval refinement → typed prompt
"""

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PromptPacket:
    """Structured, model-agnostic intermediate representation."""
    anatomy: str = "unknown"
    side: str = "none"           # left / right / bilateral / none
    level: str = ""              # e.g., "C5", "L4-L5", "upper_lobe"
    target_form: str = "focal"   # focal / multifocal / diffuse / bony
    finding_type: str = "lesion"
    count: str = "single"        # single / multiple
    # Spatial priors (filled by atlas/retrieval)
    z_range: tuple = (0.0, 1.0)  # normalized [0,1]
    box_prior: tuple = (0.0, 0.0, 1.0, 1.0)  # (y_min, x_min, y_max, x_max) normalized
    size_prior: str = "medium"   # tiny / small / medium / large
    # Post-processing rule
    post_rule: str = "keep_nearest"  # keep_nearest / keep_multi / size_clamp / organ_roi


# --------------- Text Parsing ---------------

ANATOMY_MAP = {
    # organ → canonical name
    "liver": "liver", "hepatic": "liver",
    "kidney": "kidney", "renal": "kidney",
    "lung": "lung", "pulmonary": "lung", "lobe": "lung",
    "spleen": "spleen", "splenic": "spleen",
    "heart": "heart", "cardiac": "heart", "coronary": "heart",
    "aorta": "aorta", "aortic": "aorta",
    "pancrea": "pancreas",
    "gallbladder": "gallbladder",
    "bladder": "bladder",
    "thyroid": "thyroid",
    "stomach": "stomach", "gastric": "stomach",
    "esophag": "esophagus",
    "colon": "colon", "intestin": "intestine", "bowel": "intestine",
    "pelvi": "pelvis", "inguinal": "pelvis",
    "pleura": "pleura", "pleural": "pleura",
    "mediastin": "mediastinum",
    "peritoneum": "peritoneum", "mesentery": "peritoneum",
    "prostate": "prostate",
    "uterus": "uterus", "uterine": "uterus",
    "brain": "brain", "cerebr": "brain",
    "orbit": "orbit",
    "vertebr": "spine", "cervical": "spine", "lumbar": "spine",
    "thoracic": "spine", "spinal": "spine", "spine": "spine",
    "spondyl": "spine", "disc": "spine", "intervertebral": "spine",
    "vocal": "larynx", "piriform": "larynx", "laryn": "larynx",
    "lymph": "lymph_node",
    "bone": "bone", "osteophyte": "bone", "fracture": "bone",
}

FINDING_MAP = {
    "mass": "mass", "tumor": "mass",
    "nodule": "nodule", "nodular": "nodule",
    "cyst": "cyst", "cystic": "cyst",
    "calcif": "calcification",
    "effusion": "effusion",
    "emphysema": "emphysema",
    "atelectasis": "atelectasis",
    "pneumonia": "pneumonia",
    "fracture": "fracture",
    "hernia": "hernia",
    "stenosis": "stenosis",
    "thicken": "thickening",
    "enlarg": "enlargement", "splenomegaly": "enlargement",
    "dilat": "dilation",
    "osteophyte": "osteophyte",
    "spondyl": "spondylosis",
    "dissection": "dissection",
    "hemorrhage": "hemorrhage",
    "edema": "edema",
    "cirrhosis": "cirrhosis",
    "ascites": "ascites",
}

SPINE_LEVELS = {
    "c1": "C1", "c2": "C2", "c3": "C3", "c4": "C4", "c5": "C5",
    "c6": "C6", "c7": "C7",
    "t1": "T1", "t2": "T2", "t3": "T3", "t4": "T4", "t5": "T5",
    "t6": "T6", "t7": "T7", "t8": "T8", "t9": "T9", "t10": "T10",
    "t11": "T11", "t12": "T12",
    "l1": "L1", "l2": "L2", "l3": "L3", "l4": "L4", "l5": "L5",
    "s1": "S1",
}


def parse_text(text: str) -> PromptPacket:
    """Parse free-form radiology text into a PromptPacket."""
    t = text.lower()
    pkt = PromptPacket()

    # --- Anatomy ---
    for keyword, canonical in ANATOMY_MAP.items():
        if keyword in t:
            pkt.anatomy = canonical
            break  # take first (most specific) match

    # --- Side ---
    if "bilateral" in t or "both" in t:
        pkt.side = "bilateral"
    elif "left" in t and "right" in t:
        pkt.side = "bilateral"
    elif "left" in t:
        pkt.side = "left"
    elif "right" in t:
        pkt.side = "right"

    # --- Spine level ---
    for pattern, level in SPINE_LEVELS.items():
        if re.search(rf'\b{pattern}\b', t):
            pkt.level = level
            break

    # Also check "upper/middle/lower" for lungs/kidneys
    if pkt.anatomy in ("lung", "kidney"):
        if "upper" in t or "apex" in t:
            pkt.level = "upper"
        elif "lower" in t or "base" in t:
            pkt.level = "lower"
        elif "middle" in t:
            pkt.level = "middle"

    # --- Finding type ---
    for keyword, canonical in FINDING_MAP.items():
        if keyword in t:
            pkt.finding_type = canonical
            break

    # --- Count / Focality ---
    if any(w in t for w in ["multiple", "several", "numerous"]):
        pkt.count = "multiple"
    if any(w in t for w in ["bilateral", "both"]):
        pkt.count = "multiple"

    # --- Target form ---
    if pkt.anatomy in ("spine", "bone") or pkt.finding_type in ("osteophyte", "spondylosis", "fracture"):
        pkt.target_form = "bony"
    elif any(w in t for w in ["diffuse", "scattered"]) or pkt.finding_type == "emphysema":
        pkt.target_form = "diffuse"
    elif pkt.count == "multiple" and pkt.finding_type not in ("mass",):
        pkt.target_form = "multifocal"
    else:
        pkt.target_form = "focal"

    # --- Post-processing rule ---
    if pkt.target_form == "focal":
        pkt.post_rule = "keep_nearest"
    elif pkt.target_form in ("multifocal", "diffuse"):
        pkt.post_rule = "keep_multi"
    elif pkt.target_form == "bony":
        pkt.post_rule = "keep_multi"  # bony changes are often multi-level

    # Enlargement targets (splenomegaly, etc.) are organ-level
    if pkt.finding_type in ("enlargement", "cirrhosis", "ascites"):
        pkt.post_rule = "organ_roi"
        pkt.target_form = "diffuse"

    return pkt


# --------------- Atlas Prior ---------------

# Coarse anatomy → normalized spatial prior in (32, 256, 256) volume
# z_range: (z_start_frac, z_end_frac) — fraction of total depth
# box: (y_min_frac, x_min_frac, y_max_frac, x_max_frac) — fraction of H, W

ATLAS = {
    # Head/Neck
    "brain":        {"z": (0.0, 0.3), "box": (0.1, 0.1, 0.9, 0.9)},
    "orbit":        {"z": (0.0, 0.2), "box": (0.2, 0.1, 0.6, 0.9)},
    "thyroid":      {"z": (0.1, 0.4), "box": (0.3, 0.2, 0.7, 0.8)},
    "larynx":       {"z": (0.1, 0.4), "box": (0.3, 0.2, 0.7, 0.8)},
    # Chest
    "lung":         {"z": (0.1, 0.8), "box": (0.05, 0.05, 0.95, 0.95)},
    "heart":        {"z": (0.3, 0.7), "box": (0.2, 0.2, 0.8, 0.8)},
    "aorta":        {"z": (0.1, 0.9), "box": (0.2, 0.3, 0.8, 0.7)},
    "mediastinum":  {"z": (0.2, 0.7), "box": (0.2, 0.2, 0.8, 0.8)},
    "esophagus":    {"z": (0.2, 0.8), "box": (0.3, 0.3, 0.7, 0.7)},
    "pleura":       {"z": (0.1, 0.9), "box": (0.05, 0.05, 0.95, 0.95)},
    # Abdomen
    "liver":        {"z": (0.2, 0.7), "box": (0.1, 0.05, 0.7, 0.6)},
    "gallbladder":  {"z": (0.3, 0.6), "box": (0.3, 0.2, 0.6, 0.5)},
    "spleen":       {"z": (0.2, 0.6), "box": (0.1, 0.5, 0.6, 0.95)},
    "pancreas":     {"z": (0.3, 0.6), "box": (0.2, 0.2, 0.6, 0.8)},
    "kidney":       {"z": (0.3, 0.7), "box": (0.15, 0.1, 0.7, 0.9)},
    "stomach":      {"z": (0.2, 0.6), "box": (0.2, 0.3, 0.7, 0.8)},
    "intestine":    {"z": (0.3, 0.9), "box": (0.1, 0.1, 0.9, 0.9)},
    "peritoneum":   {"z": (0.2, 0.9), "box": (0.1, 0.1, 0.9, 0.9)},
    # Pelvis
    "pelvis":       {"z": (0.6, 1.0), "box": (0.1, 0.1, 0.9, 0.9)},
    "bladder":      {"z": (0.6, 0.9), "box": (0.2, 0.2, 0.8, 0.8)},
    "prostate":     {"z": (0.7, 0.95), "box": (0.3, 0.3, 0.7, 0.7)},
    "uterus":       {"z": (0.6, 0.9), "box": (0.2, 0.2, 0.8, 0.8)},
    # Spine (wide range — will be refined by level)
    "spine":        {"z": (0.0, 1.0), "box": (0.3, 0.2, 0.9, 0.8)},
    "bone":         {"z": (0.0, 1.0), "box": (0.1, 0.1, 0.9, 0.9)},
    # Other
    "lymph_node":   {"z": (0.1, 0.9), "box": (0.1, 0.1, 0.9, 0.9)},
    "unknown":      {"z": (0.0, 1.0), "box": (0.0, 0.0, 1.0, 1.0)},
}

# Side refinement: if left/right, narrow the x-range
SIDE_X_RANGES = {
    # In standard CT orientation (patient's left = image right)
    "left":  (0.5, 1.0),  # patient's left → image right half
    "right": (0.0, 0.5),  # patient's right → image left half
}


def apply_atlas_prior(pkt: PromptPacket) -> PromptPacket:
    """Fill z_range and box_prior from atlas lookup."""
    atlas_entry = ATLAS.get(pkt.anatomy, ATLAS["unknown"])
    pkt.z_range = atlas_entry["z"]
    y_min, x_min, y_max, x_max = atlas_entry["box"]

    # Refine by side
    if pkt.side in SIDE_X_RANGES and pkt.anatomy not in ("spine", "bone", "aorta"):
        sx_min, sx_max = SIDE_X_RANGES[pkt.side]
        # Intersect atlas box with side constraint
        x_min = max(x_min, sx_min)
        x_max = min(x_max, sx_max)

    pkt.box_prior = (y_min, x_min, y_max, x_max)
    return pkt


def compile_text(text: str) -> PromptPacket:
    """Full pipeline: text → parsed → atlas-primed PromptPacket."""
    pkt = parse_text(text)
    pkt = apply_atlas_prior(pkt)
    return pkt
