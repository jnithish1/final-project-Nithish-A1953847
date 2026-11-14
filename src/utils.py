"""Small shared helpers: seeding, plotting style, fixed colours, json io."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                       # headless: write PNGs, never open windows
import matplotlib.pyplot as plt
import numpy as np

# Fixed categorical order (colour follows the entity, never its rank).
CLASS_COLORS = {0: "#2a78d6", 1: "#eb6834"}          # not hateful / hateful
CLASS_NAMES = {0: "Not hateful", 1: "Hateful"}

MODEL_ORDER = ["image_finetuned", "text_bert", "early_fusion", "late_fusion", "visual_bert",
               "hate_clipper", "cagco_no_ccl", "cagco_no_coattn", "cagco"]
CLIP_HEAD_MODELS = ["early_fusion", "late_fusion", "hate_clipper", "cagco_no_ccl", "cagco_no_coattn", "cagco"]
FINETUNED_MODELS = ["image_finetuned", "text_bert", "visual_bert"]
MODEL_LABELS = {
    "image_finetuned": "Image-only (fine-tuned ResNet)",
    "text_bert": "Text-only (fine-tuned BERT)",
    "early_fusion": "Early fusion (concat)",
    "late_fusion": "Late fusion (logit avg)",
    "visual_bert": "VisualBERT (COCO)",
    "hate_clipper": "Hate-CLIPper (cross)",
    "cagco_no_ccl": "CAGCo-Net w/o L_ccl",
    "cagco_no_coattn": "CAGCo-Net w/o co-attn",
    "cagco": "CAGCo-Net (ours)",
}
# 8 fixed categorical hues + a neutral grey for the second ablation (never a generated 9th hue)
MODEL_COLORS = dict(zip(MODEL_ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6",
                                      "#e87ba4", "#008300", "#8a8984", "#e34948"]))


def set_style():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": "#b9b8b2", "axes.labelcolor": "#0b0b0b",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e7e6e1", "grid.linewidth": 0.8,
        "xtick.color": "#52514e", "ytick.color": "#52514e",
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
        "legend.frameon": False, "lines.linewidth": 2,
    })


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def save_json(obj, path: Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)
