"""
Self-check of every model and both training loops on RANDOM data.

    python -m src.smoke_test               # CLIP-feature models only: no downloads, ~1 min on CPU
    python -m src.smoke_test --finetuned   # also ResNet-152, BERT, Faster R-CNN regions and
                                           # VisualBERT (downloads ~1.3 GB of weights; use Colab)

It checks output shapes, that the contrastive loss is finite and differentiable, and runs
one epoch of each training loop. Nothing here touches the real dataset or outputs/.
"""
import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from . import train as T
from .config import CFG
from .losses import confounder_supcon
from .models import build_model, count_params
from .utils import CLIP_HEAD_MODELS, FINETUNED_MODELS


def fake_split(root: Path, name: str, n: int, rng, with_images: bool):
    fdir = root / "out" / "features"
    y = rng.integers(0, 2, n)
    ids = [f"{name[:2]}{i:04d}" for i in range(n)]
    if with_images:
        (root / "data" / "img").mkdir(parents=True, exist_ok=True)
        for i in ids:
            arr = (rng.random((int(rng.integers(180, 320)), int(rng.integers(180, 320)), 3)) * 255).astype("uint8")
            Image.fromarray(arr).save(root / "data" / "img" / f"{i}.png")
    pd.DataFrame({"id": ids, "label": y, "img": [f"img/{i}.png" for i in ids],
                  "text": [f"random caption number {k} for testing" for k in range(n)]}
                 ).to_csv(fdir / f"{name}_index.csv", index=False)
    np.save(fdir / f"{name}_img_tokens.npy", rng.normal(size=(n, 50, 512)).astype(np.float16))
    np.save(fdir / f"{name}_txt_tokens.npy", rng.normal(size=(n, 64, 512)).astype(np.float16))
    m = np.zeros((n, 64), bool); m[:, :12] = True
    np.save(fdir / f"{name}_txt_mask.npy", m)
    np.save(fdir / f"{name}_img_pooled.npy", (rng.normal(size=(n, 512)) + y[:, None] * 0.3).astype(np.float16))
    np.save(fdir / f"{name}_txt_pooled.npy", rng.normal(size=(n, 512)).astype(np.float16))


def main(finetuned: bool):
    rng = np.random.default_rng(0)
    root = Path(tempfile.mkdtemp())
    CFG.out_dir, CFG.data_dir = root / "out", root / "data"
    CFG.ensure_dirs()
    sizes = [("train", 96), ("val", 32), ("test", 32)] if finetuned else [("train", 256), ("val", 64), ("test", 64)]
    for name, n in sizes:
        fake_split(root, name, n, rng, with_images=finetuned)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[smoke] device={device}, temp dir {root}")

    # 1. loss sanity
    z = torch.nn.functional.normalize(torch.randn(16, 128, requires_grad=True), dim=-1)
    y = torch.randint(0, 2, (16,))
    pm = torch.zeros(16, 16, dtype=torch.bool); pm[0, 1] = pm[1, 0] = True
    l = confounder_supcon(z, y, pm)
    l.backward()
    assert torch.isfinite(l), "contrastive loss is not finite"
    print(f"[smoke] confounder_supcon = {float(l):.4f} (finite, backprop ok)")

    # 2. CLIP-feature models
    data = {s: T.FeatureSplit(s, device) for s in ["train", "val", "test"]}
    n_tr = data["train"].n
    partners = {i: [i + 1] for i in range(0, n_tr - 1, 2)} | {i + 1: [i] for i in range(0, n_tr - 1, 2)}
    CFG.epochs, CFG.patience = 1, 1
    for name in CLIP_HEAD_MODELS:
        m = build_model(name, CFG).to(device)
        out = m(data["val"].batch(torch.arange(8, device=device)))
        assert out["logit"].shape == (8,), (name, out["logit"].shape)
        T.train_one(name, 0, data, partners, device)
        print(f"[smoke] {name:16s} ok  ({count_params(m):,} params)")

    # 3. fine-tuned benchmarks (optional, needs internet for pretrained weights)
    if finetuned:
        from . import extract_regions, finetune_baselines as F
        CFG.n_regions, CFG.ft_epochs, CFG.ft_batch = 8, 1, 8
        extract_regions.main()
        splits = {s: F.load_index(s) for s in ["train", "val", "test"]}
        for name in FINETUNED_MODELS:
            F.train_one(name, 0, splits, device)
            print(f"[smoke] {name:16s} ok")
    print("[smoke] all models passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--finetuned", action="store_true")
    main(ap.parse_args().finetuned)
