"""
Region features for the VisualBERT benchmark.

VisualBERT reads a caption together with a set of object regions. Kiela et al. used
Faster R-CNN region features, so we do the same with torchvision's COCO-trained
Faster R-CNN (ResNet-50-FPN v2):

  1. run the detector on each meme,
  2. keep the top-K detected boxes (K = 36), topping up with the highest-scoring
     RPN proposals when fewer than K objects are found,
  3. RoI-align every box on the FPN feature maps and pass it through the detector's
     own box head -> one 1024-d vector per region.

Saved per split:
  {split}_regions.npy      (N, K, 1024) float16
  {split}_region_mask.npy  (N, K)       bool, True = real region

Run:  python -m src.extract_regions      (after extract_features, which writes the split index)
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from .config import CFG
from .utils import save_json


def load_detector(device: str):
    from torchvision.models.detection import (FasterRCNN_ResNet50_FPN_V2_Weights,
                                              fasterrcnn_resnet50_fpn_v2)
    det = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT).to(device).eval()
    det.roi_heads.detections_per_img = CFG.n_regions
    for p in det.parameters():
        p.requires_grad_(False)
    return det


@torch.no_grad()
def region_features(det, imgs: list[torch.Tensor], K: int):
    images, _ = det.transform(imgs)                       # resize + normalise -> ImageList
    feats = det.backbone(images.tensors)                  # FPN feature maps
    proposals, _ = det.rpn(images, feats)                 # sorted by objectness after NMS
    dets, _ = det.roi_heads(feats, proposals, images.image_sizes)
    boxes = [torch.cat([d["boxes"], p], 0)[:K] for d, p in zip(dets, proposals)]
    pooled = det.roi_heads.box_roi_pool(feats, boxes, images.image_sizes)
    emb = det.roi_heads.box_head(pooled)                  # (sum of boxes, 1024)
    out = torch.zeros(len(imgs), K, emb.shape[1])
    mask = torch.zeros(len(imgs), K, dtype=torch.bool)
    s = 0
    for i, b in enumerate(boxes):
        n = len(b)
        out[i, :n] = emb[s:s + n].float().cpu()
        mask[i, :n] = True
        s += n
    return out, mask


def main():
    CFG.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    det = load_detector(device)
    fdir, K = CFG.out_dir / "features", CFG.n_regions
    timing = {}
    for split in ["train", "val", "test"]:
        idx = pd.read_csv(fdir / f"{split}_index.csv", dtype={"id": str})
        n = len(idx)
        feats = np.lib.format.open_memmap(fdir / f"{split}_regions.npy", "w+", np.float16, (n, K, 1024))
        masks = np.lib.format.open_memmap(fdir / f"{split}_region_mask.npy", "w+", np.bool_, (n, K))
        t0 = time.time()
        for s in tqdm(range(0, n, CFG.region_batch), desc=f"regions {split}"):
            chunk = idx.iloc[s:s + CFG.region_batch]
            imgs = [TF.to_tensor(Image.open(CFG.data_dir / p).convert("RGB")).to(device) for p in chunk["img"]]
            f, m = region_features(det, imgs, K)
            feats[s:s + len(chunk)] = f.numpy().astype(np.float16)
            masks[s:s + len(chunk)] = m.numpy()
        feats.flush(); masks.flush()
        timing[split] = {"n": n, "seconds": round(time.time() - t0, 1),
                         "mean_regions": float(masks.sum(1).mean())}
        print(f"[regions] {split}: {n} memes, {timing[split]['mean_regions']:.1f} regions/meme, "
              f"{timing[split]['seconds']}s")
    save_json(timing, CFG.out_dir / "tables" / "region_timing.json")


if __name__ == "__main__":
    main()
