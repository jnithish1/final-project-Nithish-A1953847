"""
Step 2 - frozen CLIP feature extraction (transfer learning).

The CLIP ViT-B/32 image and text towers are run ONCE over every meme and their
outputs are cached to disk as float16 NumPy arrays. All models in this project
(benchmarks and CAGCo-Net) then train on the same cached features, which:
  * makes the comparison fair (identical inputs for every method), and
  * makes training cheap: only the small fusion heads are optimised.

Saved per split (N memes):
  {split}_img_tokens.npy   (N, 50, 512)  CLS + 7x7 patch tokens, projected to CLIP joint space
  {split}_txt_tokens.npy   (N, 64, 512)  caption tokens, projected to CLIP joint space
  {split}_txt_mask.npy     (N, 64)       True where a real (non-padding) token is
  {split}_img_pooled.npy   (N, 512)      CLIP image embedding (= projected CLS)
  {split}_txt_pooled.npy   (N, 512)      CLIP text embedding (= projected EOS)
  {split}_index.csv        id, label, img, text in row order

Run:  python -m src.extract_features
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from .config import CFG
from .data import load_splits
from .utils import save_json


@torch.no_grad()
def encode_batch(model, processor, images, texts, device, max_len):
    enc = processor(text=texts, images=images, return_tensors="pt", padding="max_length",
                    truncation=True, max_length=max_len).to(device)

    # ---- image tower: every token goes through post-LN + projection, so patch
    #      tokens live in the same 512-d joint space as the pooled embedding
    v = model.vision_model(pixel_values=enc["pixel_values"])
    v_tok = model.visual_projection(model.vision_model.post_layernorm(v.last_hidden_state))  # (B,50,512)
    v_pool = v_tok[:, 0]                                                                   # CLS

    # ---- text tower: last_hidden_state already has the final LayerNorm applied
    t = model.text_model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    t_tok = model.text_projection(t.last_hidden_state)                                     # (B,L,512)
    t_pool = model.text_projection(t.pooler_output)                                        # EOS token
    return v_tok, t_tok, enc["attention_mask"].bool(), v_pool, t_pool


def main():
    CFG.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[extract] device={device}, backbone={CFG.clip_name}")
    model = CLIPModel.from_pretrained(CFG.clip_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(CFG.clip_name)
    for p in model.parameters():
        p.requires_grad_(False)                          # frozen backbone

    splits = load_splits(CFG.data_dir)
    fdir = CFG.out_dir / "features"
    L, B = CFG.max_text_tokens, CFG.extract_batch
    timing = {}
    for name, df in splits.items():
        n = len(df)
        n_vtok = model.vision_model.embeddings.num_positions          # 50 for ViT-B/32 @224
        d = model.projection_dim                                      # 512
        mm = {
            "img_tokens": np.lib.format.open_memmap(fdir / f"{name}_img_tokens.npy", "w+", np.float16, (n, n_vtok, d)),
            "txt_tokens": np.lib.format.open_memmap(fdir / f"{name}_txt_tokens.npy", "w+", np.float16, (n, L, d)),
            "txt_mask": np.lib.format.open_memmap(fdir / f"{name}_txt_mask.npy", "w+", np.bool_, (n, L)),
            "img_pooled": np.lib.format.open_memmap(fdir / f"{name}_img_pooled.npy", "w+", np.float16, (n, d)),
            "txt_pooled": np.lib.format.open_memmap(fdir / f"{name}_txt_pooled.npy", "w+", np.float16, (n, d)),
        }
        t0 = time.time()
        for s in tqdm(range(0, n, B), desc=f"extract {name}"):
            chunk = df.iloc[s:s + B]
            imgs = [Image.open(CFG.data_dir / p).convert("RGB") for p in chunk["img"]]
            texts = [str(t) for t in chunk["text"]]
            outs = encode_batch(model, processor, imgs, texts, device, L)
            for key, arr in zip(mm, outs):
                mm[key][s:s + len(chunk)] = arr.cpu().numpy().astype(mm[key].dtype)
        for a in mm.values():
            a.flush()
        timing[name] = {"n": n, "seconds": round(time.time() - t0, 1)}
        df[["id", "label", "img", "text"]].to_csv(fdir / f"{name}_index.csv", index=False)
        trunc = (mm["txt_mask"][:, -1]).mean()
        print(f"[extract] {name}: {n} memes in {timing[name]['seconds']}s, "
              f"{100*trunc:.2f}% captions hit the {L}-token limit")
    save_json(timing, CFG.out_dir / "tables" / "extract_timing.json")


if __name__ == "__main__":
    main()
