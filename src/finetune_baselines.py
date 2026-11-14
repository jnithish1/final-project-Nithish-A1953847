"""
Fine-tuned benchmarks from the proposal (Criterion 4).

  image_finetuned  Unimodal visual benchmark: ImageNet ResNet-152 fine-tuned end to end
                   on the meme image only (the "Image-Grid" family in Kiela et al., 2020).
  text_bert        Unimodal textual benchmark: bert-base-uncased fine-tuned on the
                   caption only ("Text BERT").
  visual_bert      Pretrained vision-language transformer: VisualBERT pretrained on COCO,
                   fine-tuned on caption tokens + 36 Faster R-CNN region features
                   ("Visual BERT COCO", the strongest baseline in the dataset paper).

They use the same splits (the *_index.csv files written by extract_features.py), the same
seeds, the same class-weighted BCE, AdamW with warm-up + cosine decay, gradient clipping and
early stopping on validation AUROC as the CLIP-feature models, and write predictions in the
same format, so evaluate.py treats every model identically.

Run:  python -m src.finetune_baselines
      python -m src.finetune_baselines --models text_bert --seeds 0 1
"""
from __future__ import annotations

import argparse
import math
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from .config import CFG
from .losses import bce_loss
from .train import lr_lambda
from .utils import FINETUNED_MODELS, save_json, seed_everything


def load_index(split: str) -> pd.DataFrame:
    return pd.read_csv(CFG.out_dir / "features" / f"{split}_index.csv", dtype={"id": str})


# --------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------
class ImageDS(Dataset):
    def __init__(self, df: pd.DataFrame, train: bool):
        from torchvision import transforms as T
        norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])   # ImageNet statistics
        self.tf = (T.Compose([T.RandomResizedCrop(224, scale=(0.7, 1.0)), T.RandomHorizontalFlip(),
                              T.ToTensor(), norm]) if train else
                   T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), norm]))
        self.paths = [CFG.data_dir / p for p in df["img"]]
        self.y = df["label"].values

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {"pixels": self.tf(Image.open(self.paths[i]).convert("RGB")),
                "y": torch.tensor(self.y[i])}


class TextDS(Dataset):
    """Tokenised captions; optionally adds VisualBERT region features."""

    def __init__(self, df: pd.DataFrame, tok, split: str | None = None, with_regions: bool = False):
        self.enc = tok(df["text"].astype(str).tolist(), padding="max_length", truncation=True,
                       max_length=CFG.ft_text_len, return_tensors="pt")
        self.y = torch.tensor(df["label"].values)
        self.regions = None
        if with_regions:
            fdir = CFG.out_dir / "features"
            self.regions = np.load(fdir / f"{split}_regions.npy", mmap_mode="r")
            self.rmask = np.load(fdir / f"{split}_region_mask.npy", mmap_mode="r")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        b = {k: v[i] for k, v in self.enc.items()}
        b["y"] = self.y[i]
        if self.regions is not None:
            b["visual_embeds"] = torch.from_numpy(np.asarray(self.regions[i], dtype=np.float32))
            b["visual_mask"] = torch.from_numpy(np.asarray(self.rmask[i], dtype=np.int64))
        return b


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        import torchvision
        self.net = torchvision.models.get_model(CFG.image_backbone, weights="DEFAULT")
        nf = self.net.fc.in_features
        self.net.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(nf, 1))

    def forward(self, b):
        return self.net(b["pixels"]).squeeze(-1)

    def param_groups(self):
        head = list(self.net.fc.parameters())
        ids = {id(p) for p in head}
        return [{"params": [p for p in self.parameters() if id(p) not in ids], "lr": CFG.ft_lr_cnn},
                {"params": head, "lr": CFG.ft_lr_head}]


class BertClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import AutoModel
        self.enc = AutoModel.from_pretrained(CFG.bert_name)
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(self.enc.config.hidden_size, 1))

    def forward(self, b):
        out = self.enc(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                       token_type_ids=b.get("token_type_ids"))
        return self.head(out.pooler_output).squeeze(-1)

    def param_groups(self):
        return [{"params": self.enc.parameters(), "lr": CFG.ft_lr_backbone},
                {"params": self.head.parameters(), "lr": CFG.ft_lr_head}]


class VisualBertClassifier(nn.Module):
    """VisualBERT (single-stream: caption tokens and region vectors share one transformer).
    A trainable linear layer maps torchvision's 1024-d box features to the 2048-d visual
    input the COCO-pretrained VisualBERT expects."""

    def __init__(self, region_dim: int):
        super().__init__()
        from transformers import VisualBertModel
        self.vb = VisualBertModel.from_pretrained(CFG.visualbert_name)
        vdim = self.vb.config.visual_embedding_dim
        self.proj = nn.Sequential(nn.Linear(region_dim, vdim), nn.LayerNorm(vdim))
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(self.vb.config.hidden_size, 1))

    def forward(self, b):
        vm = b["visual_mask"].to(b["attention_mask"].dtype)
        out = self.vb(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                      token_type_ids=b.get("token_type_ids"),
                      visual_embeds=self.proj(b["visual_embeds"]), visual_attention_mask=vm,
                      visual_token_type_ids=torch.ones_like(vm))
        return self.head(out.pooler_output).squeeze(-1)

    def param_groups(self):
        return [{"params": self.vb.parameters(), "lr": CFG.ft_lr_backbone},
                {"params": list(self.proj.parameters()) + list(self.head.parameters()), "lr": CFG.ft_lr_head}]


# --------------------------------------------------------------------------
# Cost: MACs counted with forward hooks (from scratch)
# --------------------------------------------------------------------------
def count_macs(model: nn.Module, batch: dict) -> float:
    """Multiply-accumulates of every nn.Linear and nn.Conv2d for ONE example.
    Attention score products inside BERT are not Linear layers and are left out
    (about 2 * L^2 * d per layer, small at 64 tokens)."""
    total = [0.0]

    def lin(m, i, o):
        total[0] += m.in_features * o.numel()

    def conv(m, i, o):
        k = m.kernel_size[0] * m.kernel_size[1] * (m.in_channels // m.groups)
        total[0] += k * o.numel()

    hooks = [m.register_forward_hook(lin if isinstance(m, nn.Linear) else conv)
             for m in model.modules() if isinstance(m, (nn.Linear, nn.Conv2d))]
    one = {k: v[:1] for k, v in batch.items() if k != "y"}
    model.eval()
    with torch.no_grad():
        model(one)
    for h in hooks:
        h.remove()
    return total[0]


# --------------------------------------------------------------------------
def make(name: str, splits: dict):
    """Return model and (train, val, test) loaders for one benchmark."""
    nw = min(4, CFG.num_workers or 2)
    if name == "image_finetuned":
        model = ImageClassifier()
        ds = {s: ImageDS(df, train=(s == "train")) for s, df in splits.items()}
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(CFG.bert_name)
        vis = name == "visual_bert"
        ds = {s: TextDS(df, tok, s, with_regions=vis) for s, df in splits.items()}
        if vis:
            model = VisualBertClassifier(region_dim=ds["train"].regions.shape[-1])
        else:
            model = BertClassifier()
        nw = 0 if not vis else nw
    loaders = {s: DataLoader(d, batch_size=CFG.ft_batch if s == "train" else 2 * CFG.ft_batch,
                             shuffle=(s == "train"), num_workers=nw, pin_memory=torch.cuda.is_available(),
                             drop_last=(s == "train"))
               for s, d in ds.items()}
    return model, loaders


def to_dev(b, device):
    return {k: v.to(device, non_blocking=True) for k, v in b.items()}


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs = []
    for b in loader:
        b = to_dev(b, device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
            logit = model(b)
        probs.append(torch.sigmoid(logit.float()).cpu())
    return torch.cat(probs).numpy()


def train_one(name: str, seed: int, splits: dict, device: str):
    seed_everything(seed)
    model, L = make(name, splits)
    model.to(device)
    y_tr = splits["train"]["label"].values
    pos_weight = torch.tensor((y_tr == 0).sum() / max(1, (y_tr == 1).sum()), device=device)
    opt = torch.optim.AdamW(model.param_groups(), weight_decay=CFG.weight_decay)
    total = len(L["train"]) * CFG.ft_epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(total, max(1, int(0.06 * total))))
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    hist = defaultdict(list)
    best_auc, best_state, best_epoch, bad, epoch_times = -1.0, None, -1, 0, []
    y_val = splits["val"]["label"].values
    for ep in range(CFG.ft_epochs):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for b in L["train"]:
            b = to_dev(b, device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
                logit = model(b)
            loss = bce_loss(logit.float(), b["y"], pos_weight)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot += float(loss); nb += 1
        if device == "cuda":
            torch.cuda.synchronize()
        epoch_times.append(time.time() - t0)
        auc = roc_auc_score(y_val, predict(model, L["val"], device))
        hist["train_loss"].append(tot / max(1, nb)); hist["val_auroc"].append(auc)
        print(f"    {name} seed{seed} ep{ep}: loss {tot / max(1, nb):.4f} val AUROC {auc:.4f}")
        if auc > best_auc:
            best_auc, best_epoch, bad = auc, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= CFG.ft_patience:
                break
    model.load_state_dict(best_state)

    t0 = time.time()
    pv, pt = predict(model, L["val"], device), predict(model, L["test"], device)
    if device == "cuda":
        torch.cuda.synchronize()
    infer_ms = 1000 * (time.time() - t0) / (len(pv) + len(pt))
    peak_mb = torch.cuda.max_memory_allocated() / 2 ** 20 if device == "cuda" else float("nan")
    macs = count_macs(model, to_dev(next(iter(L["val"])), device))

    tag = f"{name}_seed{seed}"
    np.savez_compressed(CFG.out_dir / "preds" / f"{tag}.npz", val_prob=pv, val_y=y_val,
                        test_prob=pt, test_y=splits["test"]["label"].values)
    run = {"model": name, "seed": seed, "best_epoch": best_epoch, "best_val_auroc": best_auc,
           "params": sum(p.numel() for p in model.parameters() if p.requires_grad),
           "macs_per_sample": macs, "sec_per_epoch": float(np.mean(epoch_times)),
           "infer_ms_per_sample": infer_ms, "peak_gpu_mb": peak_mb, "lambda_ccl": 0.0,
           "history": dict(hist)}
    save_json(run, CFG.out_dir / "runs" / f"{tag}.json")
    print(f"  {tag:28s} best val AUROC {best_auc:.4f} @ep{best_epoch} | {run['params'] / 1e6:.1f}M params")
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return run


FORCE = False


def main(models=None, seeds=None):
    CFG.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    splits = {s: load_index(s) for s in ["train", "val", "test"]}
    todo = [m for m in (models or FINETUNED_MODELS) if m in FINETUNED_MODELS]
    if "visual_bert" in todo and not (CFG.out_dir / "features" / "train_regions.npy").exists():
        from . import extract_regions
        print("[finetune] region features missing -> running extract_regions")
        extract_regions.main()
    print(f"[finetune] device={device}")
    for name in todo:
        print(f"[finetune] {name}")
        for seed in seeds or CFG.seeds:
            if (CFG.out_dir / "runs" / f"{name}_seed{seed}.json").exists() and not FORCE:
                print(f"  {name}_seed{seed} already done, skipping (use --force to retrain)")
                continue
            train_one(name, seed, splits, device)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.epochs:
        CFG.ft_epochs = a.epochs
    FORCE = a.force
    main(a.models, a.seeds)
