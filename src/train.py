"""
Step 3 - training every model on the cached CLIP features.

Run:  python -m src.train                          # all models, all seeds
      python -m src.train --models cagco hate_clipper --seeds 0 1

For each (model, seed) the script
  1. builds the model and an AdamW optimiser with warm-up + cosine decay,
  2. trains with BCE (+ the confounder contrastive loss for CAGCo-Net),
  3. keeps the epoch with the best validation AUROC (early stopping),
  4. writes val/test probabilities, history and cost measurements to
     outputs/preds/<model>_seed<k>.npz and outputs/runs/<model>_seed<k>.json.
"""
from __future__ import annotations

import argparse
import math
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from .config import CFG
from .data import get_pairs
from .losses import bce_loss, confounder_supcon
from .models import build_model, count_params, flops_per_sample
from .utils import CLIP_HEAD_MODELS, save_json, seed_everything

USES_CCL = {"cagco", "cagco_no_coattn"}            # cagco_no_ccl is the lambda=0 ablation


# --------------------------------------------------------------------------
# Feature store: whole split held as tensors on the training device
# --------------------------------------------------------------------------
class FeatureSplit:
    KEYS = {"img_tok": "img_tokens", "txt_tok": "txt_tokens", "txt_mask": "txt_mask",
            "img": "img_pooled", "txt": "txt_pooled"}

    def __init__(self, split: str, device: str):
        fdir = CFG.out_dir / "features"
        self.index = pd.read_csv(fdir / f"{split}_index.csv", dtype={"id": str})
        self.y = torch.tensor(self.index["label"].values, dtype=torch.long, device=device)
        self.t = {k: torch.from_numpy(np.load(fdir / f"{split}_{f}.npy")).to(device)
                  for k, f in self.KEYS.items()}
        self.n = len(self.index)

    def batch(self, idx: torch.Tensor) -> dict:
        b = {k: v[idx] for k, v in self.t.items()}
        for k in ("img_tok", "txt_tok", "img", "txt"):
            b[k] = b[k].float()                       # stored fp16, compute in fp32
        return b


# --------------------------------------------------------------------------
# Batch sampler that keeps confounder partners in the same mini-batch
# --------------------------------------------------------------------------
def partner_batches(n: int, partners: dict[int, list[int]], bs: int, rng: np.random.Generator):
    order = rng.permutation(n)
    used = np.zeros(n, dtype=bool)
    batch = []
    for i in order:
        if used[i]:
            continue
        batch.append(i); used[i] = True
        for j in partners.get(int(i), []):
            if not used[j] and len(batch) < bs:
                batch.append(j); used[j] = True
        if len(batch) >= bs:
            yield np.array(batch); batch = []
    if len(batch) > 1:
        yield np.array(batch)


def pair_mask_for(idx: np.ndarray, partners: dict, device) -> torch.Tensor:
    pos = {int(g): k for k, g in enumerate(idx)}
    m = torch.zeros(len(idx), len(idx), dtype=torch.bool)
    for k, g in enumerate(idx):
        for j in partners.get(int(g), []):
            if j in pos:
                m[k, pos[j]] = True
    return m.to(device)


# --------------------------------------------------------------------------
@torch.no_grad()
def predict(model, fs: FeatureSplit, bs: int = 256, keep_extra: bool = False):
    model.eval()
    probs, gates, attn = [], [], []
    for s in range(0, fs.n, bs):
        idx = torch.arange(s, min(s + bs, fs.n), device=fs.y.device)
        out = model(fs.batch(idx))
        probs.append(torch.sigmoid(out["logit"]).cpu())
        if keep_extra and "gate" in out:
            gates.append(out["gate"].cpu())
        if keep_extra and "attn_t2v" in out and s == 0:
            attn = out["attn_t2v"][:32].mean(1).cpu()       # avg heads, first 32 memes (32,L,50)
    res = {"prob": torch.cat(probs).numpy()}
    if gates:
        res["gate"] = torch.cat(gates).numpy()
    if keep_extra and len(attn):
        res["attn_t2v"] = attn.numpy()
    return res


def lr_lambda(total_steps: int, warmup: int):
    def f(step):
        if step < warmup:
            return (step + 1) / warmup
        p = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * p))
    return f


def train_one(name: str, seed: int, data: dict, partners: dict, device: str):
    seed_everything(seed)
    tr0 = data["train"]
    in_dim, n_img, n_txt = tr0.t["img"].shape[1], tr0.t["img_tok"].shape[1], tr0.t["txt_tok"].shape[1]
    model = build_model(name, CFG, in_dim).to(device)
    lam = CFG.lambda_ccl if name in USES_CCL else 0.0
    tr, va, te = data["train"], data["val"], data["test"]
    n_pos = int(tr.y.sum()); pos_weight = torch.tensor((tr.n - n_pos) / n_pos, device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    steps_per_epoch = math.ceil(tr.n / CFG.batch_size)
    total = steps_per_epoch * CFG.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(total, max(1, int(CFG.warmup_frac * total))))
    rng = np.random.default_rng(seed)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    hist = defaultdict(list)
    best_auc, best_state, best_epoch, bad = -1.0, None, -1, 0
    epoch_times = []
    for ep in range(CFG.epochs):
        model.train()
        t0 = time.time()
        tot, tot_bce, tot_ccl, nb = 0.0, 0.0, 0.0, 0
        for idx in partner_batches(tr.n, partners, CFG.batch_size, rng):
            it = torch.from_numpy(idx).to(device)
            b, y = tr.batch(it), tr.y[it]
            out = model(b)
            l_bce = bce_loss(out["logit"], y, pos_weight)
            loss = l_bce
            if lam > 0:
                l_ccl = confounder_supcon(out["z"], y, pair_mask_for(idx, partners, device),
                                          CFG.tau, CFG.beta_confounder)
                loss = loss + lam * l_ccl
                tot_ccl += float(l_ccl)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += float(loss); tot_bce += float(l_bce); nb += 1
        if device == "cuda":
            torch.cuda.synchronize()
        epoch_times.append(time.time() - t0)
        v = predict(model, va)
        auc = roc_auc_score(va.y.cpu().numpy(), v["prob"])
        hist["train_loss"].append(tot / nb); hist["train_bce"].append(tot_bce / nb)
        hist["train_ccl"].append(tot_ccl / nb); hist["val_auroc"].append(auc)
        if auc > best_auc:
            best_auc, best_epoch, bad = auc, ep, 0
            best_state = {k: v_.detach().clone() for k, v_ in model.state_dict().items()}
        else:
            bad += 1
            if bad >= CFG.patience:
                break
    model.load_state_dict(best_state)

    # ---- final predictions + inference timing
    t0 = time.time()
    pv = predict(model, va)
    pt = predict(model, te, keep_extra=True)
    if device == "cuda":
        torch.cuda.synchronize()
    infer_ms = 1000 * (time.time() - t0) / (va.n + te.n)
    peak_mb = torch.cuda.max_memory_allocated() / 2 ** 20 if device == "cuda" else float("nan")

    tag = f"{name}_seed{seed}"
    np.savez_compressed(CFG.out_dir / "preds" / f"{tag}.npz",
                        val_prob=pv["prob"], val_y=va.y.cpu().numpy(),
                        test_prob=pt["prob"], test_y=te.y.cpu().numpy(),
                        **{k: v for k, v in pt.items() if k != "prob"})
    run = {"model": name, "seed": seed, "best_epoch": best_epoch, "best_val_auroc": best_auc,
           "params": count_params(model),
           "macs_per_sample": flops_per_sample(name, CFG, n_img=n_img, n_txt=n_txt, in_dim=in_dim),
           "sec_per_epoch": float(np.mean(epoch_times)), "infer_ms_per_sample": infer_ms,
           "peak_gpu_mb": peak_mb, "lambda_ccl": lam, "history": dict(hist)}
    save_json(run, CFG.out_dir / "runs" / f"{tag}.json")
    print(f"  {tag:28s} best val AUROC {best_auc:.4f} @ep{best_epoch:2d} | "
          f"{run['params']/1e6:.2f}M params | {run['sec_per_epoch']:.1f}s/epoch")
    return run


FORCE = False


def main(models=None, seeds=None):
    CFG.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = {s: FeatureSplit(s, device) for s in ["train", "val", "test"]}

    # Recover confounder pairs on the training split and map ids -> row indices
    tr_index = data["train"].index
    pairs = get_pairs(tr_index, "train", CFG)
    row = {i: k for k, i in enumerate(tr_index["id"])}
    partners = defaultdict(list)
    for a, b in zip(pairs.id_a, pairs.id_b):
        if a in row and b in row:
            partners[row[a]].append(row[b]); partners[row[b]].append(row[a])
    print(f"[train] device={device} | {len(pairs)} confounder pairs | "
          f"{sum(1 for v in partners.values() if v)} train memes have a partner")

    for name in [m for m in (models or CLIP_HEAD_MODELS) if m in CLIP_HEAD_MODELS + ["image_only", "text_only"]]:
        print(f"[train] {name}")
        for seed in seeds or CFG.seeds:
            if (CFG.out_dir / "runs" / f"{name}_seed{seed}.json").exists() and not FORCE:
                print(f"  {name}_seed{seed} already done, skipping (use --force to retrain)")
                continue
            train_one(name, seed, data, partners, device)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.epochs:
        CFG.epochs = a.epochs
    FORCE = a.force
    main(a.models, a.seeds)
