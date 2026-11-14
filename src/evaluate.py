"""
Step 4 - evaluation, statistics and figures (Criterion 6 + complexity for Criterion 5).

Run:  python -m src.evaluate

Reads outputs/preds/*.npz and outputs/runs/*.json written by train.py and produces
  outputs/tables/metrics_per_seed.csv      every model x seed x metric
  outputs/tables/metrics_summary.csv       mean, std, 95% CI across seeds
  outputs/tables/significance.csv          CAGCo-Net vs each other model, 4 tests + Holm
  outputs/tables/complexity.csv            params, MACs, time, memory
  outputs/tables/category_accuracy.csv     accuracy on recovered confounder categories
  outputs/figures/res_*.png                ROC, AUROC CI plot, confusion matrices,
                                           learning curves, bootstrap, gate, attention
  outputs/RESULTS.md                       all tables in one markdown file for the report
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_curve)

from .config import CFG
from .data import PROPOSAL_CATEGORIES, get_pairs, proposal_category
from .stats_tests import (auroc, delong_test, holm, mcnemar_exact, paired_bootstrap, seed_tests, t_ci)
from .utils import (CLASS_COLORS, CLASS_NAMES, MODEL_COLORS, MODEL_LABELS, MODEL_ORDER, load_json,
                    plt, set_style)

OUT = CFG.out_dir
FIG = OUT / "figures"
TAB = OUT / "tables"
OURS = "cagco"


def best_threshold(y, p) -> float:
    """Threshold that maximises macro-F1 on the validation set."""
    grid = np.unique(np.quantile(p, np.linspace(0.02, 0.98, 97)))
    scores = [f1_score(y, (p >= t).astype(int), average="macro") for t in grid]
    return float(grid[int(np.argmax(scores))])


def metrics(y, p, thr) -> dict:
    yhat = (p >= thr).astype(int)
    return {"auroc": auroc(y, p), "accuracy": accuracy_score(y, yhat),
            "precision": precision_score(y, yhat, zero_division=0),
            "recall": recall_score(y, yhat, zero_division=0),
            "macro_f1": f1_score(y, yhat, average="macro")}


def load_all():
    preds, runs = {}, {}
    for f in sorted(glob.glob(str(OUT / "preds" / "*.npz"))):
        tag = Path(f).stem
        preds[tag] = dict(np.load(f))
        runs[tag] = load_json(OUT / "runs" / f"{tag}.json")
    models = [m for m in MODEL_ORDER if any(t.startswith(m + "_seed") for t in preds)]
    return preds, runs, models


def main():
    set_style()
    preds, runs, models = load_all()
    assert OURS in models, "train CAGCo-Net first (python -m src.train)"
    seeds = sorted({r["seed"] for r in runs.values()})

    # ------------------------------------------------ per-seed metrics
    rows = []
    for m in models:
        for s in seeds:
            tag = f"{m}_seed{s}"
            if tag not in preds:
                continue
            d = preds[tag]
            thr = best_threshold(d["val_y"], d["val_prob"])
            rows.append({"model": m, "seed": s, "threshold": thr, **metrics(d["test_y"], d["test_prob"], thr)})
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(TAB / "metrics_per_seed.csv", index=False)

    summ = []
    for m, g in per_seed.groupby("model", sort=False):
        r = {"model": m}
        for k in ["auroc", "accuracy", "precision", "recall", "macro_f1"]:
            lo, hi = t_ci(g[k].values, CFG.alpha)
            r[f"{k}_mean"], r[f"{k}_std"], r[f"{k}_ci_low"], r[f"{k}_ci_high"] = \
                g[k].mean(), g[k].std(ddof=1), lo, hi
        summ.append(r)
    summary = pd.DataFrame(summ).set_index("model").loc[models]
    summary.to_csv(TAB / "metrics_summary.csv")

    # ------------------------------------------------ seed ensembles (mean prob)
    y_test = preds[f"{OURS}_seed{seeds[0]}"]["test_y"]
    ens, ens_thr = {}, {}
    for m in models:
        tags = [f"{m}_seed{s}" for s in seeds if f"{m}_seed{s}" in preds]
        ens[m] = np.mean([preds[t]["test_prob"] for t in tags], 0)
        val = np.mean([preds[t]["val_prob"] for t in tags], 0)
        ens_thr[m] = best_threshold(preds[tags[0]]["val_y"], val)

    # ------------------------------------------------ significance: ours vs each
    sig = []
    boot_store = {}
    for m in models:
        if m == OURS:
            continue
        a = per_seed[per_seed.model == OURS].sort_values("seed")["auroc"].values
        b = per_seed[per_seed.model == m].sort_values("seed")["auroc"].values
        st = seed_tests(a, b) if len(a) == len(b) and len(a) > 1 else {}
        dl = delong_test(y_test, ens[OURS], ens[m])
        bs = paired_bootstrap(y_test, ens[OURS], ens[m], CFG.n_bootstrap)
        boot_store[m] = bs.pop("samples")
        mc = mcnemar_exact(y_test, (ens[OURS] >= ens_thr[OURS]).astype(int), (ens[m] >= ens_thr[m]).astype(int))
        sig.append({"vs": m, "auroc_diff_seedmean": st.get("mean_diff"), "p_paired_t": st.get("p_ttest"),
                    "p_wilcoxon": st.get("p_wilcoxon"), "delong_z": dl["z"], "p_delong": dl["p"],
                    "boot_diff": bs["diff"], "boot_ci_low": bs["ci_low"], "boot_ci_high": bs["ci_high"],
                    "p_bootstrap": bs["p"], "mcnemar_b": mc["b_1right_2wrong"], "mcnemar_c": mc["c_1wrong_2right"],
                    "p_mcnemar": mc["p"]})
    sig = pd.DataFrame(sig)
    for col in ["p_paired_t", "p_delong", "p_bootstrap", "p_mcnemar"]:
        if sig[col].notna().all():
            sig[col + "_holm"] = holm(sig[col].tolist())
    sig.to_csv(TAB / "significance.csv", index=False)

    # ------------------------------------------------ complexity
    comp = []
    for m in models:
        rs = [runs[f"{m}_seed{s}"] for s in seeds if f"{m}_seed{s}" in runs]
        comp.append({"model": m, "params": rs[0]["params"], "MACs_per_sample": rs[0]["macs_per_sample"],
                     "sec_per_epoch": np.mean([r["sec_per_epoch"] for r in rs]),
                     "infer_ms_per_sample": np.mean([r["infer_ms_per_sample"] for r in rs]),
                     "peak_gpu_mb": np.mean([r["peak_gpu_mb"] for r in rs]),
                     "best_epoch_mean": np.mean([r["best_epoch"] for r in rs])})
    comp = pd.DataFrame(comp)
    comp.to_csv(TAB / "complexity.csv", index=False)

    # ------------------------------------------------ recovered categories on test
    idx = pd.read_csv(OUT / "features" / "test_index.csv", dtype={"id": str})
    tpairs = get_pairs(idx, "test", CFG)
    cat = proposal_category(idx, tpairs)
    cat_rows = []
    for m in models:
        yhat = (ens[m] >= ens_thr[m]).astype(int)
        for c in PROPOSAL_CATEGORIES:
            sel = cat == c
            if sel.sum() == 0:
                continue
            cat_rows.append({"model": m, "category": c, "n": int(sel.sum()),
                             "accuracy": accuracy_score(y_test[sel], yhat[sel])})
    cat_df = pd.DataFrame(cat_rows)
    cat_df.to_csv(TAB / "category_accuracy.csv", index=False)

    # ================================================= FIGURES
    # 1. ROC curves (seed ensembles)
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    for m in models:
        fpr, tpr, _ = roc_curve(y_test, ens[m])
        ax.plot(fpr, tpr, color=MODEL_COLORS[m], lw=2.6 if m == OURS else 1.6,
                label=f"{MODEL_LABELS[m]} ({auroc(y_test, ens[m]):.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="#b9b8b2", lw=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title=f"ROC on test ({len(seeds)}-seed ensemble)")
    ax.legend(fontsize=7.5, loc="lower right")
    fig.savefig(FIG / "res_roc.png"); plt.close(fig)

    # 2. AUROC: mean +/- 95% CI across seeds, individual seeds as faint dots
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for i, m in enumerate(models):
        g = per_seed[per_seed.model == m]["auroc"]
        lo, hi = summary.loc[m, "auroc_ci_low"], summary.loc[m, "auroc_ci_high"]
        ax.plot([lo, hi], [i, i], color=MODEL_COLORS[m], lw=2)
        ax.scatter(g, [i] * len(g), s=12, color=MODEL_COLORS[m], alpha=0.35)
        ax.scatter([g.mean()], [i], s=60, color=MODEL_COLORS[m], edgecolor="#fcfcfb", linewidth=2, zorder=3)
    ax.set_yticks(range(len(models)), [MODEL_LABELS[m] for m in models])
    ax.invert_yaxis()
    ax.set_xlabel("Test AUROC (dot = mean, line = 95% CI over seeds)")
    ax.set_title("AUROC by model")
    fig.savefig(FIG / "res_auroc_ci.png"); plt.close(fig)

    # 3. Confusion matrices
    cols = 3
    rws = int(np.ceil(len(models) / cols))
    fig, axes = plt.subplots(rws, cols, figsize=(3 * cols, 2.9 * rws), squeeze=False)
    for ax, m in zip(axes.flat, models):
        cm = confusion_matrix(y_test, (ens[m] >= ens_thr[m]).astype(int))
        ax.imshow(cm, cmap="Blues", vmin=0)
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center",
                    color="#fcfcfb" if v > cm.max() / 2 else "#0b0b0b", fontsize=10)
        ax.set_xticks([0, 1], ["pred 0", "pred 1"]); ax.set_yticks([0, 1], ["true 0", "true 1"])
        ax.set_title(MODEL_LABELS[m], fontsize=9); ax.grid(False)
    for ax in list(axes.flat)[len(models):]:
        ax.axis("off")
    fig.suptitle("Confusion matrices on test (threshold tuned on val)")
    fig.savefig(FIG / "res_confusion.png"); plt.close(fig)

    # 4. Learning curves (seed 0)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for m in models:
        tag = f"{m}_seed{seeds[0]}"
        if tag in runs:
            h = runs[tag]["history"]["val_auroc"]
            ax.plot(range(1, len(h) + 1), h, color=MODEL_COLORS[m], label=MODEL_LABELS[m],
                    lw=2.4 if m == OURS else 1.4)
    ax.set(xlabel="Epoch", ylabel="Validation AUROC", title=f"Validation AUROC per epoch (seed {seeds[0]})")
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(FIG / "res_learning_curves.png"); plt.close(fig)

    # 5. Bootstrap distribution vs strongest benchmark
    strongest = max([m for m in models if not m.startswith("cagco")],
                    key=lambda m: summary.loc[m, "auroc_mean"])
    d = boot_store[strongest]
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.hist(d, bins=50, color=MODEL_COLORS[OURS], alpha=0.75)
    ax.axvline(0, color="#0b0b0b", lw=1)
    ax.axvline(np.percentile(d, 2.5), color="#52514e", ls="--", lw=1)
    ax.axvline(np.percentile(d, 97.5), color="#52514e", ls="--", lw=1)
    ax.set(xlabel=f"AUROC(CAGCo-Net) - AUROC({MODEL_LABELS[strongest]})", ylabel="Bootstrap samples",
           title=f"Paired bootstrap ({CFG.n_bootstrap} resamples), dashed = 95% CI")
    fig.savefig(FIG / "res_bootstrap.png"); plt.close(fig)

    # 6. Accuracy on recovered confounder categories
    if len(cat_df):
        cats = [c for c in PROPOSAL_CATEGORIES if c in set(cat_df.category)]
        fig, ax = plt.subplots(figsize=(9.6, 4.0))
        w = 0.8 / len(models)
        for k, m in enumerate(models):
            vals = [cat_df[(cat_df.model == m) & (cat_df.category == c)]["accuracy"].values[0] for c in cats]
            ax.bar(np.arange(len(cats)) + (k - len(models) / 2 + 0.5) * w, vals, w * 0.92,
                   color=MODEL_COLORS[m], label=MODEL_LABELS[m])
        ns = cat_df.groupby("category")["n"].first()
        ax.set_xticks(range(len(cats)), [f"{c.replace(' (', chr(10) + '(')}\nn={ns[c]}" for c in cats], fontsize=8)
        ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1)
        ax.set_title("Accuracy on test memes by recovered sub-category")
        ax.legend(fontsize=6.5, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.24))
        fig.savefig(FIG / "res_category_accuracy.png"); plt.close(fig)

    # 7. Gate analysis (CAGCo-Net, seed 0)
    g0 = preds[f"{OURS}_seed{seeds[0]}"]
    if "gate" in g0:
        gate = g0["gate"]
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
        for lab in [0, 1]:
            axes[0].hist(gate[y_test == lab], bins=30, alpha=0.6, density=True, color=CLASS_COLORS[lab],
                         label=CLASS_NAMES[lab])
        axes[0].set(xlabel="Gate g (1 = image stream, 0 = text stream)", ylabel="Density",
                    title="Modality gate by true label")
        axes[0].legend()
        cg = pd.DataFrame({"gate": gate, "cat": cat}).groupby("cat")["gate"].agg(["mean", "std", "count"])
        axes[1].barh(cg.index, cg["mean"], xerr=cg["std"], color="#2a78d6", capsize=4)
        axes[1].set(xlabel="Mean gate (± sd)", title="Gate by recovered sub-category")
        fig.savefig(FIG / "res_gate.png"); plt.close(fig)
        cg.to_csv(TAB / "gate_by_category.csv")

    # 8. Text->image attention overlays (CAGCo-Net, seed 0, first 4 test memes)
    if "attn_t2v" in g0:
        att = g0["attn_t2v"]                                   # (32, L, 50)
        mask = np.load(OUT / "features" / "test_txt_mask.npy", mmap_mode="r")[:len(att)]
        fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
        for k, ax in enumerate(axes):
            a = att[k][mask[k]].mean(0)[1:]                     # avg over real tokens, drop CLS
            side = int(np.sqrt(len(a)))
            heat = a.reshape(side, side)
            im = Image.open(CFG.data_dir / idx.loc[k, "img"]).convert("RGB")
            s = 224 / min(im.size)                               # CLIP: resize short side, centre-crop
            im = im.resize((round(im.size[0] * s), round(im.size[1] * s)))
            l, t = (im.size[0] - 224) // 2, (im.size[1] - 224) // 2
            im = im.crop((l, t, l + 224, t + 224))
            ax.imshow(im)
            ax.imshow(np.kron(heat, np.ones((32, 32))), cmap="inferno", alpha=0.45)
            ax.set_title(f"{CLASS_NAMES[int(y_test[k])]} | p={ens[OURS][k]:.2f}", fontsize=9)
            ax.axis("off")
        fig.suptitle("Where the caption looks: text-to-image co-attention (CAGCo-Net)")
        fig.savefig(FIG / "res_attention.png"); plt.close(fig)

    write_markdown(summary, sig, comp, cat_df, models, strongest)
    print("[evaluate] done ->", OUT / "RESULTS.md")


def write_markdown(summary, sig, comp, cat_df, models, strongest):
    L = ["# Results (auto-generated by src/evaluate.py)\n",
         "## Test metrics, mean ± std over seeds (95% CI for AUROC)\n",
         "| Model | AUROC | 95% CI | Accuracy | Precision | Recall | Macro-F1 |",
         "|---|---|---|---|---|---|---|"]
    for m in models:
        r = summary.loc[m]
        L.append(f"| {MODEL_LABELS[m]} | {r.auroc_mean:.4f} ± {r.auroc_std:.4f} | "
                 f"[{r.auroc_ci_low:.4f}, {r.auroc_ci_high:.4f}] | {r.accuracy_mean:.4f} ± {r.accuracy_std:.4f} | "
                 f"{r.precision_mean:.4f} | {r.recall_mean:.4f} | {r.macro_f1_mean:.4f} ± {r.macro_f1_std:.4f} |")
    L += ["\n## CAGCo-Net vs each model (Holm-adjusted p in brackets)\n",
          "| vs | ΔAUROC (seed mean) | paired t p | DeLong p | bootstrap ΔAUROC [95% CI] | bootstrap p | McNemar b/c | McNemar p |",
          "|---|---|---|---|---|---|---|---|"]
    for r in sig.itertuples():
        hp = lambda c: f" ({getattr(r, c + '_holm'):.3g})" if hasattr(r, c + "_holm") else ""
        L.append(f"| {MODEL_LABELS[r.vs]} | {r.auroc_diff_seedmean:+.4f} | {r.p_paired_t:.3g}{hp('p_paired_t')} | "
                 f"{r.p_delong:.3g}{hp('p_delong')} | {r.boot_diff:+.4f} [{r.boot_ci_low:+.4f}, {r.boot_ci_high:+.4f}] | "
                 f"{r.p_bootstrap:.3g}{hp('p_bootstrap')} | {r.mcnemar_b}/{r.mcnemar_c} | {r.p_mcnemar:.3g}{hp('p_mcnemar')} |")
    L += ["\n## Cost\n", "| Model | Params (trainable) | MACs/meme (trainable part) | s/epoch | ms/meme inference | Peak GPU MB |",
          "|---|---|---|---|---|---|"]
    for r in comp.itertuples():
        L.append(f"| {MODEL_LABELS[r.model]} | {r.params:,} | {r.MACs_per_sample:,.0f} | {r.sec_per_epoch:.2f} | "
                 f"{r.infer_ms_per_sample:.3f} | {r.peak_gpu_mb:.0f} |")
    if len(cat_df):
        L += ["\n## Accuracy by recovered sub-category (test)\n"]
        piv = cat_df.pivot(index="model", columns="category", values="accuracy").loc[models]
        piv = piv[[c for c in PROPOSAL_CATEGORIES if c in piv.columns]]
        L.append("| Model | " + " | ".join(piv.columns) + " |")
        L.append("|---" * (len(piv.columns) + 1) + "|")
        for m, r in piv.iterrows():
            L.append(f"| {MODEL_LABELS[m]} | " + " | ".join(f"{v:.3f}" for v in r.values) + " |")
    L.append(f"\nStrongest non-CAGCo benchmark by mean AUROC: **{MODEL_LABELS[strongest]}**.")
    (OUT / "RESULTS.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
