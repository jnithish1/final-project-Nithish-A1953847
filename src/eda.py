"""
Exploratory data analysis (Criterion 1).

Run:  python -m src.eda                 (raw-data figures, needs only images + jsonl)
      python -m src.eda --embeddings    (adds the CLIP t-SNE; run after extract_features)

Figures go to outputs/figures/eda_*.png and the numbers behind them to
outputs/tables/eda_summary.json, so the report can quote them directly.
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats

from .config import CFG
from .data import PROPOSAL_CATEGORIES, get_pairs, load_splits, proposal_category
from .utils import CLASS_COLORS, CLASS_NAMES, save_json, set_style, plt

FIG = CFG.out_dir / "figures"
STOP = set("""a an the and or but if of to in on at for with by from is are was were be been
being it its it's this that these those i you he she we they me him her us them my your his our
their what when where who whom which why how not no do does did have has had just so as than
too very can will would should could all any some more most into up down out about over again
then there here only own same s t don dont im""".split())


# ---------------------------------------------------------------- helpers
def image_stats(path: Path) -> tuple[int, int, float, float]:
    """width, height, mean brightness (0-255), Hasler-Suesstrunk colourfulness."""
    im = Image.open(path)
    w, h = im.size
    small = np.asarray(im.convert("RGB").resize((64, 64)), dtype=np.float32)
    r, g, b = small[..., 0], small[..., 1], small[..., 2]
    bright = float((0.299 * r + 0.587 * g + 0.114 * b).mean())
    rg, yb = r - g, 0.5 * (r + g) - b
    colour = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))
    return w, h, bright, colour


def tokens(t: str) -> list[str]:
    return [w for w in re.findall(r"[a-z']+", str(t).lower()) if w not in STOP and len(w) > 2]


def log_odds_dirichlet(c1: Counter, c0: Counter, prior: Counter, a0: float = 500.0):
    """Monroe et al. (2008) weighted log-odds with an informative Dirichlet prior.
    Returns z-scores: positive -> word is over-represented in class 1."""
    n1, n0, np_ = sum(c1.values()), sum(c0.values()), sum(prior.values())
    out = {}
    for w in set(c1) | set(c0):
        aw = a0 * prior[w] / np_
        l1 = np.log((c1[w] + aw) / (n1 + a0 - c1[w] - aw))
        l0 = np.log((c0[w] + aw) / (n0 + a0 - c0[w] - aw))
        var = 1.0 / (c1[w] + aw) + 1.0 / (c0[w] + aw)
        out[w] = (l1 - l0) / np.sqrt(var)
    return pd.Series(out).sort_values()


TOKEN_UNIT = "CLIP tokens"


def caption_token_lengths(texts: list[str]) -> np.ndarray:
    """Number of CLIP BPE tokens per caption (without BOS/EOS). Falls back to
    whitespace words if transformers is not installed."""
    global TOKEN_UNIT
    try:
        from transformers import CLIPTokenizer
        tok = CLIPTokenizer.from_pretrained(CFG.clip_name)
        ids = tok([str(t) for t in texts], add_special_tokens=False)["input_ids"]
        TOKEN_UNIT = "CLIP tokens"
        return np.array([len(i) for i in ids])
    except Exception as e:                      # offline / not installed
        print("[eda] CLIP tokenizer unavailable, using words:", type(e).__name__)
        TOKEN_UNIT = "words"
        return np.array([len(str(t).split()) for t in texts])


def lexicon_score(texts, weights: pd.Series, z_min: float = 1.96) -> np.ndarray:
    """Toxicity-lexicon score: mean weighted log-odds z of the caption's words that are in
    the lexicon (|z| >= 1.96 on train). Positive = words typical of hateful captions.
    The lexicon is learned from training captions only (Monroe et al., 2008 weighting)."""
    lex = weights[weights.abs() >= z_min]
    if len(lex) < 20:                                    # tiny / noisy data: use the 100 strongest words
        lex = weights.reindex(weights.abs().sort_values().index[-100:])
    lex = lex.to_dict()
    out = []
    for t in texts:
        hits = [lex[w] for w in tokens(t) if w in lex]
        out.append(float(np.mean(hits)) if hits else 0.0)
    return np.array(out)


# ---------------------------------------------------------------- main EDA
def run_raw_eda():
    set_style()
    CFG.ensure_dirs()
    splits = load_splits(CFG.data_dir)
    df = pd.concat(splits.values(), ignore_index=True)
    summary = {}

    # ---- 1. class distribution per split
    tab = pd.crosstab(df["split"], df["label"]).reindex(["train", "val", "test"])
    summary["class_counts"] = tab.to_dict()
    summary["hateful_share"] = (tab[1] / tab.sum(1)).round(4).to_dict()
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    x = np.arange(len(tab))
    for k, lab in enumerate([0, 1]):
        bars = ax.bar(x + (k - 0.5) * 0.38, tab[lab], 0.36, color=CLASS_COLORS[lab], label=CLASS_NAMES[lab])
        ax.bar_label(bars, fontsize=8, color="#52514e", padding=2)
    ax.set_xticks(x, tab.index)
    ax.set_ylabel("Memes")
    ax.set_title("Class distribution per split")
    ax.legend()
    fig.savefig(FIG / "eda_class_distribution.png"); plt.close(fig)

    # ---- 2. caption token length by label (CLIP BPE tokens, the unit the model sees)
    df["n_words"] = caption_token_lengths(df["text"].tolist())
    summary["caption_words"] = df.groupby("label")["n_words"].describe().round(2).to_dict()
    u = stats.mannwhitneyu(df.loc[df.label == 1, "n_words"], df.loc[df.label == 0, "n_words"])
    summary["caption_len_mannwhitney_p"] = float(u.pvalue)
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    bins = np.arange(0, df["n_words"].quantile(0.99) + 2)
    for lab in [0, 1]:
        ax.hist(df.loc[df.label == lab, "n_words"], bins=bins, density=True, histtype="step", lw=2,
                color=CLASS_COLORS[lab], label=CLASS_NAMES[lab])
    ax.set_xlabel(f"Caption length ({TOKEN_UNIT})"); ax.set_ylabel("Density")
    ax.set_title(f"Caption length by label (Mann-Whitney p={u.pvalue:.3g})")
    ax.legend()
    fig.savefig(FIG / "eda_caption_length.png"); plt.close(fig)

    # ---- 3. image properties
    print("[eda] reading image statistics ...")
    st = [image_stats(CFG.data_dir / p) for p in df["img"]]
    df[["width", "height", "brightness", "colourfulness"]] = pd.DataFrame(st, index=df.index)
    df["aspect"] = df["width"] / df["height"]
    summary["image_size"] = df[["width", "height", "aspect"]].describe().round(2).to_dict()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for lab in [0, 1]:
        d = df[df.label == lab]
        axes[0].scatter(d.width, d.height, s=4, alpha=0.3, color=CLASS_COLORS[lab], label=CLASS_NAMES[lab])
        axes[1].hist(d.brightness, bins=40, density=True, histtype="step", lw=2, color=CLASS_COLORS[lab])
        axes[2].hist(d.colourfulness, bins=40, density=True, histtype="step", lw=2, color=CLASS_COLORS[lab])
    axes[0].set(xlabel="Width (px)", ylabel="Height (px)", title="Image resolution")
    axes[1].set(xlabel="Mean brightness (0-255)", title="Brightness by label")
    axes[2].set(xlabel="Colourfulness (Hasler-Suesstrunk)", title="Colourfulness by label")
    axes[0].legend(markerscale=4)
    for name, col in [("brightness", "brightness"), ("colourfulness", "colourfulness")]:
        summary[f"{name}_ks_p"] = float(stats.ks_2samp(df.loc[df.label == 1, col], df.loc[df.label == 0, col]).pvalue)
    fig.savefig(FIG / "eda_image_properties.png"); plt.close(fig)

    # ---- 4. distinctive words (log-odds) — shows how much text alone carries
    c1 = Counter(w for t in df.loc[(df.split == "train") & (df.label == 1), "text"] for w in tokens(t))
    c0 = Counter(w for t in df.loc[(df.split == "train") & (df.label == 0), "text"] for w in tokens(t))
    z = log_odds_dirichlet(c1, c0, c1 + c0)
    top = pd.concat([z.head(12), z.tail(12)])
    summary["top_words_hateful"] = z.tail(15)[::-1].round(2).to_dict()
    summary["top_words_not_hateful"] = z.head(15).round(2).to_dict()
    fig, ax = plt.subplots(figsize=(6.2, 6))
    ax.barh(top.index, top.values, color=[CLASS_COLORS[int(v > 0)] for v in top.values])
    ax.axvline(0, color="#52514e", lw=0.8)
    ax.set_xlabel("Weighted log-odds z-score (positive = hateful)")
    ax.set_title("Words most associated with each label (train)")
    fig.savefig(FIG / "eda_log_odds_words.png"); plt.close(fig)

    # ---- 5. recovered confounders -> the five sub-categories of the paper, per split
    pairs_by_split, cats = {}, []
    for name, d in splits.items():
        pairs_by_split[name] = get_pairs(d, name, CFG)
        cats.append(proposal_category(d, pairs_by_split[name]))
    df["category"] = np.concatenate(cats)          # same order as the concat above
    pairs = pairs_by_split["train"]
    tr = df[df.split == "train"].copy()
    summary["recovered_pairs"] = {k: v["kind"].value_counts().to_dict() for k, v in pairs_by_split.items()}
    ct_all = pd.crosstab(df["split"], df["category"]).reindex(index=["train", "val", "test"],
                                                              columns=PROPOSAL_CATEGORIES, fill_value=0)
    summary["subcategory_counts"] = ct_all.to_dict()
    share = ct_all.div(ct_all.sum(1), axis=0)
    cat_colors = ["#eb6834", "#e87ba4", "#1baf7a", "#2a78d6", "#b9b8b2"]
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    left = np.zeros(len(share))
    for c, col in zip(PROPOSAL_CATEGORIES, cat_colors):
        ax.barh(share.index, share[c], left=left, color=col, label=c, edgecolor="#fcfcfb", linewidth=2)
        left += share[c].values
    for i, n in enumerate(ct_all.sum(1)):
        ax.text(1.01, i, f"n={n}", va="center", fontsize=8, color="#52514e")
    ax.set_xlim(0, 1.1); ax.invert_yaxis(); ax.set_xlabel("Share of memes")
    ax.set_title("Recovered sub-categories per split (Kiela et al. taxonomy)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=8)
    fig.savefig(FIG / "eda_subcategories_by_split.png"); plt.close(fig)

    # ---- 5b. toxicity-lexicon score distribution across sub-categories
    df["lexicon_score"] = lexicon_score(df["text"], z)
    try:
        kw = stats.kruskal(*[g["lexicon_score"].values for _, g in df.groupby("category") if len(g) > 1])
    except ValueError:                                   # all scores identical
        kw = type("KW", (), {"pvalue": float("nan")})()
    summary["lexicon_score_by_category"] = df.groupby("category")["lexicon_score"].describe().round(3).to_dict()
    summary["lexicon_kruskal_p"] = float(kw.pvalue)
    present = [c for c in PROPOSAL_CATEGORIES if (df.category == c).sum() > 0]
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    bp = ax.boxplot([df.loc[df.category == c, "lexicon_score"] for c in present], vert=False,
                    patch_artist=True, widths=0.55, showfliers=False)
    for patch, c in zip(bp["boxes"], present):
        patch.set_facecolor(cat_colors[PROPOSAL_CATEGORIES.index(c)]); patch.set_alpha(0.75)
    for med in bp["medians"]:
        med.set_color("#0b0b0b")
    ax.set_yticks(range(1, len(present) + 1), present)
    ax.axvline(0, color="#52514e", lw=0.8)
    ax.set_xlabel("Toxicity-lexicon score (mean log-odds z of lexicon words)")
    ax.set_title(f"Lexicon score by sub-category, all splits (Kruskal-Wallis p={kw.pvalue:.3g})")
    fig.savefig(FIG / "eda_lexicon_by_subcategory.png"); plt.close(fig)

    # ---- 6. example confounder pairs (image grid)
    ex = pd.concat([pairs[pairs.kind == "text_swap"].head(2), pairs[pairs.kind == "image_swap"].head(2)])
    if len(ex):
        lookup = tr.set_index("id")
        fig, axes = plt.subplots(len(ex), 2, figsize=(7, 3.2 * len(ex)), squeeze=False)
        for r, row in enumerate(ex.itertuples()):
            for c, mid in enumerate([row.id_a, row.id_b]):
                m = lookup.loc[mid]
                axes[r, c].imshow(Image.open(CFG.data_dir / m.img).convert("RGB"))
                axes[r, c].set_title(f"{row.kind} | {CLASS_NAMES[m.label]}", fontsize=9,
                                     color=CLASS_COLORS[m.label])
                axes[r, c].axis("off")
        fig.suptitle("Recovered confounder pairs: one edit flips the label")
        fig.savefig(FIG / "eda_confounder_examples.png"); plt.close(fig)

    df.drop(columns=["img"]).to_csv(CFG.out_dir / "tables" / "eda_per_meme.csv", index=False)
    save_json(summary, CFG.out_dir / "tables" / "eda_summary.json")
    print("[eda] done ->", FIG)


def run_embedding_eda():
    """t-SNE of frozen CLIP image, text and (image ⊙ text) embeddings, coloured by label."""
    from sklearn.manifold import TSNE
    set_style()
    fdir = CFG.out_dir / "features"
    ids = pd.read_csv(fdir / "train_index.csv", dtype={"id": str})
    img = np.load(fdir / "train_img_pooled.npy").astype(np.float32)
    txt = np.load(fdir / "train_txt_pooled.npy").astype(np.float32)
    img /= np.linalg.norm(img, axis=1, keepdims=True)
    txt /= np.linalg.norm(txt, axis=1, keepdims=True)
    rng = np.random.default_rng(0)
    sel = rng.choice(len(ids), size=min(3000, len(ids)), replace=False)
    y = ids["label"].values[sel]
    views = {"Image embedding": img[sel], "Text embedding": txt[sel], "Image ⊙ text": img[sel] * txt[sel]}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, (name, X) in zip(axes, views.items()):
        P = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(X)
        for lab in [0, 1]:
            ax.scatter(P[y == lab, 0], P[y == lab, 1], s=4, alpha=0.5, color=CLASS_COLORS[lab],
                       label=CLASS_NAMES[lab])
        ax.set_title(name); ax.set_xticks([]); ax.set_yticks([])
    axes[0].legend(markerscale=4)
    fig.suptitle("t-SNE of frozen CLIP features, coloured by label (3,000 train memes)")
    fig.savefig(FIG / "eda_clip_tsne.png"); plt.close(fig)
    print("[eda] t-SNE saved")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", action="store_true")
    a = ap.parse_args()
    if a.embeddings:
        run_embedding_eda()
    else:
        run_raw_eda()
