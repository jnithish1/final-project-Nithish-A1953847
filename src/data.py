"""
Data loading, split construction and confounder-pair recovery.

Hateful Memes ships as an img/ folder plus JSON-lines files. Each line is
    {"id": 42953, "img": "img/42953.png", "label": 0, "text": "its their character not their color that matters"}

Different mirrors of the dataset name the files slightly differently
(dev.jsonl vs dev_seen.jsonl, test.jsonl with or without labels), so
`load_splits` looks for whichever files exist and builds three labelled
splits: train / val / test.

The public release does not include the paper's sub-category tags
(unimodal hate, benign text confounder, ...). We recover the two confounder
types approximately:

  * text-swap pair  : same image (perceptual hash within a few bits),
                      different caption, opposite labels
  * image-swap pair : same caption (after normalisation),
                      different image, opposite labels

These pairs drive the confounder-weighted contrastive loss and the EDA.
The perceptual hash (dHash) is written from scratch with Pillow + NumPy.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


# --------------------------------------------------------------------------
# 1. Reading the jsonl files
# --------------------------------------------------------------------------
def read_jsonl(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    df = pd.DataFrame(rows)
    df["id"] = df["id"].astype(str).str.zfill(5)
    if "label" not in df.columns:
        df["label"] = -1                        # unlabelled test file
    df["label"] = df["label"].astype(int)
    return df


def _first_existing(data_dir: Path, names: list[str]) -> Path | None:
    for n in names:
        p = data_dir / n
        if p.exists():
            return p
    return None


def _labelled(df: pd.DataFrame | None) -> bool:
    return df is not None and (df["label"] >= 0).all()


def load_splits(data_dir: Path, seed: int = 42) -> dict[str, pd.DataFrame]:
    """Return {"train", "val", "test"} DataFrames, all labelled.

    Priority:
      1. test_seen.jsonl has labels  -> val = dev_seen, test = test_seen
      2. dev_unseen.jsonl exists     -> val = dev_unseen, test = dev_seen
      3. otherwise                   -> test = dev(_seen), val = group-aware 10% of train
    """
    data_dir = Path(data_dir)
    train = read_jsonl(data_dir / "train.jsonl")
    p_dev = _first_existing(data_dir, ["dev_seen.jsonl", "dev.jsonl"])
    p_dev_un = _first_existing(data_dir, ["dev_unseen.jsonl"])
    p_test = _first_existing(data_dir, ["test_seen.jsonl", "test.jsonl"])
    dev = read_jsonl(p_dev) if p_dev else None
    dev_un = read_jsonl(p_dev_un) if p_dev_un else None
    test = read_jsonl(p_test) if p_test else None

    if _labelled(test) and _labelled(dev):
        val, tst, how = dev, test, "val=dev_seen, test=test_seen"
    elif _labelled(dev_un) and _labelled(dev):
        # dev_unseen overlaps dev_seen on ~400 ids; drop overlap from val
        val = dev_un[~dev_un["id"].isin(dev["id"])].reset_index(drop=True)
        tst, how = dev, "val=dev_unseen (minus overlap), test=dev_seen"
    elif _labelled(dev):
        tst = dev
        train, val = group_split(train, frac=0.10, seed=seed, data_dir=data_dir)
        how = "val=10% group split of train, test=dev_seen"
    else:
        raise FileNotFoundError(
            f"Need at least train.jsonl and a labelled dev file in {data_dir}")

    # Drop memes whose image file is missing (some mirrors are incomplete) and report it,
    # so the numbers in the report always say exactly which memes were used.
    def keep_present(df, name):
        ok = df["img"].map(lambda p: (data_dir / p).exists())
        if (~ok).sum():
            print(f"[data] WARNING {name}: {(~ok).sum()} of {len(df)} memes have no image file -> dropped")
        return df[ok]

    train, val, tst = keep_present(train, "train"), keep_present(val, "val"), keep_present(tst, "test")
    out = {"train": train.reset_index(drop=True),
           "val": val.reset_index(drop=True),
           "test": tst.reset_index(drop=True)}
    for k, v in out.items():
        v["split"] = k
    print(f"[data] {how} | sizes: " + ", ".join(f"{k}={len(v)}" for k, v in out.items()))
    return out


# --------------------------------------------------------------------------
# 2. Perceptual hashing (dHash) from scratch
# --------------------------------------------------------------------------
def dhash(path: Path, hash_size: int = 8) -> int:
    """Difference hash: grey-scale, resize to (hash_size+1, hash_size),
    compare each pixel with its right neighbour -> hash_size**2 bits."""
    img = Image.open(path).convert("L").resize((hash_size + 1, hash_size), Image.BILINEAR)
    px = np.asarray(img, dtype=np.int16)
    bits = (px[:, 1:] > px[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def near_duplicate_groups(hashes: dict[str, int], max_dist: int, n_bits: int = 64) -> list[set]:
    """Group ids whose hashes differ by <= max_dist bits.

    Brute force is O(N^2) (~45M comparisons for 10k memes). We use the
    pigeonhole trick: split the 64-bit hash into (max_dist+1) chunks; two
    hashes within max_dist bits must agree exactly on at least one chunk.
    Only ids that share a chunk are compared -> near-linear in practice.
    """
    k = max_dist + 1
    chunk = int(np.ceil(n_bits / k))
    buckets = defaultdict(list)
    for i, h in hashes.items():
        for c in range(k):
            key = (c, (h >> (c * chunk)) & ((1 << chunk) - 1))
            buckets[key].append(i)

    parent = {i: i for i in hashes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    seen = set()
    for members in buckets.values():
        if len(members) < 2 or len(members) > 500:   # skip degenerate (blank) buckets
            continue
        for a_i in range(len(members)):
            for b_i in range(a_i + 1, len(members)):
                a, b = members[a_i], members[b_i]
                if (a, b) in seen:
                    continue
                seen.add((a, b))
                if hamming(hashes[a], hashes[b]) <= max_dist:
                    parent[find(a)] = find(b)

    groups = defaultdict(set)
    for i in hashes:
        groups[find(i)].add(i)
    return [g for g in groups.values() if len(g) > 1]


# --------------------------------------------------------------------------
# 3. Confounder-pair recovery
# --------------------------------------------------------------------------
def similar_caption_pairs(ids: list[str], texts: list[str], threshold: float = 0.90,
                          max_neighbours: int = 12) -> list[tuple[str, str]]:
    """All (id_a, id_b) whose captions have TF-IDF char 3-5-gram cosine >= threshold.
    Radius search on L2-normalised sparse vectors; captions with more than
    max_neighbours matches are generic ("me when", "")-style and are skipped."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors
    if len(ids) < 2:
        return []
    X = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1).fit_transform(texts)
    nn_ = NearestNeighbors(metric="cosine", algorithm="brute").fit(X)
    dist, ind = nn_.radius_neighbors(X, radius=1.0 - threshold)
    out = set()
    for i, (d, js) in enumerate(zip(dist, ind)):
        js = [j for j in js if j != i]
        if len(js) > max_neighbours:
            continue
        for j in js:
            a, b = sorted((ids[i], ids[j]))
            out.add((a, b))
    return sorted(out)


def normalise_text(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", str(t).lower()).strip()


def recover_confounders(df: pd.DataFrame, data_dir: Path, hash_size: int = 8,
                        max_dist: int = 4, cache: Path | None = None,
                        max_group: int = 12, caption_sim: float = 0.90) -> pd.DataFrame:
    """Return a DataFrame of pairs (id_a, id_b, kind) with kind in
    {"text_swap", "image_swap"}; pairs always have opposite labels.
    Images match by perceptual hash; captions match by TF-IDF cosine similarity."""
    if cache is not None and Path(cache).exists():
        return pd.read_csv(cache, dtype={"id_a": str, "id_b": str})

    lab = df.set_index("id")["label"].to_dict()
    txt = df.set_index("id")["text"].map(normalise_text).to_dict()

    # (a) text-swap: same image, different text
    print("[confounders] hashing images ...")
    hashes = {}
    for r in df.itertuples():
        try:
            hashes[r.id] = dhash(Path(data_dir) / r.img, hash_size)
        except Exception as e:                          # unreadable file -> skip
            print("  skip", r.img, e)
    pairs = []
    for g in near_duplicate_groups(hashes, max_dist, hash_size * hash_size):
        if len(g) > max_group:            # huge groups = near-blank images, not real confounders
            continue
        g = sorted(g)
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                a, b = g[i], g[j]
                if lab[a] != lab[b] and txt[a] != txt[b]:
                    pairs.append((a, b, "text_swap"))

    # (b) image-swap: similar caption, different image.
    #     Caption similarity = cosine of TF-IDF character 3-5-gram vectors, so small OCR
    #     differences ("its" vs "it's", an extra word) still match.
    ids_all = [i for i in df["id"] if len(txt[i]) > 0]
    for a, b in similar_caption_pairs(ids_all, [txt[i] for i in ids_all], caption_sim):
        if lab[a] != lab[b] and hashes.get(a) is not None and hashes.get(b) is not None \
                and hamming(hashes[a], hashes[b]) > max_dist:
            pairs.append((a, b, "image_swap"))

    out = pd.DataFrame(pairs, columns=["id_a", "id_b", "kind"]).drop_duplicates()
    if cache is not None:
        out.to_csv(cache, index=False)
    print(f"[confounders] {len(out)} pairs "
          f"({(out.kind=='text_swap').sum()} text-swap, {(out.kind=='image_swap').sum()} image-swap)")
    return out


# --------------------------------------------------------------------------
# 4. Group-aware validation split (keeps confounder pairs in one split)
# --------------------------------------------------------------------------
def group_split(df: pd.DataFrame, frac: float, seed: int, data_dir: Path):
    pairs = recover_confounders(df, data_dir)
    parent = {i: i for i in df["id"]}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in zip(pairs.id_a, pairs.id_b):
        parent[find(a)] = find(b)
    groups = df["id"].map(find)
    uniq = np.array(groups.unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_val = int(frac * len(df))
    chosen, count = set(), 0
    sizes = groups.value_counts()
    for g in uniq:
        if count >= n_val:
            break
        chosen.add(g)
        count += sizes[g]
    is_val = groups.isin(chosen)
    return df[~is_val].copy(), df[is_val].copy()


def recovered_category(df: pd.DataFrame, pairs: pd.DataFrame) -> pd.Series:
    """Tag each meme with the recovered sub-category used in the EDA."""
    cat = pd.Series("no_pair", index=df["id"].values)
    for kind in ["image_swap", "text_swap"]:          # text_swap wins ties
        ids = set(pairs.loc[pairs.kind == kind, "id_a"]) | set(pairs.loc[pairs.kind == kind, "id_b"])
        cat[cat.index.isin(ids)] = kind
    return cat.values


PROPOSAL_CATEGORIES = ["multimodal hate (paired)", "hate (unpaired)", "benign text confounder",
                       "benign image confounder", "random non-hateful"]


def proposal_category(df: pd.DataFrame, pairs: pd.DataFrame) -> np.ndarray:
    """Map recovered pairs onto the five sub-categories of Kiela et al. (2020).

    * hateful meme that is in a pair     -> multimodal hate: one swap flips it, so it needs both
    * hateful meme with no pair found    -> hate (unpaired): unimodal OR multimodal, cannot tell
    * benign member of a text-swap pair  -> benign text confounder (text replaced, image kept)
    * benign member of an image-swap pair-> benign image confounder (image replaced, text kept)
    * benign meme with no pair           -> random non-hateful
    The released data has no ground-truth sub-category tags, so this is an approximation.
    """
    ts = set(pairs.loc[pairs.kind == "text_swap", "id_a"]) | set(pairs.loc[pairs.kind == "text_swap", "id_b"])
    im = set(pairs.loc[pairs.kind == "image_swap", "id_a"]) | set(pairs.loc[pairs.kind == "image_swap", "id_b"])
    out = []
    for i, y in zip(df["id"], df["label"]):
        if y == 1:
            out.append(PROPOSAL_CATEGORIES[0] if (i in ts or i in im) else PROPOSAL_CATEGORIES[1])
        elif i in ts:
            out.append(PROPOSAL_CATEGORIES[2])
        elif i in im:
            out.append(PROPOSAL_CATEGORIES[3])
        else:
            out.append(PROPOSAL_CATEGORIES[4])
    return np.array(out)


def get_pairs(df: pd.DataFrame, split: str, cfg) -> pd.DataFrame:
    """Recover (and cache) confounder pairs for one split with the config's thresholds."""
    return recover_confounders(df, cfg.data_dir, cfg.hash_size, cfg.hash_max_hamming,
                               cache=cfg.out_dir / "tables" / f"confounder_pairs_{split}.csv",
                               caption_sim=cfg.caption_sim)
