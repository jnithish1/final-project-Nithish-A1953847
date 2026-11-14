"""
Statistical tests used in Criterion 6, implemented from scratch with NumPy/SciPy.

  delong_test        DeLong et al. (1988) test for two correlated AUROCs,
                     fast O(n log n) version of Sun & Xu (2014)
  paired_bootstrap   bootstrap CI and p-value for the AUROC difference
  mcnemar_exact      exact (binomial) McNemar test on per-meme correctness
  seed_tests         paired t-test and Wilcoxon signed-rank across seeds
  holm               Holm-Bonferroni correction for the 7 comparisons
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def auroc(y: np.ndarray, p: np.ndarray) -> float:
    """Rank-based AUROC (Mann-Whitney U / (n_pos * n_neg)), ties get mid-ranks."""
    r = stats.rankdata(p)
    n1 = int((y == 1).sum()); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    z = x[order]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = t
    return out


def delong_test(y: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> dict:
    pos, neg = y == 1, y == 0
    m, n = pos.sum(), neg.sum()
    aucs, v10, v01 = [], [], []
    for p in (p1, p2):
        x, yv = p[pos], p[neg]
        tx, ty, tz = _midrank(x), _midrank(yv), _midrank(np.concatenate([x, yv]))
        aucs.append(tz[:m].sum() / (m * n) - (m + 1) / (2 * n))
        v10.append((tz[:m] - tx) / n)          # structural components for positives
        v01.append(1 - (tz[m:] - ty) / m)      # ... and negatives
    s10, s01 = np.cov(np.vstack(v10)), np.cov(np.vstack(v01))
    S = s10 / m + s01 / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    diff = aucs[0] - aucs[1]
    z = diff / np.sqrt(var) if var > 0 else 0.0
    return {"auc1": float(aucs[0]), "auc2": float(aucs[1]), "diff": float(diff),
            "z": float(z), "p": float(2 * stats.norm.sf(abs(z)))}


def paired_bootstrap(y, p1, p2, n_boot: int = 2000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = []
    while len(diffs) < n_boot:
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():
            continue                                   # need both classes
        diffs.append(auroc(y[idx], p1[idx]) - auroc(y[idx], p2[idx]))
    diffs = np.array(diffs)
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"diff": float(auroc(y, p1) - auroc(y, p2)),
            "ci_low": float(np.percentile(diffs, 2.5)), "ci_high": float(np.percentile(diffs, 97.5)),
            "p": float(min(1.0, max(p, 1.0 / n_boot))), "samples": diffs}


def mcnemar_exact(y, pred1, pred2) -> dict:
    c1, c2 = pred1 == y, pred2 == y
    b = int((c1 & ~c2).sum())       # model 1 right, model 2 wrong
    c = int((~c1 & c2).sum())
    p = stats.binomtest(min(b, c), b + c, 0.5).pvalue if b + c > 0 else 1.0
    return {"b_1right_2wrong": b, "c_1wrong_2right": c, "p": float(p)}


def seed_tests(a: np.ndarray, b: np.ndarray) -> dict:
    """a, b: per-seed metric values of two models trained with the same seeds."""
    t = stats.ttest_rel(a, b)
    try:
        w = stats.wilcoxon(a, b).pvalue
    except ValueError:                  # all differences zero
        w = 1.0
    return {"mean_diff": float(np.mean(a - b)), "t": float(t.statistic), "p_ttest": float(t.pvalue),
            "p_wilcoxon": float(w)}


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def t_ci(x: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    x = np.asarray(x, float)
    if len(x) < 2:
        return float(x.mean()), float(x.mean())
    h = stats.t.ppf(1 - alpha / 2, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() - h), float(x.mean() + h)
