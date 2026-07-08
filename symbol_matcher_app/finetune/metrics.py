"""Dependency-free classification metrics (no sklearn needed in the vsam env)."""

from __future__ import annotations

import numpy as np


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC-AUC via the rank-sum (Mann-Whitney U) identity, with tie handling."""
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_score, kind="mergesort")
    s = y_score[order]
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = 0.5 * (i + j) + 1.0  # average rank (1-based) for ties
        i = j + 1
    pos_rank_sum = ranks[y_true[order] == 1].sum()
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def prf_at(y_true: np.ndarray, y_score: np.ndarray, thr: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(np.int64)
    pred = (np.asarray(y_score) >= thr).astype(np.int64)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn}


def best_f1(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """Sweep candidate thresholds (the unique scores) and return the P/R/F1 and
    threshold that maximize F1. Useful for an uncalibrated ranker whose optimal
    operating point is not necessarily 0.5."""
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_score.size == 0:
        return {"thr": 0.5, **prf_at(y_true, y_score, 0.5)}
    best = {"thr": 0.5, "f1": -1.0}
    for thr in np.unique(y_score):
        m = prf_at(y_true, y_score, float(thr))
        if m["f1"] > best["f1"]:
            best = {"thr": round(float(thr), 4), **m}
    return best


def calibration(y_true: np.ndarray, y_score: np.ndarray, bins: int = 10) -> list[dict]:
    y_true = np.asarray(y_true).astype(np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    out = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (y_score >= lo) & (y_score < hi if hi < 1.0 else y_score <= hi)
        if m.sum() == 0:
            continue
        out.append({"bin": f"[{lo:.1f},{hi:.1f})", "n": int(m.sum()),
                    "mean_score": round(float(y_score[m].mean()), 3),
                    "frac_pos": round(float(y_true[m].mean()), 3)})
    return out
