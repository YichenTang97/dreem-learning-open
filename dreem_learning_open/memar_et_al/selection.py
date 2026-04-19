"""Kruskal–Wallis screen and mRMR feature selection (Memar et al. Sec. V)."""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
from scipy import stats
from sklearn.feature_selection import mutual_info_classif


def _mrmr_mi_greedy(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    k: int,
    random_state: int = 0,
) -> List[str]:
    """
    Peng-style mRMR approximation when ``mrmr-selection`` is unavailable:
    greedily maximize relevance (MI with y) minus mean absolute correlation
    with already-selected features.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    k = int(min(k, X.shape[1]))
    if k < 1:
        return []
    mi = mutual_info_classif(X, y, random_state=random_state)
    names = list(feature_names)
    remaining = list(range(X.shape[1]))
    selected: List[int] = []
    for _ in range(k):
        best_j = None
        best_score = -np.inf
        for j in remaining:
            rel = mi[j]
            if not selected:
                score = rel
            else:
                cors = []
                for s in selected:
                    c = np.corrcoef(X[:, j], X[:, s])[0, 1]
                    if np.isfinite(c):
                        cors.append(abs(c))
                red = float(np.mean(cors)) if cors else 0.0
                score = rel - red
            if score > best_score:
                best_score = score
                best_j = j
        if best_j is None:
            break
        selected.append(best_j)
        remaining.remove(best_j)
    return [names[i] for i in selected]


def kruskal_wallis_mask(
    X: np.ndarray,
    y: np.ndarray,
    p_threshold: float = 0.01,
) -> np.ndarray:
    """
    Keep features with Kruskal–Wallis p-value <= p_threshold (discard p > threshold).
    Returns boolean mask of shape (n_features,).
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    n_features = X.shape[1]
    mask = np.zeros(n_features, dtype=bool)
    classes = np.unique(y)
    for j in range(n_features):
        groups = [X[y == c, j] for c in classes]
        groups = [g for g in groups if g.size > 0]
        if len(groups) < 2:
            mask[j] = True
            continue
        try:
            _, p = stats.kruskal(*groups)
        except ValueError:
            mask[j] = True
            continue
        if np.isfinite(p) and p <= p_threshold:
            mask[j] = True
    return mask


def mrmr_select_features(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    k: int,
    random_state: int = 0,
) -> List[str]:
    """
    Max-relevance min-redundancy subset. Tries ``mrmr-selection`` (``mrmr.mrmr_classif``);
    falls back to a greedy MI–correlation heuristic if that import fails.
    """
    k = int(min(k, X.shape[1]))
    if k < 1:
        return []
    try:
        import pandas as pd
        from mrmr import mrmr_classif

        df = pd.DataFrame(X, columns=list(feature_names))
        selected = mrmr_classif(X=df, y=y, K=k)
        if isinstance(selected, (list, tuple)):
            return list(selected)
        return list(selected)
    except Exception:
        return _mrmr_mi_greedy(X, y, feature_names, k, random_state=random_state)
