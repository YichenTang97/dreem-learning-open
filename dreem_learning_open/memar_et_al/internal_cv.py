"""
Subject-wise internal k-fold CV to choose mRMR size and RF ``n_estimators`` (Memar & Faradji–style).

Outer evaluation remains LOSO on held-out test subjects. On the remaining training subjects,
we split by **subject** (not by epoch) into k folds; each inner fold retrains KW + mRMR + RF on
inner-train subjects only and scores on inner-validation subjects to avoid leakage.
"""
from __future__ import annotations

import itertools
import logging
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

from dreem_learning_open.memar_et_al.feature_cache import stack_labeled_training_from_cache
from dreem_learning_open.memar_et_al.pipeline import fit_memar_rf_pipeline, predict_memar_rf


def _macro_f1_scored_epochs(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    m = y_true >= 0
    if not np.any(m):
        return float("nan")
    return float(
        f1_score(y_true[m], y_pred[m], average="macro", zero_division=0)
    )


def select_mrmr_n_estimators_subject_cv(
    train_records: List[str],
    feature_cache_dir: str,
    feature_dim: int,
    all_names: Sequence[str],
    mrmr_k_candidates: Sequence[int],
    n_estimators_candidates: Sequence[int],
    kw_p: float,
    n_splits: int,
    random_state: int,
    rf_n_jobs: int,
    show_progress: bool,
    fold_label: str,
    quiet: bool,
) -> Tuple[int, int, Dict[str, Any]]:
    """
    Grid search over (mrmr_k, n_estimators) with subject-wise K-fold on ``train_records``.

    Returns ``(best_mrmr_k, best_n_estimators, details)``.
    """
    log = logging.getLogger("memar_et_al")
    n_subj = len(train_records)
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if n_subj < n_splits:
        raise ValueError(
            "internal CV needs at least as many training subjects as folds: "
            "got {} train subjects and n_splits={}".format(n_subj, n_splits)
        )

    mk_grid = sorted({int(x) for x in mrmr_k_candidates})
    ne_grid = sorted({int(x) for x in n_estimators_candidates})
    if not mk_grid or not ne_grid:
        raise ValueError("mrmr_k and n_estimators grids must be non-empty")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    subject_idx = np.arange(n_subj)
    fold_splits = list(kf.split(subject_idx))

    grid_scores: Dict[Tuple[int, int], List[float]] = {}

    grid = list(itertools.product(mk_grid, ne_grid))
    for mk, ne in grid:
        scores_fold: List[float] = []
        for inner_train_i, inner_val_i in fold_splits:
            tr_paths = [train_records[j] for j in inner_train_i]
            va_paths = [train_records[j] for j in inner_val_i]
            X_tr, y_tr = stack_labeled_training_from_cache(
                tr_paths,
                feature_cache_dir,
                show_progress=False,
                progress_desc="{} inner train".format(fold_label),
                feature_dim=feature_dim,
            )
            X_va, y_va = stack_labeled_training_from_cache(
                va_paths,
                feature_cache_dir,
                show_progress=False,
                progress_desc="{} inner val".format(fold_label),
                feature_dim=feature_dim,
            )
            if X_tr.shape[0] == 0 or X_va.shape[0] == 0:
                scores_fold.append(0.0)
                continue
            rs = random_state + int(inner_train_i[0]) * 10007 + mk * 131 + ne
            rf, _picked, cols = fit_memar_rf_pipeline(
                X_tr,
                y_tr,
                all_names,
                mrmr_k=mk,
                n_estimators=ne,
                kw_p=kw_p,
                random_state=rs,
                rf_n_jobs=rf_n_jobs,
            )
            pred = predict_memar_rf(X_va, rf, cols)
            scores_fold.append(_macro_f1_scored_epochs(y_va, pred))

        grid_scores[(mk, ne)] = scores_fold
        if not quiet and show_progress:
            log.info(
                "%s | internal CV | mrmr_k=%d n_estimators=%d | mean macro-F1=%.4f (%d folds)",
                fold_label,
                mk,
                ne,
                float(np.nanmean(scores_fold)),
                len(fold_splits),
            )

    row_means = [
        (float(np.nanmean(sc)), mk, ne) for (mk, ne), sc in grid_scores.items()
    ]
    finite_means = [m for m, _, _ in row_means if np.isfinite(m)]
    if not finite_means:
        raise RuntimeError("internal CV produced no finite mean scores (check training data)")
    best_m_val = max(finite_means)
    shortlist = [(mk, ne) for m, mk, ne in row_means if np.isfinite(m) and abs(m - best_m_val) < 1e-9]
    shortlist.sort(key=lambda p: (p[1], p[0]))
    best_mk, best_ne = shortlist[0]
    best_mean = best_m_val

    details: Dict[str, Any] = {
        "n_splits": n_splits,
        "mrmr_k_candidates": mk_grid,
        "n_estimators_candidates": ne_grid,
        "mean_macro_f1_per_grid_point": {
            "{},{}".format(mk, ne): float(np.nanmean(sc))
            for (mk, ne), sc in grid_scores.items()
        },
        "per_fold_macro_f1": {
            "{},{}".format(mk, ne): [float(s) for s in sc]
            for (mk, ne), sc in grid_scores.items()
        },
        "selected_mrmr_k": best_mk,
        "selected_n_estimators": best_ne,
        "best_mean_macro_f1": best_mean,
    }
    return best_mk, best_ne, details
