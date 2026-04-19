"""KW → mRMR → RandomForest (Memar et al.); shared by LOSO and internal CV."""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from dreem_learning_open.memar_et_al.selection import kruskal_wallis_mask, mrmr_select_features


def fit_memar_rf_pipeline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    all_names: Sequence[str],
    mrmr_k: int,
    n_estimators: int,
    kw_p: float,
    random_state: int,
    rf_n_jobs: int,
) -> Tuple[RandomForestClassifier, List[str], List[int]]:
    """
    Kruskal–Wallis screen, mRMR, then ``RandomForestClassifier`` on selected columns.

    Returns ``(rf, picked_feature_names, column_indices_into_X)``.
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.int64)
    all_names = list(all_names)

    kw_mask = kruskal_wallis_mask(X_train, y_train, p_threshold=kw_p)
    if not np.any(kw_mask):
        kw_mask[:] = True

    name_arr = np.array(all_names)
    names_kw = name_arr[kw_mask].tolist()
    X_kw = X_train[:, kw_mask]

    k_sub = min(int(mrmr_k), X_kw.shape[1])
    try:
        picked = mrmr_select_features(
            X_kw, y_train, names_kw, k_sub, random_state=random_state
        )
    except Exception:
        picked = names_kw[:k_sub]

    if not picked:
        picked = names_kw[: min(10, len(names_kw))]

    name_to_idx = {n: i for i, n in enumerate(all_names)}
    cols = [name_to_idx[n] for n in picked]
    X_sel = X_train[:, cols]

    max_f = max(1, int(np.floor(np.sqrt(X_sel.shape[1]))))
    rf = RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_features=max_f,
        random_state=random_state,
        n_jobs=rf_n_jobs,
        class_weight="balanced_subsample",
    )
    rf.fit(X_sel, y_train)
    return rf, picked, cols


def predict_memar_rf(
    X: np.ndarray,
    rf: RandomForestClassifier,
    cols: Sequence[int],
) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    return rf.predict(X[:, cols]).astype(np.int64)
