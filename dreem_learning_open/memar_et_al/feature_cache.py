"""
On-disk cache of full Memar feature matrices (all epochs) per subject.

Stored under ``<experiment_save_folder>/memar_features_cache/<cache_key>/`` so LOSO folds
reuse extraction instead of recomputing features for every train/test split.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any, Dict, List, Tuple

import numpy as np

from dreem_learning_open.memar_et_al.features import (
    FEATURE_DIM,
    extract_memar_features_vector,
    load_bands_config,
    precompute_band_sos_list,
)
from dreem_learning_open.memar_et_al.io import epoch_iterator, load_hypnogram

CACHE_DIRNAME = "memar_features_cache"
MANIFEST_NAME = "cache_manifest.json"


def compute_feature_cache_key(
    memmap_description: dict,
    eeg_channel: str,
    bands_config: Dict[str, Any],
) -> str:
    """Short hash for cache directory name; invalidates when inputs change."""
    h = hashlib.sha1()
    h.update(json.dumps(memmap_description, sort_keys=True).encode("utf-8"))
    h.update(eeg_channel.encode("utf-8"))
    h.update(json.dumps(bands_config, sort_keys=True).encode("utf-8"))
    h.update(str(FEATURE_DIM).encode("ascii"))
    return h.hexdigest()[:24]


def feature_cache_path(save_folder: str, cache_key: str) -> str:
    return os.path.join(save_folder, CACHE_DIRNAME, cache_key)


def _subject_npz_path(cache_dir: str, record_path: str) -> str:
    bn = os.path.basename(os.path.normpath(record_path))
    return os.path.join(cache_dir, "{}.npz".format(bn))


def _npz_valid(npz_path: str, record_path: str) -> bool:
    if not os.path.isfile(npz_path):
        return False
    try:
        hyp = load_hypnogram(record_path)
        n = int(hyp.shape[0])
        z = np.load(npz_path)
        X = z["X"]
        y = z["y"]
        if tuple(X.shape) != (n, FEATURE_DIM) or tuple(y.shape) != (n,):
            return False
        if "feature_dim" in z.files and int(z["feature_dim"]) != FEATURE_DIM:
            return False
    except Exception:
        return False
    return True


def _extract_all_epochs_one_record(
    record_path: str,
    memmap_description: dict,
    channel_signal: str | None,
    config_path: str | None,
    bands: dict,
    fs: float,
    band_sos_list: list,
) -> Tuple[np.ndarray, np.ndarray]:
    """All epochs in fixed order (same as :func:`epoch_iterator`)."""
    rows: list = []
    labs: list = []
    for _i, ep, lab in epoch_iterator(
        record_path, memmap_description, channel_signal=channel_signal, config_path=config_path
    ):
        rows.append(extract_memar_features_vector(ep, fs, bands, band_sos_list=band_sos_list))
        labs.append(lab)
    if not rows:
        return np.zeros((0, FEATURE_DIM), dtype=np.float64), np.zeros((0,), dtype=np.int64)
    return np.vstack(rows), np.asarray(labs, dtype=np.int64)


def _write_manifest(
    cache_dir: str,
    cache_key: str,
    memmap_hash: str,
    eeg_channel: str,
    bands_config: Dict[str, Any],
    n_records: int,
) -> None:
    payload = {
        "cache_key": cache_key,
        "memmap_hash": memmap_hash,
        "eeg_channel": eeg_channel,
        "feature_dim": FEATURE_DIM,
        "bands_fs": float(bands_config["fs"]),
        "bands_sha1": hashlib.sha1(
            json.dumps(bands_config, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "n_subjects": n_records,
    }
    with open(os.path.join(cache_dir, MANIFEST_NAME), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def ensure_memar_feature_cache(
    cache_dir: str,
    cache_key: str,
    memmap_hash: str,
    record_paths: List[str],
    memmap_description: dict,
    eeg_channel: str,
    memar_config_path: str | None,
    feat_n_jobs: int,
    show_progress: bool,
    quiet: bool,
) -> None:
    """
    Ensure ``cache_dir`` contains one ``<subject_basename>.npz`` per record with keys
    ``X`` (n_epochs, FEATURE_DIM), ``y`` (n_epochs,) hypnogram labels, ``feature_dim``.
    Extracts only missing or invalid files.
    """
    log = logging.getLogger("memar_et_al")
    os.makedirs(cache_dir, exist_ok=True)
    bands = load_bands_config()
    fs = float(bands["fs"])
    band_sos_list = precompute_band_sos_list(fs, bands)

    missing: List[str] = []
    for rec in record_paths:
        p = _subject_npz_path(cache_dir, rec)
        if not _npz_valid(p, rec):
            missing.append(rec)

    _write_manifest(cache_dir, cache_key, memmap_hash, eeg_channel, bands, len(record_paths))

    if not missing:
        if not quiet:
            log.info(
                "Feature cache | up to date | %d subjects at %s",
                len(record_paths),
                cache_dir,
            )
        return

    if not quiet:
        log.info(
            "Feature cache | extracting %d / %d subjects (missing or stale) | %s",
            len(missing),
            len(record_paths),
            cache_dir,
        )

    if feat_n_jobs == 1 or len(missing) == 1:
        rec_iter: Any = missing
        if show_progress:
            try:
                from tqdm import tqdm

                rec_iter = tqdm(missing, desc="Feature cache extract", unit="subj")
            except ImportError:
                pass
        for rec in rec_iter:
            X, y = _extract_all_epochs_one_record(
                rec,
                memmap_description,
                eeg_channel,
                memar_config_path,
                bands,
                fs,
                band_sos_list,
            )
            out = _subject_npz_path(cache_dir, rec)
            np.savez_compressed(out, X=X, y=y, feature_dim=FEATURE_DIM)
    else:
        from joblib import Parallel, delayed

        delayed_calls = [
            delayed(_extract_all_epochs_one_record)(
                rec,
                memmap_description,
                eeg_channel,
                memar_config_path,
                bands,
                fs,
                band_sos_list,
            )
            for rec in missing
        ]
        if show_progress:
            try:
                from tqdm import tqdm
                from tqdm.contrib.joblib import tqdm_joblib

                with tqdm_joblib(
                    tqdm(
                        total=len(missing),
                        desc="Feature cache extract",
                        unit="subj",
                        leave=True,
                    )
                ):
                    parts = Parallel(n_jobs=feat_n_jobs, verbose=0)(delayed_calls)
            except ImportError:
                parts = Parallel(n_jobs=feat_n_jobs, verbose=10)(delayed_calls)
        else:
            parts = Parallel(n_jobs=feat_n_jobs, verbose=0)(delayed_calls)

        for rec, (X, y) in zip(missing, parts):
            out = _subject_npz_path(cache_dir, rec)
            np.savez_compressed(out, X=X, y=y, feature_dim=FEATURE_DIM)

    if not quiet:
        log.info("Feature cache | done | %d subject files written under %s", len(missing), cache_dir)


def stack_labeled_training_from_cache(
    train_records: List[str],
    cache_dir: str,
    show_progress: bool,
    progress_desc: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Concatenate labeled rows (hypnogram >= 0) from cached per-subject matrices."""
    xs: list = []
    ys: list = []
    rec_iter: Any = train_records
    if show_progress:
        try:
            from tqdm import tqdm

            rec_iter = tqdm(train_records, desc=progress_desc, unit="subj")
        except ImportError:
            pass
    for rec in rec_iter:
        z = np.load(_subject_npz_path(cache_dir, rec))
        X = z["X"]
        y = z["y"]
        m = y >= 0
        xs.append(X[m])
        ys.append(y[m])
    if not xs:
        return np.zeros((0, FEATURE_DIM), dtype=np.float64), np.zeros((0,), dtype=np.int64)
    return np.vstack(xs), np.concatenate(ys)


def load_cached_subject_matrix(cache_dir: str, record_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Full epoch matrix and hypnogram vector for one subject."""
    z = np.load(_subject_npz_path(cache_dir, record_path))
    return z["X"], z["y"]
