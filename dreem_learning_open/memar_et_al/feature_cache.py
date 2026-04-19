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
    extract_memar_features_multichannel,
    load_bands_config,
    memar_multichannel_eeg_paths,
    precompute_band_sos_list,
    total_memar_feature_dim,
)
from dreem_learning_open.memar_et_al.io import (
    epoch_iterator,
    load_hypnogram,
)

CACHE_DIRNAME = "memar_features_cache"
MANIFEST_NAME = "cache_manifest.json"


def compute_feature_cache_key(
    memmap_description: dict,
    eeg_channel: str,
    bands_config: Dict[str, Any],
    all_eeg_channels: bool,
) -> str:
    """Short hash for cache directory name; invalidates when inputs change."""
    h = hashlib.sha1()
    h.update(json.dumps(memmap_description, sort_keys=True).encode("utf-8"))
    h.update(eeg_channel.encode("utf-8"))
    h.update(json.dumps(bands_config, sort_keys=True).encode("utf-8"))
    h.update(str(all_eeg_channels).encode("ascii"))
    if all_eeg_channels:
        paths = memar_multichannel_eeg_paths(memmap_description)
        h.update(json.dumps(paths, sort_keys=True).encode("utf-8"))
        h.update(str(total_memar_feature_dim(len(paths))).encode("ascii"))
    else:
        h.update(str(FEATURE_DIM).encode("ascii"))
    return h.hexdigest()[:24]


def feature_cache_path(save_folder: str, cache_key: str) -> str:
    return os.path.join(save_folder, CACHE_DIRNAME, cache_key)


def _subject_npz_path(cache_dir: str, record_path: str) -> str:
    bn = os.path.basename(os.path.normpath(record_path))
    return os.path.join(cache_dir, "{}.npz".format(bn))


def expected_feature_dim(memmap_description: dict, all_eeg_channels: bool) -> int:
    if not all_eeg_channels:
        return FEATURE_DIM
    return total_memar_feature_dim(len(memar_multichannel_eeg_paths(memmap_description)))


def _npz_valid(npz_path: str, record_path: str, expected_dim: int) -> bool:
    if not os.path.isfile(npz_path):
        return False
    try:
        hyp = load_hypnogram(record_path)
        n = int(hyp.shape[0])
        z = np.load(npz_path)
        X = z["X"]
        y = z["y"]
        if tuple(X.shape) != (n, expected_dim) or tuple(y.shape) != (n,):
            return False
        if "feature_dim" in z.files and int(z["feature_dim"]) != expected_dim:
            return False
    except Exception:
        return False
    return True


def _extract_epoch_chunk_ordered(
    record_path: str,
    epoch_indices: np.ndarray,
    memmap_description: dict,
    all_eeg_channels: bool,
    channel_signal: str | None,
    config_path: str | None,
    bands: dict,
    fs: float,
    band_sos_list: list,
) -> Tuple[np.ndarray, np.ndarray]:
    want_set = {int(i) for i in np.asarray(epoch_indices).ravel()}
    idx_to_vec: dict = {}
    idx_to_lab: dict = {}
    for i, ep, lab in epoch_iterator(
        record_path,
        memmap_description,
        channel_signal=channel_signal,
        config_path=config_path,
        all_eeg_channels=all_eeg_channels,
    ):
        if i not in want_set:
            continue
        idx_to_vec[i] = extract_memar_features_multichannel(ep, fs, bands, band_sos_list=band_sos_list)
        idx_to_lab[i] = lab
        if len(idx_to_vec) == len(want_set):
            break
    if len(idx_to_vec) != len(want_set):
        raise RuntimeError("incomplete epochs for {}".format(record_path))
    X = np.vstack([idx_to_vec[int(i)] for i in epoch_indices])
    y = np.asarray([idx_to_lab[int(i)] for i in epoch_indices], dtype=np.int64)
    return X, y


def _extract_all_epochs_one_record(
    record_path: str,
    memmap_description: dict,
    all_eeg_channels: bool,
    channel_signal: str | None,
    config_path: str | None,
    bands: dict,
    fs: float,
    band_sos_list: list,
    epoch_inner_n_jobs: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """All epochs in order 0 .. n-1."""
    hyp = load_hypnogram(record_path)
    n_epochs = int(hyp.shape[0])
    indices = np.arange(n_epochs, dtype=np.intp)
    inner = max(1, int(epoch_inner_n_jobs))
    if inner <= 1 or n_epochs <= 1:
        return _extract_epoch_chunk_ordered(
            record_path,
            indices,
            memmap_description,
            all_eeg_channels,
            channel_signal,
            config_path,
            bands,
            fs,
            band_sos_list,
        )

    from joblib import Parallel, delayed

    chunks = np.array_split(indices, min(inner, n_epochs))
    delayed_calls = [
        delayed(_extract_epoch_chunk_ordered)(
            record_path,
            ch,
            memmap_description,
            all_eeg_channels,
            channel_signal,
            config_path,
            bands,
            fs,
            band_sos_list,
        )
        for ch in chunks
        if ch.size > 0
    ]
    parts = Parallel(n_jobs=inner, verbose=0)(delayed_calls)
    X = np.vstack([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    return X, y


def _write_manifest(
    cache_dir: str,
    cache_key: str,
    memmap_hash: str,
    eeg_channel: str,
    bands_config: Dict[str, Any],
    n_records: int,
    all_eeg_channels: bool,
    feature_dim: int,
    memmap_description: dict,
) -> None:
    payload: Dict[str, Any] = {
        "cache_key": cache_key,
        "memmap_hash": memmap_hash,
        "eeg_channel": eeg_channel,
        "all_eeg_channels": all_eeg_channels,
        "feature_dim": feature_dim,
        "bands_fs": float(bands_config["fs"]),
        "bands_sha1": hashlib.sha1(
            json.dumps(bands_config, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "n_subjects": n_records,
    }
    if all_eeg_channels:
        payload["eeg_signal_paths"] = memar_multichannel_eeg_paths(memmap_description)
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
    epoch_inner_n_jobs: int,
    all_eeg_channels: bool,
    show_progress: bool,
    quiet: bool,
) -> None:
    """
    Ensure ``cache_dir`` contains one ``<subject_basename>.npz`` per record with keys
    ``X`` (n_epochs, feature_dim), ``y`` (n_epochs,) hypnogram labels, ``feature_dim``.
    """
    log = logging.getLogger("memar_et_al")
    os.makedirs(cache_dir, exist_ok=True)
    bands = load_bands_config()
    fs = float(bands["fs"])
    band_sos_list = precompute_band_sos_list(fs, bands)
    exp_dim = expected_feature_dim(memmap_description, all_eeg_channels)

    missing: List[str] = []
    for rec in record_paths:
        p = _subject_npz_path(cache_dir, rec)
        if not _npz_valid(p, rec, exp_dim):
            missing.append(rec)

    _write_manifest(
        cache_dir,
        cache_key,
        memmap_hash,
        eeg_channel,
        bands,
        len(record_paths),
        all_eeg_channels,
        exp_dim,
        memmap_description,
    )

    if not missing:
        if not quiet:
            log.info(
                "Feature cache | up to date | %d subjects at %s (--feat-workers unused; nothing to extract)",
                len(record_paths),
                cache_dir,
            )
        return

    n_missing = len(missing)
    inner_eff = max(1, int(epoch_inner_n_jobs))
    effective_parallel = min(max(1, int(feat_n_jobs)), n_missing)
    if not quiet:
        log.info(
            "Feature cache | extracting %d / %d subjects (missing or stale) | %s",
            n_missing,
            len(record_paths),
            cache_dir,
        )
        if feat_n_jobs != 1:
            log.info(
                "Feature cache | --feat-workers=%d → up to %d concurrent subjects (joblib processes)",
                feat_n_jobs,
                effective_parallel,
            )
        if inner_eff > 1:
            log.info(
                "Feature cache | --epoch-workers=%d → nested per-subject epoch chunks (joblib)",
                inner_eff,
            )

    def _run_one(rec: str) -> Tuple[np.ndarray, np.ndarray]:
        return _extract_all_epochs_one_record(
            rec,
            memmap_description,
            all_eeg_channels,
            eeg_channel if not all_eeg_channels else None,
            memar_config_path,
            bands,
            fs,
            band_sos_list,
            inner_eff,
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
            X, y = _run_one(rec)
            out = _subject_npz_path(cache_dir, rec)
            np.savez_compressed(out, X=X, y=y, feature_dim=exp_dim)
    else:
        from joblib import Parallel, delayed

        delayed_calls = [delayed(_run_one)(rec) for rec in missing]
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
            np.savez_compressed(out, X=X, y=y, feature_dim=exp_dim)

    if not quiet:
        log.info("Feature cache | done | %d subject files written under %s", len(missing), cache_dir)


def stack_labeled_training_from_cache(
    train_records: List[str],
    cache_dir: str,
    show_progress: bool,
    progress_desc: str,
    feature_dim: int,
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
        if X.shape[1] != feature_dim:
            raise ValueError(
                "cached feature_dim mismatch: expected {}, got {} for {}".format(
                    feature_dim, X.shape[1], rec
                )
            )
        m = y >= 0
        xs.append(X[m])
        ys.append(y[m])
    if not xs:
        return np.zeros((0, feature_dim), dtype=np.float64), np.zeros((0,), dtype=np.int64)
    return np.vstack(xs), np.concatenate(ys)


def load_cached_subject_matrix(cache_dir: str, record_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Full epoch matrix and hypnogram vector for one subject."""
    z = np.load(_subject_npz_path(cache_dir, record_path))
    return z["X"], z["y"]
