"""
Memar & Faradji (2018) classical RF baseline (`memar_et_al`): 104 features, KW + mRMR + RF, LOSO.

Matches **subject cross-validation** (paper Sec. VIII-B): one subject held out for testing;
**all other subjects** are used for Kruskal–Wallis + mRMR + RF training (no held-out
validation split). Default ``--mrmr-k 40`` and ``--n-estimators 100`` follow the paper.

Run from repository root with the package on PYTHONPATH or installed::

    python scripts/run_memar_et_al.py --memmap-only
    python scripts/run_memar_et_al.py --folds 0 --no-force --skip-memmap-build

Parallel folds: use ``--workers N`` to run multiple LOSO folds in parallel processes, or the
same manual pattern as ``run_simple_sleep_net_only.py`` (one shell process per ``--folds``).
Unless ``--quiet``, each fold still prints step logs and tqdm (stdout may interleave).

Features are computed **once per subject** and stored under
``<save_folder>/memar_features_cache/<key>/`` (compressed ``.npz``). Each LOSO fold loads from
this cache for training and test. Use ``--refresh-feature-cache`` to force re-extraction.
Parallel extraction uses ``--feat-workers`` (joblib). Kraskov uses ``scipy.spatial.cKDTree``;
Butterworth SOS are precomputed once per run.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import json
import logging
import os
import random as rd
import sys
import shutil
import tarfile
import tempfile
import time
import uuid

import git
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

from dreem_learning_open.memar_et_al.config import default_memar_et_al_config_path, get_eeg_signal
from dreem_learning_open.memar_et_al.feature_cache import (
    ensure_memar_feature_cache,
    feature_cache_path,
    compute_feature_cache_key,
    load_cached_subject_matrix,
    stack_labeled_training_from_cache,
)
from dreem_learning_open.memar_et_al.features import build_feature_names, load_bands_config
from dreem_learning_open.memar_et_al.selection import kruskal_wallis_mask, mrmr_select_features
from dreem_learning_open.preprocessings.h5_to_memmap import h5_to_memmaps
from dreem_learning_open.settings import DODH_SETTINGS, EXPERIMENTS_DIRECTORY
from dreem_learning_open.utils.memmap_eeg import filter_memmap_signals_eeg_only


def memmap_hash(memmap_description: dict) -> str:
    return hashlib.sha1(json.dumps(memmap_description).encode()).hexdigest()[:10]


def memmap_directory_ready(memmaps_dir: str) -> bool:
    return os.path.isfile(os.path.join(memmaps_dir, "groups_description.json")) and os.path.isfile(
        os.path.join(memmaps_dir, "features_description.json")
    )


def _to_rel(p: str) -> str:
    try:
        return os.path.relpath(p, os.getcwd()).replace("\\", "/")
    except Exception:
        return p.replace("\\", "/")


def _memar_logger(quiet: bool) -> logging.Logger:
    """Per-process logger (safe for ``ProcessPoolExecutor`` workers)."""
    log = logging.getLogger("memar_et_al")
    level = logging.WARNING if quiet else logging.INFO
    log.setLevel(level)
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setLevel(logging.DEBUG)
        h.setFormatter(
            logging.Formatter("[%(asctime)s] [memar_et_al] %(message)s", datefmt="%H:%M:%S")
        )
        log.addHandler(h)
        log.propagate = False
    else:
        log.setLevel(level)
    return log


def save_sklearn_bundle_tar(
    out_path: str,
    bundle: dict,
) -> None:
    """``best_model.gz``: uncompressed tar (same convention as ModuloNet exports)."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".joblib") as tmp:
        joblib.dump(bundle, tmp.name, compress=3)
        tpath = tmp.name
    try:
        with tarfile.open(out_path, "w") as tar:
            tar.add(tpath, arcname="memar_rf.joblib")
    finally:
        os.unlink(tpath)


def run_fold(
    memmap_description: dict,
    dataset_setting: dict,
    train_records: list,
    test_records: list,
    save_folder: str,
    mrmr_k: int,
    n_estimators: int,
    kw_p: float,
    random_state: int,
    eeg_channel: str,
    memar_config_path: str | None,
    rf_n_jobs: int = -1,
    fold_idx: int | None = None,
    show_progress: bool = True,
    quiet: bool = False,
    feature_cache_dir: str = "",
) -> str:
    log = _memar_logger(quiet)
    fold_label = "fold {}".format(fold_idx) if fold_idx is not None else "fold"
    test_record = test_records[0]
    test_basename_preview = os.path.basename(os.path.normpath(test_record))

    t_fold = time.time()
    if not feature_cache_dir:
        raise ValueError("feature_cache_dir is required (populate feature cache before run_fold).")
    log.info(
        "%s | start | test_subject=%s | train_subjects=%d | eeg=%s",
        fold_label,
        test_basename_preview,
        len(train_records),
        eeg_channel,
    )

    dataset_dir = os.path.join(dataset_setting["memmap_directory"], memmap_hash(memmap_description))
    log.info("%s | step 1/6 | reading memmap index (groups_description, features_description)", fold_label)
    groups_description = json.load(open(os.path.join(dataset_dir, "groups_description.json")))
    features_description = json.load(
        open(os.path.join(dataset_dir, "features_description.json"))
    )

    bands_cfg = load_bands_config()
    all_names = build_feature_names(bands_cfg)

    log.info(
        "%s | step 2/6 | assembling training matrix from feature cache (%d subjects, labeled epochs)",
        fold_label,
        len(train_records),
    )
    t0 = time.time()
    X_train, y_train = stack_labeled_training_from_cache(
        train_records,
        feature_cache_dir,
        show_progress=show_progress,
        progress_desc="{} train feats".format(fold_label),
    )
    if X_train.shape[0] == 0:
        raise RuntimeError("No labeled training epochs for this fold.")
    log.info(
        "%s | step 2/6 | done | X.shape=%s | elapsed %.1fs",
        fold_label,
        X_train.shape,
        time.time() - t0,
    )

    log.info("%s | step 3/6 | Kruskal–Wallis (keep p <= %g)", fold_label, kw_p)
    t0 = time.time()
    kw_mask = kruskal_wallis_mask(X_train, y_train, p_threshold=kw_p)
    if not np.any(kw_mask):
        log.warning("%s | KW kept no features; using all %d features", fold_label, len(all_names))
        kw_mask[:] = True
    n_kw = int(np.sum(kw_mask))
    log.info(
        "%s | step 3/6 | done | kept %d/%d features | %.1fs",
        fold_label,
        n_kw,
        len(all_names),
        time.time() - t0,
    )

    name_arr = np.array(all_names)
    names_kw = name_arr[kw_mask].tolist()
    X_kw = X_train[:, kw_mask]

    k_sub = min(mrmr_k, X_kw.shape[1])
    log.info("%s | step 4/6 | mRMR: select up to %d from %d (after KW)", fold_label, k_sub, X_kw.shape[1])
    t0 = time.time()
    try:
        picked = mrmr_select_features(
            X_kw, y_train, names_kw, k_sub, random_state=random_state
        )
    except Exception as exc:
        log.warning("%s | mRMR failed (%s); falling back to first %d names", fold_label, exc, k_sub)
        picked = names_kw[:k_sub]

    if not picked:
        picked = names_kw[: min(10, len(names_kw))]
    log.info(
        "%s | step 4/6 | done | selected %d features | %.1fs | e.g. %s",
        fold_label,
        len(picked),
        time.time() - t0,
        ", ".join(picked[: min(5, len(picked))]),
    )

    name_to_idx = {n: i for i, n in enumerate(all_names)}
    cols = [name_to_idx[n] for n in picked]
    X_sel = X_train[:, cols]

    max_f = max(1, int(np.floor(np.sqrt(X_sel.shape[1]))))
    log.info(
        "%s | step 5/6 | RandomForest fit | trees=%d max_features=%d n_jobs=%s | samples=%d",
        fold_label,
        n_estimators,
        max_f,
        rf_n_jobs,
        X_sel.shape[0],
    )
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_features=max_f,
        random_state=random_state,
        n_jobs=rf_n_jobs,
        class_weight="balanced_subsample",
    )
    rf.fit(X_sel, y_train)
    log.info("%s | step 5/6 | done | RF fit %.1fs", fold_label, time.time() - t0)

    X_test, hyp_true = load_cached_subject_matrix(feature_cache_dir, test_record)
    n_ep = int(X_test.shape[0])
    log.info(
        "%s | step 6/6 | predict test subject | epochs=%d (vectorized from cache)",
        fold_label,
        n_ep,
    )
    t0 = time.time()
    X_sel = X_test[:, cols]
    hyp_pred = rf.predict(X_sel).astype(np.int64)
    hyp_true = np.asarray(hyp_true, dtype=np.int64)
    log.info("%s | step 6/6 | done | predicted %d epochs | %.1fs", fold_label, len(hyp_pred), time.time() - t0)
    mask = hyp_true >= 0
    if mask.sum():
        acc = float(accuracy_score(hyp_true[mask], hyp_pred[mask]))
        kap = float(cohen_kappa_score(hyp_true[mask], hyp_pred[mask]))
        f1m = float(
            f1_score(hyp_true[mask], hyp_pred[mask], average="macro", zero_division=0)
        )
    else:
        acc = kap = f1m = float("nan")

    test_basename = test_basename_preview
    log.info(
        "%s | metrics | accuracy=%.4f cohen_kappa=%.4f f1_macro=%.4f",
        fold_label,
        acc,
        kap,
        f1m,
    )

    experiment_id = str(uuid.uuid4())
    run_dir = os.path.join(save_folder, experiment_id)
    os.makedirs(run_dir, exist_ok=True)
    record_parent = os.path.dirname(os.path.normpath(test_record))

    bundle = {
        "kind": "memar_et_al_sklearn_rf",
        "rf": rf,
        "selected_feature_indices": cols,
        "selected_feature_names": picked,
        "all_feature_names": all_names,
        "mrmr_k": mrmr_k,
        "n_estimators": n_estimators,
        "kw_p": kw_p,
        "eeg_channel": eeg_channel,
        "memar_et_al_config_path": memar_config_path,
    }
    save_sklearn_bundle_tar(os.path.join(run_dir, "best_model.gz"), bundle)

    hyp_out = {record_parent: hyp_pred.astype(int).tolist()}
    with open(os.path.join(run_dir, "hypnograms.json"), "w", encoding="utf-8") as f:
        json.dump(hyp_out, f, indent=2)

    try:
        repo = git.Repo(search_parent_directories=True)
        git_branch = repo.active_branch.name
        git_hash = repo.head.object.hexsha
    except Exception:
        git_branch = ""
        git_hash = ""
    metadata = {
        "git_branch": git_branch,
        "git_hash": git_hash,
        "begin": int(time.time()),
        "end": int(time.time()),
        "experiment_id": experiment_id,
    }

    perf_test = {"accuracy": acc, "cohen_kappa": kap, "f1_macro": f1m}
    perf_rec = {
        record_parent.replace("\\", "/"): {
            "accuracy": acc,
            "cohen_kappa": kap,
            "f1_macro": f1m,
        }
    }

    records_split = {
        "train_records": [os.path.basename(os.path.normpath(r)) for r in train_records],
        "validation_records": [],
        "test_records": [test_basename],
    }

    dataset_settings_desc = copy.deepcopy(dataset_setting)
    for key in ("h5_directory", "memmap_directory"):
        if key in dataset_settings_desc:
            dataset_settings_desc[key] = _to_rel(dataset_settings_desc[key])

    dp_desc = {
        "split": {
            "train": [_to_rel(r) for r in train_records],
            "val": [],
            "test": [_to_rel(test_record)],
        },
        "temporal_context": 21,
        "temporal_context_mode": "sequential",
        "transform_parameters": None,
    }

    experiment_description = {
        "metadata": metadata,
        "memar_et_al_subject_cv": True,
        "dataset_settings": dataset_settings_desc,
        "memmap_description": memmap_description,
        "groups_description": groups_description,
        "features_description": features_description,
        "dataset_parameters": dp_desc,
        "normalization_parameters": {"type": "none", "args": {}},
        "trainers_parameters": {
            "args": {
                "model": "sklearn.ensemble.RandomForestClassifier",
                "n_estimators": n_estimators,
                "max_features_sqrt": True,
                "mrmr_k": mrmr_k,
                "kruskal_wallis_p": kw_p,
                "memar_subject_cv_viii_b": True,
                "all_non_test_subjects_for_training": True,
            }
        },
        "net_parameters": {
            "type": "memar_et_al_rf",
            "feature_dim": len(all_names),
            "eeg_signal": eeg_channel,
            "memar_et_al_config": _to_rel(memar_config_path)
            if memar_config_path
            else None,
        },
        "performance_on_test_set": perf_test,
        "performance_per_records": perf_rec,
        "records_split": records_split,
    }

    with open(os.path.join(run_dir, "description.json"), "w", encoding="utf-8") as f:
        json.dump(experiment_description, f, indent=2)

    log.info(
        "%s | saved | %s | fold wall time %.1fs",
        fold_label,
        run_dir,
        time.time() - t_fold,
    )

    return run_dir


def _run_fold_unpack(kwargs: dict) -> str:
    """Top-level helper for ``ProcessPoolExecutor`` (must be picklable on Windows)."""
    return run_fold(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--folds",
        type=int,
        nargs="*",
        default=None,
        metavar="N",
        help="LOSO fold indices (default: all).",
    )
    parser.add_argument("--skip-memmap-build", action="store_true")
    parser.add_argument(
        "--no-force",
        action="store_true",
        help="Do not delete existing outputs under the experiment save folder.",
    )
    parser.add_argument(
        "--mrmr-k",
        type=int,
        default=40,
        help="mRMR target count after KW (paper Sec. VIII-B subject CV: 40).",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="RandomForest trees (paper Sec. VIII-B subject CV: 100).",
    )
    parser.add_argument("--kw-p", type=float, default=0.01, help="Kruskal–Wallis p threshold (keep if <=)")
    parser.add_argument("--memmap-only", action="store_true", help="Only build memmaps; no training.")
    parser.add_argument("--eeg-only", action="store_true", help="EEG-only memmap (matches *_eeg hash if used).")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to memar_et_al_config.json (default: scripts/base_experiments/memar_et_al/memar_et_al_config.json).",
    )
    parser.add_argument(
        "--eeg-signal",
        type=str,
        default=None,
        metavar="PATH",
        help="Override eeg_signal for this run (e.g. signals/eeg/C3_M2). Default: value from --config.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="N",
        help="Number of parallel processes for LOSO folds (default: 1 sequential). "
        "Use with --no-force when reusing an existing experiment directory. "
        "Combine with --skip-memmap-build so workers do not rewrite memmaps.",
    )
    parser.add_argument(
        "--rf-n-jobs",
        type=int,
        default=None,
        metavar="N",
        help="RandomForest n_jobs per fold. Default: -1 if --workers 1, else 1 (avoid CPU oversubscription).",
    )
    parser.add_argument(
        "--feat-workers",
        type=int,
        default=None,
        metavar="N",
        help="Parallel processes for step 2 (Memar features on training subjects). "
        "Default: all logical CPUs when --workers 1; 1 when --workers > 1. Use 1 for sequential tqdm.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only warnings/errors to the console (disables per-step INFO and tqdm bars).",
    )
    parser.add_argument(
        "--refresh-feature-cache",
        action="store_true",
        help="Delete cached Memar .npz features for this memmap/EEG/bands key and re-extract all subjects.",
    )
    args = parser.parse_args()

    experiments_directory = os.path.join(os.path.dirname(__file__), "base_experiments")
    exp_name = "memar_et_al"
    dataset = "dodh"
    datasets = {dataset: DODH_SETTINGS}

    experiment_directory = os.path.join(experiments_directory, exp_name)
    memar_config_path = os.path.abspath(args.config) if args.config else default_memar_et_al_config_path()
    eeg_channel = args.eeg_signal.strip() if args.eeg_signal else get_eeg_signal(memar_config_path)

    memmaps_description = json.load(open(os.path.join(experiment_directory, "memmaps.json")))
    raw_memmap = [m for m in memmaps_description if m.get("dataset") == dataset][0]
    memmap_description = copy.deepcopy(raw_memmap)
    del memmap_description["dataset"]
    if args.eeg_only:
        memmap_description = filter_memmap_signals_eeg_only(memmap_description)

    dataset_parameters = json.load(open(os.path.join(experiment_directory, "dataset.json")))
    dataset_parameter = dataset_parameters[0]
    dataset_setting = datasets[dataset]
    save_folder = os.path.join(EXPERIMENTS_DIRECTORY, dataset, exp_name)
    if args.eeg_only:
        save_folder = os.path.join(EXPERIMENTS_DIRECTORY, dataset, exp_name + "_eeg")

    if os.path.exists(save_folder) and not args.no_force and not args.memmap_only:
        shutil.rmtree(save_folder)

    description_hash = memmap_hash(memmap_description)
    dataset_dir = os.path.join(dataset_setting["memmap_directory"], description_hash)

    if not args.skip_memmap_build:
        h5_to_memmaps(
            records=[
                os.path.join(dataset_setting["h5_directory"], r)
                for r in os.listdir(dataset_setting["h5_directory"])
            ],
            memmap_directory=dataset_setting["memmap_directory"],
            memmap_description=memmap_description,
            parallel=False,
            error_tolerant=False,
        )
    elif not os.path.isdir(dataset_dir):
        raise FileNotFoundError("skip_memmap_build but missing {}".format(dataset_dir))
    if not memmap_directory_ready(dataset_dir):
        raise FileNotFoundError("Memmap incomplete: {}".format(dataset_dir))

    if args.memmap_only:
        print("memmap-only: ready at {!r}".format(dataset_dir))
        return

    available_dreem_records = [
        os.path.join(dataset_dir, record)
        for record in os.listdir(dataset_dir)
        if ".json" not in record
    ]
    rd.seed(2019)
    rd.shuffle(available_dreem_records)
    folds = [[r] for r in available_dreem_records]

    if args.folds is None:
        effective_fold_indices = list(range(len(folds)))
    else:
        effective_fold_indices = list(args.folds)

    os.makedirs(save_folder, exist_ok=True)

    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if args.rf_n_jobs is None:
        rf_n_jobs = 1 if args.workers > 1 else -1
    else:
        rf_n_jobs = args.rf_n_jobs

    if args.feat_workers is None:
        feat_n_jobs = 1 if args.workers > 1 else max(1, os.cpu_count() or 1)
    else:
        feat_n_jobs = max(1, int(args.feat_workers))

    fold_jobs: list[dict] = []
    for i, fold in enumerate(folds):
        if i not in effective_fold_indices:
            continue
        train_records = [r for r in available_dreem_records if r not in fold]
        rd.seed(2019 + i)
        rd.shuffle(train_records)
        fold_jobs.append(
            {
                "fold_idx": i,
                "memmap_description": memmap_description,
                "dataset_setting": dataset_setting,
                "train_records": train_records,
                "test_records": fold,
                "save_folder": save_folder,
                "mrmr_k": args.mrmr_k,
                "n_estimators": args.n_estimators,
                "kw_p": args.kw_p,
                "random_state": 2019 + i,
                "eeg_channel": eeg_channel,
                "memar_config_path": memar_config_path,
                "rf_n_jobs": rf_n_jobs,
                "show_progress": not args.quiet,
                "quiet": args.quiet,
            }
        )

    bands_cfg = load_bands_config()
    feature_cache_key = compute_feature_cache_key(memmap_description, eeg_channel, bands_cfg)
    fcache_dir = feature_cache_path(save_folder, feature_cache_key)
    if fold_jobs:
        _memar_logger(args.quiet)
        if args.refresh_feature_cache and os.path.isdir(fcache_dir):
            shutil.rmtree(fcache_dir)
        ensure_memar_feature_cache(
            fcache_dir,
            feature_cache_key,
            description_hash,
            available_dreem_records,
            memmap_description,
            eeg_channel,
            memar_config_path,
            feat_n_jobs,
            show_progress=not args.quiet,
            quiet=args.quiet,
        )
        for job in fold_jobs:
            job["feature_cache_dir"] = fcache_dir

    if args.workers == 1:
        for job in fold_jobs:
            j = dict(job)
            fi = j["fold_idx"]
            tr = j["train_records"]
            print(
                "Fold {} test={} train_subjects={} eeg_signal={}".format(
                    fi, j["test_records"][0], len(tr), eeg_channel
                )
            )
            run_fold(**j)
    else:
        print(
            "Running {} fold(s) with {} worker process(es); RF n_jobs={} per fold.".format(
                len(fold_jobs), args.workers, rf_n_jobs
            )
        )
        if not args.quiet:
            print(
                "Each fold logs and shows tqdm in its own process; lines may interleave on the console."
            )
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            futs = []
            for job in fold_jobs:
                j = dict(job)
                fi = j["fold_idx"]
                tr = j["train_records"]
                print(
                    "Submit fold {} test={} train_subjects={} eeg_signal={}".format(
                        fi, j["test_records"][0], len(tr), eeg_channel
                    )
                )
                futs.append((fi, executor.submit(_run_fold_unpack, j)))
            for fi, fut in futs:
                run_dir = fut.result()
                print("Fold {} finished: {}".format(fi, run_dir))


if __name__ == "__main__":
    main()
