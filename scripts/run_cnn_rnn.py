"""
run_cnn_rnn.py
==============
Train the proposed CNN-RNN model (CNNMaxPoolEpochEncoder + unidirectional LSTM)
on a Dreem dataset using Leave-One-Out Cross-Validation (LOOCV), matching the
evaluation protocol used for SimpleSleepNet in run_base_experiments.py.

Resume semantics match ``dreem_learning_open.utils.run_experiments`` /
``scripts/run_simple_sleep_net_only.py`` (``--no-force`` mode):

* Memmaps: if ``groups_description.json`` already exists for this memmap hash,
  skip ``h5_to_memmaps`` unless ``--rebuild-memmaps``.
* ``--memmap-only``: build or reuse memmaps for this dataset/config then exit (no training).
* Folds: skip runs that already have a *complete* UUID directory (see
  ``_is_run_complete`` in ``run_experiments``).
* Failed / interrupted folds: reuse the same UUID directory when resuming
  (same as ``--reuse-incomplete-uuids``), unless ``--no-reuse-incomplete-uuids``
  or ``--force``.

Minimal usage (from repository root):
    python scripts/run_cnn_rnn.py

Custom usage:
    python scripts/run_cnn_rnn.py \\
        --dataset dodo \\
        --folds 0 1 2 \\
        --out_dir /custom/experiment/dir/
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sol_experiments.sol_config import (
    DATASET_SETTINGS,
    CNN_RNN_CONFIGS_DIR,
    TRAIN_DEFAULTS,
    exp_dir as default_exp_dir,
    print_config,
)
from dreem_learning_open.logger.logger import log_experiment
from dreem_learning_open.utils.experiment_fold_index import loov_record_paths_in_fold_index_order
from dreem_learning_open.preprocessings.h5_to_memmap import h5_to_memmaps
from dreem_learning_open.utils.train_test_val_split import train_test_val_split
from dreem_learning_open.utils.memmap_eeg import filter_memmap_signals_eeg_only, with_eeg_model_suffix
from dreem_learning_open.utils.run_experiments import (
    memmap_hash,
    _find_incomplete_run_ids_by_test_record,
    _is_run_complete,
    _recover_test_record_id,
)

MODEL_NAME_BASE = "cnn_rnn"


def _complete_test_record_ids(save_folder: str) -> set:
    """Test-record basenames that already have a finished run under ``save_folder``."""
    complete = set()
    if not os.path.isdir(save_folder):
        return complete
    for run_uuid in os.listdir(save_folder):
        run_dir = os.path.join(save_folder, run_uuid)
        if not os.path.isdir(run_dir):
            continue
        desc_path = os.path.join(run_dir, "description.json")
        if not os.path.isfile(desc_path):
            continue
        try:
            with open(desc_path, "r") as f:
                description = json.load(f)
        except Exception:
            continue
        if not _is_run_complete(run_dir, description):
            continue
        tid = _recover_test_record_id(description)
        if tid:
            complete.add(tid)
    return complete


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_configs(dataset_name: str, *, eeg_only: bool = False) -> tuple:
    """Load all JSON config files for the CNN-RNN experiment."""
    all_memmaps = json.load(open(os.path.join(CNN_RNN_CONFIGS_DIR, "memmaps.json")))
    memmap_desc = next(
        (m for m in all_memmaps if m.get("dataset") == dataset_name),
        None,
    )
    if memmap_desc is None:
        available = [m.get("dataset") for m in all_memmaps]
        raise ValueError(
            f"No memmap block for dataset {dataset_name!r} in memmaps.json "
            f"(available: {available})"
        )
    memmap_desc = {k: v for k, v in memmap_desc.items() if k != "dataset"}
    if eeg_only:
        memmap_desc = filter_memmap_signals_eeg_only(memmap_desc)

    normalization = json.load(open(os.path.join(CNN_RNN_CONFIGS_DIR, "normalization.json")))
    trainer_cfg   = json.load(open(os.path.join(CNN_RNN_CONFIGS_DIR, "trainer.json")))
    net_cfg       = json.load(open(os.path.join(CNN_RNN_CONFIGS_DIR, "net.json")))
    dataset_cfg   = json.load(open(os.path.join(CNN_RNN_CONFIGS_DIR, "dataset.json")))[0]
    transform     = json.load(open(os.path.join(CNN_RNN_CONFIGS_DIR, "transform.json")))

    return memmap_desc, normalization, trainer_cfg, net_cfg, dataset_cfg, transform


# ---------------------------------------------------------------------------
# LOOCV setup
# ---------------------------------------------------------------------------

def _memmap_pipeline_ready(memmaps_dir: str) -> bool:
    """True if a prior ``h5_to_memmaps`` run finished for this hash (cf. ``log_experiment``)."""
    return (
        os.path.isfile(os.path.join(memmaps_dir, "groups_description.json"))
        and os.path.isfile(os.path.join(memmaps_dir, "features_description.json"))
    )


def build_memmaps_and_folds(
    dataset_settings: dict,
    memmap_desc: dict,
    *,
    rebuild_memmaps: bool,
) -> tuple:
    memmaps_dir = os.path.join(
        dataset_settings["memmap_directory"], memmap_hash(memmap_desc)
    )
    h5_files = [
        os.path.join(dataset_settings["h5_directory"], f)
        for f in os.listdir(dataset_settings["h5_directory"])
        if f.endswith(".h5")
    ]
    if not h5_files:
        raise FileNotFoundError(
            f"No .h5 files in {dataset_settings['h5_directory']}"
        )

    if rebuild_memmaps or not _memmap_pipeline_ready(memmaps_dir):
        print(
            f"  Building memmaps under {memmaps_dir} "
            f"({'rebuild' if rebuild_memmaps else 'cache missing'}) …"
        )
        h5_to_memmaps(
            records=h5_files,
            memmap_description=memmap_desc,
            memmap_directory=dataset_settings["memmap_directory"],
            parallel=False,
            error_tolerant=False,
            force=rebuild_memmaps,
        )
    else:
        print(f"  Using existing memmaps: {memmaps_dir}")

    if not _memmap_pipeline_ready(memmaps_dir):
        raise FileNotFoundError(
            f"Memmap directory is unusable after build: {memmaps_dir!r}"
        )

    # Same construction order as run_experiments / index_experiments (portable fold indices).
    memmap_records = loov_record_paths_in_fold_index_order(memmaps_dir)
    print(f"  {len(memmap_records)} memmap records ready.")
    folds = [[r] for r in memmap_records]   # LOOCV: one held-out per fold
    return memmap_records, folds


# ---------------------------------------------------------------------------
# Single fold
# ---------------------------------------------------------------------------

def run_fold(
    fold_idx: int,
    fold: list,
    all_records: list,
    memmap_desc: dict,
    dataset_settings: dict,
    normalization: dict,
    trainer_cfg: dict,
    net_cfg: dict,
    dataset_cfg: dict,
    transform: list,
    save_folder: str,
    experiment_id: str | None,
) -> None:
    other = [r for r in all_records if r not in fold]
    random.seed(2019 + fold_idx)
    random.shuffle(other)
    train_records, val_records, _ = train_test_val_split(other, 0.8, 0.2, 0.0, seed=2019)

    test_name = os.path.basename(fold[0])
    print(f"\n--- Fold {fold_idx:02d} | test={test_name} | "
          f"train={len(train_records)} val={len(val_records)} ---")

    log_experiment(
        memmap_description=memmap_desc,
        dataset_settings=dataset_settings,
        trainer_parameters=trainer_cfg,
        normalization_parameters=normalization,
        net_parameters=net_cfg,
        dataset_parameters={
            "split": {"train": train_records, "val": val_records, "test": fold},
            "temporal_context":      dataset_cfg["temporal_context"],
            "temporal_context_mode": dataset_cfg["temporal_context_mode"],
            "transform_parameters":  transform,
        },
        save_folder=save_folder,
        parralel=True,
        generate_memmaps=False,
        experiment_id=experiment_id,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="LOOCV training of the CNN-RNN model on a Dreem dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dataset", default=TRAIN_DEFAULTS["dataset"],
        choices=list(DATASET_SETTINGS.keys()),
        help="Dataset to train on.",
    )
    p.add_argument(
        "--folds", nargs="+", type=int, default=TRAIN_DEFAULTS["folds"],
        help="Specific LOOCV fold indices to run (default: all).",
    )
    p.add_argument(
        "--out_dir", default=None,
        help="Experiment output directory override. "
             "Default: EXPERIMENTS_DIRECTORY/<dataset>/cnn_rnn/",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Delete the experiment output directory before running (all folds from scratch).",
    )
    p.add_argument(
        "--rebuild-memmaps",
        action="store_true",
        help="Force recomputation of memmaps for this pipeline hash (passes force=True to h5_to_memmaps).",
    )
    p.add_argument(
        "--no-reuse-incomplete-uuids",
        action="store_true",
        help="Always create a new UUID per fold instead of reusing an incomplete run directory.",
    )
    p.add_argument(
        "--eeg-only",
        action="store_true",
        help="Train on EEG channels only; save under EXPERIMENTS_DIRECTORY/<dataset>/cnn_rnn_eeg/.",
    )
    p.add_argument(
        "--memmap-only",
        action="store_true",
        help="Only ensure memmaps exist for this memmap hash (build if missing); no fold training.",
    )
    return p


def main(args: argparse.Namespace) -> None:
    dataset_settings = DATASET_SETTINGS[args.dataset]
    model_name = with_eeg_model_suffix(MODEL_NAME_BASE, args.eeg_only)
    out_dir = args.out_dir or default_exp_dir(args.dataset, model_name)

    if args.force and not args.memmap_only and os.path.isdir(out_dir):
        print(f"  --force: removing {out_dir}")
        shutil.rmtree(out_dir)

    print_config("run_cnn_rnn.py", {
        "dataset":    args.dataset,
        "model_name": model_name,
        "out_dir":    out_dir,
        "folds":      args.folds or "all",
        "config_dir": CNN_RNN_CONFIGS_DIR,
        "eeg_only":   args.eeg_only,
        "memmap_only": args.memmap_only,
        "force":      args.force,
        "rebuild_memmaps": args.rebuild_memmaps,
        "reuse_incomplete_uuids": not args.no_reuse_incomplete_uuids and not args.force,
    })

    memmap_desc, normalization, trainer_cfg, net_cfg, dataset_cfg, transform = \
        load_configs(args.dataset, eeg_only=args.eeg_only)

    all_records, folds = build_memmaps_and_folds(
        dataset_settings, memmap_desc, rebuild_memmaps=args.rebuild_memmaps
    )

    if args.memmap_only:
        memmaps_dir = os.path.join(
            dataset_settings["memmap_directory"], memmap_hash(memmap_desc)
        )
        print(
            f"\nMemmap-only: pipeline ready at {memmaps_dir!r} "
            f"({len(all_records)} record(s)). Exiting without training.\n"
        )
        return

    complete_tests = _complete_test_record_ids(out_dir)
    incomplete_by_test = {}
    if not args.force and not args.no_reuse_incomplete_uuids:
        incomplete_by_test = _find_incomplete_run_ids_by_test_record(out_dir)

    folds_to_run = args.folds if args.folds else list(range(len(folds)))
    print(f"Running {len(folds_to_run)} / {len(folds)} fold(s).\n")

    for i in folds_to_run:
        if i >= len(folds):
            print(f"  WARNING: fold {i} out of range ({len(folds)} total). Skipping.")
            continue
        test_basename = os.path.basename(folds[i][0])
        if test_basename in complete_tests:
            print(f"  Fold {i:02d} | test={test_basename} — already complete, skipping.")
            continue
        exp_id = None
        if not args.force and not args.no_reuse_incomplete_uuids:
            exp_id = incomplete_by_test.get(test_basename)
        run_fold(
            i, folds[i], all_records, memmap_desc, dataset_settings,
            normalization, trainer_cfg, net_cfg, dataset_cfg, transform, out_dir,
            experiment_id=exp_id,
        )

    print(f"\nTraining complete.  Results saved to: {out_dir}")
    print(f"Next step: python sol_experiments/evaluate_sol.py\n")


if __name__ == "__main__":
    main(build_parser().parse_args())
