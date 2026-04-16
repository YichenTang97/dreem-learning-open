"""
train_cnn_rnn.py  —  Script 2
==============================
Train the proposed CNN-RNN model (CNNMaxPoolEpochEncoder + unidirectional LSTM)
on a Dreem dataset using Leave-One-Out Cross-Validation (LOOCV), matching the
evaluation protocol used for SimpleSleepNet in run_base_experiments.py.

Minimal usage (all paths/settings resolved automatically):
    python sol_experiments/train_cnn_rnn.py

Custom usage:
    python sol_experiments/train_cnn_rnn.py \\
        --dataset dodo \\
        --folds 0 1 2 \\
        --out_dir /custom/experiment/dir/
"""

from __future__ import annotations

import argparse
import json
import os
import random
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
from dreem_learning_open.preprocessings.h5_to_memmap import h5_to_memmaps
from dreem_learning_open.utils.train_test_val_split import train_test_val_split
from dreem_learning_open.utils.run_experiments import memmap_hash

MODEL_NAME = "cnn_rnn"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_configs(dataset_name: str) -> tuple:
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

    normalization = json.load(open(os.path.join(CNN_RNN_CONFIGS_DIR, "normalization.json")))
    trainer_cfg   = json.load(open(os.path.join(CNN_RNN_CONFIGS_DIR, "trainer.json")))
    net_cfg       = json.load(open(os.path.join(CNN_RNN_CONFIGS_DIR, "net.json")))
    dataset_cfg   = json.load(open(os.path.join(CNN_RNN_CONFIGS_DIR, "dataset.json")))[0]
    transform     = json.load(open(os.path.join(CNN_RNN_CONFIGS_DIR, "transform.json")))

    return memmap_desc, normalization, trainer_cfg, net_cfg, dataset_cfg, transform


# ---------------------------------------------------------------------------
# LOOCV setup
# ---------------------------------------------------------------------------

def build_memmaps_and_folds(dataset_settings: dict, memmap_desc: dict) -> tuple:
    h5_files = [
        os.path.join(dataset_settings["h5_directory"], f)
        for f in os.listdir(dataset_settings["h5_directory"])
        if f.endswith(".h5")
    ]
    if not h5_files:
        raise FileNotFoundError(
            f"No .h5 files in {dataset_settings['h5_directory']}"
        )
    print(f"  Pre-computing memmaps for {len(h5_files)} recordings ...")
    h5_to_memmaps(
        records=h5_files,
        memmap_description=memmap_desc,
        memmap_directory=dataset_settings["memmap_directory"],
        parallel=False,
        error_tolerant=False,
    )
    memmaps_dir = os.path.join(
        dataset_settings["memmap_directory"], memmap_hash(memmap_desc)
    )
    # Same construction order as dreem_learning_open.utils.run_experiments.run_experiments
    memmap_records = [
        os.path.join(memmaps_dir, r)
        for r in os.listdir(memmaps_dir)
        if ".json" not in r
    ]
    print(f"  {len(memmap_records)} memmap records ready.")
    random.seed(2019)
    random.shuffle(memmap_records)
    folds = [[r] for r in memmap_records]   # LOOCV: one held-out per fold
    return memmap_records, folds


# ---------------------------------------------------------------------------
# Single fold
# ---------------------------------------------------------------------------

def run_fold(fold_idx: int, fold: list, all_records: list,
             memmap_desc: dict, dataset_settings: dict,
             normalization: dict, trainer_cfg: dict,
             net_cfg: dict, dataset_cfg: dict, transform: list,
             save_folder: str) -> None:
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
    return p


def main(args: argparse.Namespace) -> None:
    dataset_settings = DATASET_SETTINGS[args.dataset]
    out_dir = args.out_dir or default_exp_dir(args.dataset, MODEL_NAME)

    print_config("train_cnn_rnn.py", {
        "dataset":    args.dataset,
        "out_dir":    out_dir,
        "folds":      args.folds or "all",
        "config_dir": CNN_RNN_CONFIGS_DIR,
    })

    memmap_desc, normalization, trainer_cfg, net_cfg, dataset_cfg, transform = \
        load_configs(args.dataset)

    all_records, folds = build_memmaps_and_folds(dataset_settings, memmap_desc)

    folds_to_run = args.folds if args.folds else list(range(len(folds)))
    print(f"Running {len(folds_to_run)} / {len(folds)} fold(s).\n")

    for i in folds_to_run:
        if i >= len(folds):
            print(f"  WARNING: fold {i} out of range ({len(folds)} total). Skipping.")
            continue
        run_fold(i, folds[i], all_records, memmap_desc, dataset_settings,
                 normalization, trainer_cfg, net_cfg, dataset_cfg, transform, out_dir)

    print(f"\nTraining complete.  Results saved to: {out_dir}")
    print(f"Next step: python sol_experiments/evaluate_sol.py\n")


if __name__ == "__main__":
    main(build_parser().parse_args())
