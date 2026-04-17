"""
sol_config.py
=============
Central configuration for SOL (Sleep Onset Latency) experiment scripts.

Staging **pretraining** (under ``scripts/``) uses **consensus** hypnogram labels.
That is not the same supervision as **SOL** derived from expert scorers. The
``sol_experiments`` pipeline is **model-agnostic**: any folder name under
``EXPERIMENTS_DIRECTORY/<dataset>/<model>/`` can be evaluated, fine-tuned on SOL,
and re-evaluated.

**Leave-one-subject-out (LOOCV)** matches pretraining: one held-out test record
per fold. SOL **targets** are dataset-wide (every record has a reference SOL).
SOL **evaluations** and **finetuned** artifacts are stored **per fold** under
``SOL_DIRECTORY`` (see ``sol_eval_model_root``, ``sol_eval_fold_dir``,
``finetune_dir``).

Paths come from ``dreem_learning_open.settings`` (``BASE_DIRECTORY``,
``REPO_ROOT``, ``SOL_DIRECTORY``, ``EXPERIMENTS_DIRECTORY``).
"""

from __future__ import annotations

import os
import sys

from typing import Optional

_REPO_ROOT_FOR_IMPORT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT_FOR_IMPORT not in sys.path:
    sys.path.insert(0, _REPO_ROOT_FOR_IMPORT)

try:
    from dreem_learning_open.settings import (
        BASE_DIRECTORY,
        DODO_SETTINGS,
        DODH_SETTINGS,
        EXPERIMENTS_DIRECTORY,
        REPO_ROOT,
        SOL_DIRECTORY,
    )
except ModuleNotFoundError:
    import textwrap

    _msg = textwrap.dedent(
        """
        ┌─────────────────────────────────────────────────────────────┐
        │  SETUP REQUIRED: create settings.py before running scripts  │
        │                                                             │
        │  1. Copy the template:                                      │
        │       cp dreem_learning_open/settings_template.py           │
        │          dreem_learning_open/settings.py                    │
        │                                                             │
        │  2. Edit settings.py and set BASE_DIRECTORY to the folder │
        │     where you want data, memmaps and experiments stored.    │
        │                                                             │
        │  3. Re-run your script.                                     │
        └─────────────────────────────────────────────────────────────┘
    """
    ).strip()
    print(_msg)
    raise SystemExit(1)

# ---------------------------------------------------------------------------
# Package / repo paths
# ---------------------------------------------------------------------------

SOL_DIR = os.path.dirname(os.path.abspath(__file__))

#: CNN-RNN **pretraining** configs (not SOL-specific); lives under scripts/base_experiments.
CNN_RNN_CONFIGS_DIR = os.path.join(REPO_ROOT, "scripts", "base_experiments", "cnn_rnn")

_EVAL_REPO_CANDIDATE = os.path.join(REPO_ROOT, "dreem-learning-evaluation")
EVAL_REPO_DIR: Optional[str] = (
    _EVAL_REPO_CANDIDATE if os.path.isdir(_EVAL_REPO_CANDIDATE) else None
)

# ---------------------------------------------------------------------------
# SOL artifact roots under BASE_DIRECTORY/sol/
# ---------------------------------------------------------------------------

SOL_TARGETS_ROOT = os.path.join(SOL_DIRECTORY, "targets")
SOL_EVALUATIONS_ROOT = os.path.join(SOL_DIRECTORY, "evaluations")
SOL_FINETUNED_ROOT = os.path.join(SOL_DIRECTORY, "finetuned")
BASE_DIRECTORY_ABS = os.path.abspath(BASE_DIRECTORY)


def sol_targets_path(dataset: str = "dodh") -> str:
    """Dataset-wide expert SOL reference JSON (all records)."""
    d = os.path.join(SOL_TARGETS_ROOT, dataset)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "sol_targets.json")


def sol_eval_model_root(dataset: str, model: str) -> str:
    """Root directory for per-fold SOL evaluation outputs for one pretrained model."""
    p = os.path.join(SOL_EVALUATIONS_ROOT, dataset, model)
    os.makedirs(p, exist_ok=True)
    return p


def sol_eval_fold_dir(dataset: str, model: str, fold_idx: int) -> str:
    """Per-fold LOSO evaluation directory (``fold_XX`` matches finetune_sol)."""
    p = os.path.join(sol_eval_model_root(dataset, model), "fold_{:02d}".format(fold_idx))
    os.makedirs(p, exist_ok=True)
    return p


def sol_eval_summary_path(dataset: str, model: str) -> str:
    """Optional rollup JSON aggregating all folds for one model."""
    return os.path.join(sol_eval_model_root(dataset, model), "summary.json")


def exp_dir(dataset: str = "dodh", model: str = "cnn_rnn") -> str:
    """Pretrained experiment directory (LOOCV UUID folders live here)."""
    return os.path.join(EXPERIMENTS_DIRECTORY, dataset, model)


def finetune_dir(
    dataset: str = "dodh",
    base_model: str = "cnn_rnn",
    cutoff_minutes: float = 10.0,
    alpha: float = 0.5,
) -> str:
    """
    Root directory for SOL fine-tuning outputs (contains ``fold_XX/`` per LOSO fold).
    Lives under ``SOL_DIRECTORY/finetuned``, not under EXPERIMENTS_DIRECTORY.
    """
    tag = "ft_c{:.0f}m_a{:.2f}".format(cutoff_minutes, alpha)
    p = os.path.join(SOL_FINETUNED_ROOT, dataset, "{}_{}".format(base_model, tag))
    os.makedirs(p, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Dataset shortcuts
# ---------------------------------------------------------------------------

DATASET_SETTINGS: dict = {
    "dodh": DODH_SETTINGS,
    "dodo": DODO_SETTINGS,
}


def h5_dir(dataset: str = "dodh") -> str:
    return DATASET_SETTINGS[dataset]["h5_directory"]


def to_base_directory_relative(path: str) -> str:
    """
    Return `path` in repo-style ``data/...`` form when possible.
    If the path is outside BASE_DIRECTORY (or cannot be relativized),
    return it unchanged.
    """
    if not path:
        return path
    abs_path = os.path.abspath(path)
    try:
        if os.path.commonpath([BASE_DIRECTORY_ABS, abs_path]) == BASE_DIRECTORY_ABS:
            rel_from_data = os.path.relpath(abs_path, BASE_DIRECTORY_ABS).replace("\\", "/")
            return "data/{}".format(rel_from_data)
    except ValueError:
        # Different drive letters on Windows can raise ValueError.
        return path
    return path


TRAIN_DEFAULTS: dict = {
    "dataset": "dodh",
    "folds": None,
}

FINETUNE_DEFAULTS: dict = {
    "dataset": "dodh",
    "base_model": "cnn_rnn",
    "cutoff_minutes": 10.0,
    "alpha": 0.5,
    "lr": 1e-4,
    "epochs": 30,
    "patience": 7,
    "folds": None,
}

EVALUATE_DEFAULTS: dict = {
    "dataset": "dodh",
    "model": "cnn_rnn",
    "require_consecutive": 1,
}

TARGETS_DEFAULTS: dict = {
    "dataset": "dodh",
    "require_consecutive": 1,
    "inspect_first": False,
}


def print_config(script_name: str, params: dict) -> None:
    width = 60
    print("\n{}".format("=" * width))
    print("  {}".format(script_name))
    print("{}".format("=" * width))
    for k, v in params.items():
        print("  {:<26}: {}".format(k, v))
    print("{}\n".format("=" * width))
