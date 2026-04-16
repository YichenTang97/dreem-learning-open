"""
sol_config.py
=============
Central configuration for all SOL experiment scripts.

All scripts import their default paths and hyperparameters from here so that:
  - Running `python sol_experiments/<script>.py` with no args just works.
  - Settings from the main project (settings.py) are the single source of truth
    for data/experiment directories.
  - Changing a path or hyperparameter in one place propagates everywhere.

Override any value at the command line if needed — every script exposes its
full parameter set via argparse.
"""

from __future__ import annotations

import os
import sys

# Make sure the package root is importable regardless of where the script is
# invoked from.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from dreem_learning_open.settings import (
        DODH_SETTINGS,
        DODO_SETTINGS,
        EXPERIMENTS_DIRECTORY,
    )
except ModuleNotFoundError:
    # settings.py has not been created yet — give a clear actionable message.
    import textwrap
    _msg = textwrap.dedent("""
        ┌─────────────────────────────────────────────────────────────┐
        │  SETUP REQUIRED: create settings.py before running scripts  │
        │                                                             │
        │  1. Copy the template:                                      │
        │       cp dreem_learning_open/settings_template.py           │
        │          dreem_learning_open/settings.py                    │
        │                                                             │
        │  2. Edit settings.py and set BASE_DIRECTORY to the folder   │
        │     where you want data, memmaps and experiments stored.    │
        │                                                             │
        │  3. Re-run your script.                                     │
        └─────────────────────────────────────────────────────────────┘
    """).strip()
    print(_msg)
    raise SystemExit(1)

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------

#: Absolute path to the sol_experiments/ folder itself.
SOL_DIR = os.path.dirname(os.path.abspath(__file__))

#: Where SOL target JSON files are written by compute_sol_targets.py.
SOL_DATA_DIR = os.path.join(SOL_DIR, "data")

#: Where SOL evaluation result JSON files are written by evaluate_sol.py.
SOL_RESULTS_DIR = os.path.join(SOL_DIR, "results")

#: Where the CNN-RNN config JSONs live.
CNN_RNN_CONFIGS_DIR = os.path.join(SOL_DIR, "configs", "cnn_rnn")

# ---------------------------------------------------------------------------
# Dataset shortcuts  (map dataset name → dreem settings dict)
# ---------------------------------------------------------------------------

DATASET_SETTINGS: dict = {
    "dodh": DODH_SETTINGS,
    "dodo": DODO_SETTINGS,
}

# ---------------------------------------------------------------------------
# Per-dataset default paths
# ---------------------------------------------------------------------------

def h5_dir(dataset: str = "dodh") -> str:
    """Return the h5 directory for *dataset* (from project settings)."""
    return DATASET_SETTINGS[dataset]["h5_directory"]


def sol_targets_path(dataset: str = "dodh") -> str:
    """Standard output path for compute_sol_targets.py."""
    os.makedirs(SOL_DATA_DIR, exist_ok=True)
    return os.path.join(SOL_DATA_DIR, f"sol_targets_{dataset}.json")


def exp_dir(dataset: str = "dodh", model: str = "cnn_rnn") -> str:
    """Experiment directory for *model* on *dataset* (inside EXPERIMENTS_DIRECTORY)."""
    return os.path.join(EXPERIMENTS_DIRECTORY, dataset, model)


def sol_results_path(dataset: str = "dodh", model: str = "cnn_rnn",
                     tag: str = "") -> str:
    """Standard path for an evaluate_sol.py output JSON."""
    os.makedirs(SOL_RESULTS_DIR, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    return os.path.join(SOL_RESULTS_DIR, f"sol_{model}_{dataset}{suffix}.json")


def finetune_dir(dataset: str = "dodh", base_model: str = "cnn_rnn",
                 cutoff_minutes: float = 10.0, alpha: float = 0.5) -> str:
    """
    Standard output directory for a fine-tuned model.
    Name encodes the key hyperparameters so multiple runs don't overwrite.
    """
    tag = f"ft_c{cutoff_minutes:.0f}m_a{alpha:.2f}"
    return os.path.join(EXPERIMENTS_DIRECTORY, dataset, f"{base_model}_{tag}")


# ---------------------------------------------------------------------------
# Default hyperparameters
# ---------------------------------------------------------------------------

#: CNN-RNN training defaults.
TRAIN_DEFAULTS: dict = {
    "dataset":  "dodh",
    "folds":    None,      # None = run all LOOCV folds
}

#: SOL fine-tuning defaults (all exposed as CLI args).
FINETUNE_DEFAULTS: dict = {
    "dataset":         "dodh",
    "base_model":      "cnn_rnn",
    "cutoff_minutes":  10.0,   # minutes after true SOL to include in training window
    "alpha":           0.5,    # weight of CE loss; (1-alpha) = SOL loss weight
    "lr":              1e-4,
    "epochs":          30,
    "patience":        7,
    "folds":           None,   # None = all folds
}

#: SOL evaluation defaults.
EVALUATE_DEFAULTS: dict = {
    "dataset":              "dodh",
    "model":                "cnn_rnn",
    "require_consecutive":  1,
}

#: SOL target extraction defaults.
TARGETS_DEFAULTS: dict = {
    "dataset":              "dodh",
    "require_consecutive":  1,
    "inspect_first":        False,
}

# ---------------------------------------------------------------------------
# Convenience printer (called by each script at startup)
# ---------------------------------------------------------------------------

def print_config(script_name: str, params: dict) -> None:
    """Pretty-print the resolved configuration for a script run."""
    width = 60
    print(f"\n{'='*width}")
    print(f"  {script_name}")
    print(f"{'='*width}")
    for k, v in params.items():
        print(f"  {k:<26}: {v}")
    print(f"{'='*width}\n")
