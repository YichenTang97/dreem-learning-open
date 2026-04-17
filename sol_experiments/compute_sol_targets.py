"""
compute_sol_targets.py  —  Script 1
=====================================
Compute per-scorer and consensus Sleep Onset Latency (SOL) targets for every
recording in a Dreem dataset and save them for downstream use.

DATA SOURCES
------------
Individual scorer labels come from the **dreem-learning-evaluation** submodule:
    scorers/{dataset}/scorer_{1-5}/{record_id}.json
Each file is a flat JSON list of integer sleep stage labels (0-4, 30s epochs).

The h5 files (downloaded from S3) contain only a single consensus hypnogram
under the key 'hypnogram'. The submodule is the only source for per-scorer data.

SOL DEFINITION
--------------
This script uses the AASM standard:
    SOL = first epoch scored as any non-wake stage (1, 2, 3, or 4).

The dreem-learning-evaluation repo (evaluation.py) uses a 3-consecutive-epoch
criterion and reports the INDEX of the 3rd such epoch — which is the AASM
first-sleep index + 2 epochs (60 s) offset. The difference is small for most
records; use --require_consecutive 3 if you want to match evaluation.py's
behaviour (but note the 2-epoch index offset, i.e. subtract 1 min).

Consensus SOL = mean( SOL_scorer_1, ..., SOL_scorer_5 )

**Expert reference (dataset-wide):** output is one JSON with a row per record.
LOSOCV fold membership for model evaluation comes from pretraining
(``experiment_fold_index`` / ``index_experiments``), not from this file.

Minimal usage (uses submodule by default):
    python sol_experiments/compute_sol_targets.py

Force h5-only fallback (no per-scorer data):
    python sol_experiments/compute_sol_targets.py --eval_repo_dir none

Custom dataset:
    python sol_experiments/compute_sol_targets.py --dataset dodo
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sol_experiments.sol_config import (
    h5_dir as default_h5_dir,
    sol_targets_path as default_targets_path,
    to_base_directory_relative,
    TARGETS_DEFAULTS,
    EVAL_REPO_DIR,
    print_config,
    DATASET_SETTINGS,
)
from sol_experiments.utils.sol_metrics import compute_sol, EPOCH_DURATION_S

N_SCORERS = 5   # DOD-H and DOD-O both have 5 scorers


# ---------------------------------------------------------------------------
# Individual scorer loading from dreem-learning-evaluation submodule
# ---------------------------------------------------------------------------

def check_eval_repo(eval_repo_dir: str, dataset: str) -> bool:
    """
    Return True if the evaluation repo contains scorer data for *dataset*.
    Expected path: {eval_repo_dir}/scorers/{dataset}/scorer_1/
    """
    required = os.path.join(eval_repo_dir, "scorers", dataset, "scorer_1")
    return os.path.isdir(required)


def load_scorer_labels(
        record_id: str,
        eval_repo_dir: str,
        dataset: str,
) -> Tuple[List[np.ndarray], List[str]]:
    """
    Load all N_SCORERS hypnograms for *record_id* from the evaluation repo.

    File layout: scorers/{dataset}/scorer_{i}/{record_id}.json
    Each file contains a flat JSON list: [0, 0, 1, 2, ...]

    Returns (arrays, names) where arrays is a list of 1-D int arrays,
    one per scorer.  Scorers whose file is missing are silently skipped.
    """
    base = os.path.join(eval_repo_dir, "scorers", dataset)
    arrays, names = [], []
    for i in range(1, N_SCORERS + 1):
        path = os.path.join(base, f"scorer_{i}", f"{record_id}.json")
        if os.path.exists(path):
            with open(path) as f:
                arr = np.array(json.load(f), dtype=int)
            arrays.append(arr)
            names.append(f"scorer_{i}")
    return arrays, names


# ---------------------------------------------------------------------------
# Fallback: consensus hypnogram from h5 file
# ---------------------------------------------------------------------------

def load_consensus_from_h5(h5_path: str) -> Optional[np.ndarray]:
    """Read the single majority-vote consensus 'hypnogram' from an h5 file."""
    with h5py.File(h5_path, "r") as f:
        if "hypnogram" not in f:
            return None
        hyp = f["hypnogram"][:]
        return hyp.astype(int) if hyp.ndim == 1 else None


def inspect_h5_structure(h5_path: str) -> None:
    """Print all dataset keys/shapes in an h5 file (debugging aid)."""
    print(f"\n{'='*60}\nInspecting: {os.path.basename(h5_path)}\n{'='*60}")
    with h5py.File(h5_path, "r") as f:
        keys: List[str] = []
        f.visititems(lambda n, obj: keys.append(n)
                     if isinstance(obj, h5py.Dataset) else None)
        for k in keys:
            try:
                print(f"  {k:<50}  shape={f[k].shape}  dtype={f[k].dtype}")
            except Exception as e:
                print(f"  {k:<50}  [error: {e}]")
    print()


# ---------------------------------------------------------------------------
# Per-record processing
# ---------------------------------------------------------------------------

def process_record(
        h5_path: str,
        eval_repo_dir: Optional[str],
        dataset: str,
        require_consecutive: int = 1,
        epoch_duration_s: int = EPOCH_DURATION_S,
) -> Dict:
    record_id = os.path.splitext(os.path.basename(h5_path))[0]
    result: Dict = {
        "record_id":           record_id,
        "h5_path":             to_base_directory_relative(h5_path),
        "source":              None,
        "n_scorers":           0,
        "scorer_keys_used":    [],
        "scorer_sols_min":     {},
        "consensus_sol_min":   None,
        "interrater_std_min":  None,
        "n_scorers_with_sleep": 0,
        "warning":             None,
    }

    scorer_arrays: List[np.ndarray] = []
    scorer_names:  List[str]        = []

    # ---- Preferred: individual scorer labels from evaluation submodule ----
    if eval_repo_dir is not None:
        scorer_arrays, scorer_names = load_scorer_labels(
            record_id, eval_repo_dir, dataset
        )
        if scorer_arrays:
            result["source"] = "eval_repo"
            if len(scorer_arrays) < N_SCORERS:
                result["warning"] = (
                    f"Only {len(scorer_arrays)}/{N_SCORERS} scorer files found "
                    f"for this record in the evaluation repo."
                )

    # ---- Fallback: consensus from h5 ----
    if not scorer_arrays:
        consensus = load_consensus_from_h5(h5_path)
        if consensus is None:
            result["warning"] = "No hypnogram found in h5 file."
            return result
        scorer_arrays = [consensus]
        scorer_names  = ["h5_consensus"]
        result["source"] = "h5_consensus"
        if eval_repo_dir is not None:
            result["warning"] = (
                f"Record '{record_id}' not found in evaluation repo "
                f"({eval_repo_dir}/scorers/{dataset}/). "
                "Fell back to h5 consensus."
            )
        else:
            result["warning"] = (
                "Individual scorer data not available (eval repo not provided). "
                "Using h5 consensus only — inter-rater statistics unavailable."
            )

    # ---- Compute SOL per scorer ----
    result["n_scorers"]        = len(scorer_arrays)
    result["scorer_keys_used"] = scorer_names

    valid_sols: List[float] = []
    for key, arr in zip(scorer_names, scorer_arrays):
        sol = compute_sol(arr, epoch_duration_s, require_consecutive)
        result["scorer_sols_min"][key] = round(sol, 3) if sol is not None else None
        if sol is not None:
            valid_sols.append(sol)

    result["n_scorers_with_sleep"] = len(valid_sols)
    if valid_sols:
        result["consensus_sol_min"] = round(float(np.mean(valid_sols)), 3)
        result["interrater_std_min"] = (
            round(float(np.std(valid_sols, ddof=0)), 3)
            if len(valid_sols) > 1 else None
        )
    else:
        result["warning"] = (result.get("warning") or "") + " No sleep detected."

    return result


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def print_summary(records_data: List[Dict]) -> None:
    valid    = [r for r in records_data if r["consensus_sol_min"] is not None]
    all_sols = [r["consensus_sol_min"] for r in valid]
    warnings = [r for r in records_data if r.get("warning")]
    sources  = {r.get("source") for r in records_data if r.get("source")}
    n_sc     = max((r["n_scorers"] for r in valid), default=0)

    print(f"\n{'='*65}")
    print("SOL TARGET SUMMARY")
    print(f"{'='*65}")
    print(f"  Records processed      : {len(records_data)}")
    print(f"  Records with valid SOL : {len(valid)}")
    print(f"  Data source(s)         : {', '.join(sorted(sources))}")
    print(f"  Scorers per record     : {n_sc}")

    if all_sols:
        print(f"\n  Consensus SOL (minutes) — mean of {n_sc} scorer SOLs:")
        print(f"    Mean ± SD : {np.mean(all_sols):.2f} ± {np.std(all_sols):.2f}")
        print(f"    Median    : {np.median(all_sols):.2f}")
        print(f"    Min / Max : {np.min(all_sols):.2f} / {np.max(all_sols):.2f}")

    valid_std = [r["interrater_std_min"] for r in valid
                 if r["interrater_std_min"] is not None]
    if valid_std and n_sc > 1:
        mean_std = np.mean(valid_std)
        print(f"\n  Inter-rater SOL variability (SD across {n_sc} scorers, minutes):")
        print(f"    Mean SD across records : {mean_std:.2f}")
        print(f"    Max SD                 : {np.max(valid_std):.2f}")
        print(f"\n  ↳ Human ceiling: your model should aim for MAE < {mean_std:.1f} min")
        print(f"    (This is the average disagreement between expert human scorers)")
    elif n_sc <= 1:
        print(f"\n  ↳ Inter-rater statistics unavailable (only consensus hypnogram).")
        print(f"    Add --eval_repo_dir or initialise the dreem-learning-evaluation")
        print(f"    submodule to enable per-scorer analysis.")

    if warnings:
        n_show = min(5, len(warnings))
        print(f"\n  Warnings ({len(warnings)} records, first {n_show}):")
        for r in warnings[:n_show]:
            print(f"    [{r['record_id']}] {r['warning']}")
        if len(warnings) > n_show:
            print(f"    … and {len(warnings)-n_show} more.")
    print(f"{'='*65}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compute SOL targets from Dreem scorer annotations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Scorer annotations are loaded from the dreem-learning-evaluation "
            "submodule at scorers/{dataset}/scorer_{1-5}/{record_id}.json. "
            "Initialise the submodule with: "
            "git submodule update --init --recursive"
        ),
    )
    p.add_argument(
        "--dataset", default=TARGETS_DEFAULTS["dataset"],
        choices=list(DATASET_SETTINGS.keys()),
        help="Which Dreem dataset to process.",
    )
    p.add_argument(
        "--h5_dir", default=None,
        help="Directory containing .h5 files. Default: from settings.py.",
    )
    p.add_argument(
        "--eval_repo_dir", default=EVAL_REPO_DIR,
        metavar="PATH",
        help=(
            "Root of the dreem-learning-evaluation repo. "
            "Default: dreem-learning-evaluation/ submodule (auto-detected). "
            "Pass 'none' to force the h5-consensus fallback."
        ),
    )
    p.add_argument(
        "--out", default=None,
        help="Output JSON path. "
             "Default: BASE_DIRECTORY/sol/targets/<dataset>/sol_targets.json",
    )
    p.add_argument(
        "--require_consecutive", type=int,
        default=TARGETS_DEFAULTS["require_consecutive"],
        help=(
            "Min consecutive non-wake epochs to confirm SOL. "
            "1 = AASM standard (first non-wake epoch). "
            "Note: dreem-learning-evaluation's evaluation.py uses 3, but "
            "reports the index of the 3rd epoch (adds ~1 min vs this script "
            "with require_consecutive=3)."
        ),
    )
    p.add_argument(
        "--inspect_first", action="store_true",
        default=TARGETS_DEFAULTS["inspect_first"],
        help="Print all h5 key names/shapes for the first file (debugging).",
    )
    return p


def main(args: argparse.Namespace) -> None:
    resolved_h5_dir = args.h5_dir or default_h5_dir(args.dataset)
    resolved_out    = args.out    or default_targets_path(args.dataset)

    # Handle --eval_repo_dir none (explicit opt-out)
    eval_repo_arg = (
        None if (args.eval_repo_dir or "").strip().lower() == "none"
        else args.eval_repo_dir
    )

    # Validate eval repo if provided
    if eval_repo_arg is not None:
        if not os.path.isdir(eval_repo_arg):
            print(f"WARNING: eval_repo_dir not found: {eval_repo_arg}")
            print("  Run: git submodule update --init --recursive")
            print("  Falling back to h5 consensus.")
            eval_repo_arg = None
        elif not check_eval_repo(eval_repo_arg, args.dataset):
            print(f"WARNING: eval repo found at {eval_repo_arg} but "
                  f"scorers/{args.dataset}/scorer_1/ does not exist.")
            print(f"  Available datasets may differ. Falling back to h5 consensus.")
            eval_repo_arg = None
        else:
            n_records = len([
                f for f in os.listdir(
                    os.path.join(eval_repo_arg, "scorers", args.dataset, "scorer_1")
                ) if f.endswith(".json")
            ])
            print(f"  Evaluation repo: {eval_repo_arg}")
            print(f"  Scorer data    : {N_SCORERS} scorers × {n_records} records "
                  f"for '{args.dataset}'")

    print_config("compute_sol_targets.py", {
        "dataset":             args.dataset,
        "h5_dir":              resolved_h5_dir,
        "eval_repo_dir":       eval_repo_arg or "(disabled — h5 consensus only)",
        "require_consecutive": args.require_consecutive,
        "out":                 resolved_out,
    })

    if not os.path.isdir(resolved_h5_dir):
        print(f"ERROR: h5_dir not found: {resolved_h5_dir}")
        print("  Check BASE_DIRECTORY in settings.py or pass --h5_dir explicitly.")
        sys.exit(1)

    h5_files = sorted([
        os.path.join(resolved_h5_dir, f)
        for f in os.listdir(resolved_h5_dir) if f.endswith(".h5")
    ])
    if not h5_files:
        print(f"ERROR: No .h5 files found in {resolved_h5_dir}")
        sys.exit(1)
    print(f"\nProcessing {len(h5_files)} recording(s) ...\n")

    if args.inspect_first:
        inspect_h5_structure(h5_files[0])

    records_data: List[Dict] = []
    for h5_path in h5_files:
        print(f"  {os.path.basename(h5_path):<40}", end=" ", flush=True)
        r = process_record(
            h5_path, eval_repo_arg, args.dataset,
            args.require_consecutive, EPOCH_DURATION_S,
        )
        if r["consensus_sol_min"] is not None:
            sols = list(r["scorer_sols_min"].values())
            sol_str = " | ".join(
                f"{v:.1f}" if v is not None else "—" for v in sols
            )
            std_str = (f"  SD={r['interrater_std_min']:.2f}"
                       if r["interrater_std_min"] is not None else "")
            print(f"mean={r['consensus_sol_min']:.1f}min  [{sol_str}]{std_str}")
        else:
            print("NO SLEEP" + (f"  [!] {r['warning'][:50]}" if r.get("warning") else ""))
        records_data.append(r)

    print_summary(records_data)

    os.makedirs(os.path.dirname(os.path.abspath(resolved_out)), exist_ok=True)
    out_payload = {r["record_id"]: r for r in records_data}
    out_payload["_meta"] = {
        "description": (
            "Expert-derived SOL reference per record (dataset-wide). "
            "Use with LOSO fold indices from pretraining when evaluating models."
        ),
        "dataset": args.dataset,
        "require_consecutive": args.require_consecutive,
        "n_records": len(records_data),
    }
    with open(resolved_out, "w") as f:
        json.dump(out_payload, f, indent=2)
    print(f"Saved → {resolved_out}\n")


if __name__ == "__main__":
    main(build_parser().parse_args())
