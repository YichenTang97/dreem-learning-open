"""
compute_sol_targets.py  —  Script 1
=====================================
Load every .h5 file from a Dreem dataset directory, extract per-scorer
hypnograms, compute individual and consensus Sleep Onset Latency (SOL),
and save the results for downstream use.

Consensus SOL definition (as per proposal):
    consensus_SOL = mean( SOL_scorer_1, ..., SOL_scorer_N )

Minimal usage (all paths resolved from settings.py):
    python sol_experiments/compute_sol_targets.py

Custom usage:
    python sol_experiments/compute_sol_targets.py \\
        --dataset dodo \\
        --h5_dir  /my/custom/h5/path/ \\
        --out     /my/custom/output.json \\
        --require_consecutive 2 \\
        --inspect_first
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import h5py
import numpy as np

# Resolve imports regardless of CWD
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sol_experiments.sol_config import (
    h5_dir as default_h5_dir,
    sol_targets_path as default_targets_path,
    TARGETS_DEFAULTS,
    print_config,
    DATASET_SETTINGS,
)
from sol_experiments.utils.sol_metrics import compute_sol, EPOCH_DURATION_S


# ---------------------------------------------------------------------------
# h5 introspection helpers
# ---------------------------------------------------------------------------

def list_all_h5_keys(h5file: h5py.File) -> List[str]:
    """Recursively collect every dataset path in an h5 file."""
    keys: List[str] = []
    h5file.visititems(lambda name, obj: keys.append(name)
                      if isinstance(obj, h5py.Dataset) else None)
    return keys


def inspect_h5_structure(h5_path: str) -> None:
    """Pretty-print dataset keys and shapes for a single h5 file."""
    print(f"\n{'='*60}")
    print(f"Inspecting: {os.path.basename(h5_path)}")
    print(f"{'='*60}")
    with h5py.File(h5_path, "r") as f:
        for k in list_all_h5_keys(f):
            try:
                print(f"  {k:<50}  shape={f[k].shape}  dtype={f[k].dtype}")
            except Exception as e:
                print(f"  {k:<50}  [error: {e}]")
    print()


def find_scorer_hypnograms(
        h5file: h5py.File,
        all_keys: List[str]
) -> tuple:
    """
    Try to locate individual-scorer hypnograms in the h5 file.

    Strategies (in priority order):
    1. Keys matching 'hypnogram_scorer_N' or similar per-scorer patterns.
    2. A 2-D 'hypnogram' array with shape (n_epochs, N_scorers).
    3. Fall back to the single 1-D consensus 'hypnogram' key.

    Returns (list_of_1d_arrays, list_of_key_names).
    """
    hyp_keys = [k for k in all_keys
                if "hypnogram" in k.lower() or "scoring" in k.lower()]

    # Strategy 1: per-scorer keys
    scorer_keys = sorted([
        k for k in hyp_keys
        if any(f"scorer_{i}" in k.lower() or f"scorer{i}" in k.lower()
               for i in range(1, 10))
    ])
    if scorer_keys:
        return [h5file[k][:].astype(int) for k in scorer_keys], scorer_keys

    # Strategy 2: any extra hypnogram-like 1-D key beyond the main one
    extra = sorted([k for k in hyp_keys if k != "hypnogram"])
    if extra:
        arrays = [h5file[k][:].astype(int) for k in extra if h5file[k][:].ndim == 1]
        if arrays:
            return arrays, extra

    # Strategy 3: 2-D hypnogram (n_epochs × n_scorers)
    if "hypnogram" in h5file:
        hyp = h5file["hypnogram"][:]
        if hyp.ndim == 2 and hyp.shape[1] > 1:
            return ([hyp[:, i].astype(int) for i in range(hyp.shape[1])],
                    [f"hypnogram_col_{i}" for i in range(hyp.shape[1])])

    # Strategy 4: single 1-D consensus (fallback)
    if "hypnogram" in h5file:
        hyp = h5file["hypnogram"][:]
        if hyp.ndim == 1:
            return [hyp.astype(int)], ["hypnogram"]

    return [], []


# ---------------------------------------------------------------------------
# Per-record processing
# ---------------------------------------------------------------------------

def process_record(
        h5_path: str,
        require_consecutive: int = 1,
        epoch_duration_s: int = EPOCH_DURATION_S,
) -> Dict:
    record_id = os.path.splitext(os.path.basename(h5_path))[0]
    result: Dict = {
        "record_id":          record_id,
        "h5_path":            h5_path,
        "n_scorers":          0,
        "scorer_keys_used":   [],
        "scorer_sols_min":    {},
        "consensus_sol_min":  None,
        "interrater_std_min": None,
        "n_scorers_with_sleep": 0,
        "warning":            None,
    }
    try:
        with h5py.File(h5_path, "r") as f:
            all_keys = list_all_h5_keys(f)
            scorer_arrays, scorer_keys = find_scorer_hypnograms(f, all_keys)

        if not scorer_arrays:
            result["warning"] = "No hypnogram found in h5 file."
            return result

        if len(scorer_arrays) == 1 and scorer_keys == ["hypnogram"]:
            result["warning"] = (
                "Only a single consensus 'hypnogram' key found — "
                "inter-rater statistics unavailable."
            )

        result["n_scorers"]        = len(scorer_arrays)
        result["scorer_keys_used"] = scorer_keys

        valid_sols: List[float] = []
        for key, arr in zip(scorer_keys, scorer_arrays):
            sol = compute_sol(arr, epoch_duration_s, require_consecutive)
            result["scorer_sols_min"][key] = round(sol, 3) if sol is not None else None
            if sol is not None:
                valid_sols.append(sol)

        result["n_scorers_with_sleep"] = len(valid_sols)
        if valid_sols:
            result["consensus_sol_min"]   = round(float(np.mean(valid_sols)), 3)
            result["interrater_std_min"]  = round(float(np.std(valid_sols, ddof=0)), 3)
        else:
            result["warning"] = (result["warning"] or "") + \
                                 " No scorer detected any sleep."

    except Exception as e:
        result["warning"] = f"Failed: {e}"

    return result


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(records_data: List[Dict]) -> None:
    valid     = [r for r in records_data if r["consensus_sol_min"] is not None]
    all_sols  = [r["consensus_sol_min"] for r in valid]
    warnings  = [r for r in records_data if r.get("warning")]
    n_scorers = records_data[0]["n_scorers"] if records_data else 0

    print(f"\n{'='*65}")
    print("SOL TARGET SUMMARY")
    print(f"{'='*65}")
    print(f"  Total records processed   : {len(records_data)}")
    print(f"  Records with valid SOL    : {len(valid)}")
    print(f"  Records with warnings     : {len(warnings)}")
    print(f"  Scorers per record        : {n_scorers} "
          f"({'individual' if n_scorers > 1 else 'consensus only'})")

    if all_sols:
        print(f"\n  Consensus SOL (minutes):")
        print(f"    Mean ± SD : {np.mean(all_sols):.2f} ± {np.std(all_sols):.2f}")
        print(f"    Median    : {np.median(all_sols):.2f}")
        print(f"    Min / Max : {np.min(all_sols):.2f} / {np.max(all_sols):.2f}")

    valid_std = [r["interrater_std_min"] for r in valid
                 if r["interrater_std_min"] is not None]
    if valid_std and n_scorers > 1:
        mean_std = np.mean(valid_std)
        print(f"\n  Inter-rater SOL variability (SD across scorers, minutes):")
        print(f"    Mean SD : {mean_std:.2f}  |  Max SD : {np.max(valid_std):.2f}")
        print(f"\n  ↳ Human ceiling: your model should aim for MAE < {mean_std:.1f} min")

    if warnings:
        print(f"\n  Warnings ({len(warnings)} records):")
        for r in warnings[:5]:
            print(f"    [{r['record_id']}] {r['warning']}")
        if len(warnings) > 5:
            print(f"    … and {len(warnings)-5} more.")
    print(f"{'='*65}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract per-scorer SOL targets from Dreem h5 files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dataset", default=TARGETS_DEFAULTS["dataset"],
        choices=list(DATASET_SETTINGS.keys()),
        help="Which dataset to process.",
    )
    p.add_argument(
        "--h5_dir", default=None,
        help="Directory containing .h5 files. Default: read from settings.py for dataset.",
    )
    p.add_argument(
        "--out", default=None,
        help="Output JSON path. Default: sol_experiments/data/sol_targets_<dataset>.json",
    )
    p.add_argument(
        "--require_consecutive", type=int,
        default=TARGETS_DEFAULTS["require_consecutive"],
        help="Min consecutive non-wake epochs to confirm SOL.",
    )
    p.add_argument(
        "--inspect_first", action="store_true",
        default=TARGETS_DEFAULTS["inspect_first"],
        help="Print the full key listing of the first h5 file (useful for debugging).",
    )
    return p


def main(args: argparse.Namespace) -> None:
    # Resolve defaults that depend on other args
    resolved_h5_dir  = args.h5_dir  or default_h5_dir(args.dataset)
    resolved_out     = args.out     or default_targets_path(args.dataset)

    print_config("compute_sol_targets.py", {
        "dataset":            args.dataset,
        "h5_dir":             resolved_h5_dir,
        "out":                resolved_out,
        "require_consecutive": args.require_consecutive,
        "inspect_first":      args.inspect_first,
    })

    if not os.path.isdir(resolved_h5_dir):
        print(f"ERROR: h5_dir not found: {resolved_h5_dir}")
        print("  Set the correct path in settings.py or pass --h5_dir explicitly.")
        sys.exit(1)

    h5_files = sorted([
        os.path.join(resolved_h5_dir, f)
        for f in os.listdir(resolved_h5_dir) if f.endswith(".h5")
    ])
    if not h5_files:
        print(f"ERROR: No .h5 files found in {resolved_h5_dir}")
        sys.exit(1)

    print(f"Found {len(h5_files)} h5 file(s).\n")

    if args.inspect_first:
        inspect_h5_structure(h5_files[0])

    records_data = []
    for h5_path in h5_files:
        print(f"  {os.path.basename(h5_path):<35}", end=" ", flush=True)
        r = process_record(h5_path, args.require_consecutive, EPOCH_DURATION_S)
        status = (f"SOL={r['consensus_sol_min']:.1f} min "
                  f"({r['n_scorers']} scorer{'s' if r['n_scorers'] != 1 else ''})"
                  if r["consensus_sol_min"] is not None else "NO SLEEP")
        if r.get("warning"):
            status += f"  [!]"
        print(status)
        records_data.append(r)

    print_summary(records_data)

    os.makedirs(os.path.dirname(os.path.abspath(resolved_out)), exist_ok=True)
    output = {r["record_id"]: r for r in records_data}
    with open(resolved_out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved → {resolved_out}\n")


if __name__ == "__main__":
    main(build_parser().parse_args())
