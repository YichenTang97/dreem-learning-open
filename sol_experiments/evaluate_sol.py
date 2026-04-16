"""
evaluate_sol.py  —  Script 3
==============================
Evaluate any trained staging model's LOOCV outputs on SOL estimation.
Works on the hypnograms.json files produced by run_base_experiments.py
(SimpleSleepNet etc.) and train_cnn_rnn.py.

Minimal usage (evaluates CNN-RNN on DODH):
    python sol_experiments/evaluate_sol.py

Custom usage:
    python sol_experiments/evaluate_sol.py \\
        --model simple_sleep_net \\
        --dataset dodh \\
        --exp_dir /custom/experiment/dir/ \\
        --sol_targets /custom/sol_targets.json \\
        --out /custom/results.json \\
        --require_consecutive 2

Available model names (matching the experiment folder names):
    cnn_rnn              ← our proposed model (default)
    simple_sleep_net     ← SimpleSleepNet baseline (Guillot et al. 2020)
    chambon_et_al        ← Chambon et al. baseline
    deep_sleep_net       ← DeepSleepNet baseline
    tsinalis_et_al       ← Tsinalis et al. baseline
    (any folder name under EXPERIMENTS_DIRECTORY/<dataset>/)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sol_experiments.sol_config import (
    DATASET_SETTINGS,
    EVALUATE_DEFAULTS,
    exp_dir as default_exp_dir,
    sol_targets_path as default_targets_path,
    sol_results_path as default_results_path,
    print_config,
)
from sol_experiments.utils.sol_metrics import (
    compute_sol,
    compute_sol_metrics,
    load_sol_targets,
    extract_consensus_sol,
    sol_from_hypnograms_json,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_hypnogram_files(exp_dir: str) -> List[str]:
    """Recursively find all hypnograms.json under exp_dir (one per LOOCV fold)."""
    found = []
    for root, _, files in os.walk(exp_dir):
        if "hypnograms.json" in files:
            found.append(os.path.join(root, "hypnograms.json"))
    return sorted(found)


def n1_f1_from_hypnograms(hyp_path: str) -> Dict[str, Optional[float]]:
    """Per-record N1 F1-score from a hypnograms.json file."""
    with open(hyp_path) as f:
        data = json.load(f)
    results = {}
    for rid, hyps in data.items():
        pred = np.array(hyps["predicted"], dtype=int)
        true = np.array(hyps["target"],    dtype=int)
        mask = true >= 0
        if mask.sum() == 0:
            results[rid] = None
            continue
        try:
            f1 = f1_score(true[mask], pred[mask], labels=[1],
                          average="macro", zero_division=0)
            results[rid] = round(float(f1), 4)
        except Exception:
            results[rid] = None
    return results


def mean_staging_accuracy(hyp_path: str) -> Optional[float]:
    """Mean per-record staging accuracy from a hypnograms.json file."""
    with open(hyp_path) as f:
        data = json.load(f)
    accs = []
    for hyps in data.values():
        pred = np.array(hyps["predicted"], dtype=int)
        true = np.array(hyps["target"],    dtype=int)
        mask = true >= 0
        if mask.sum():
            accs.append((pred[mask] == true[mask]).mean())
    return float(np.mean(accs)) if accs else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate a trained sleep staging model on SOL estimation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model", default=EVALUATE_DEFAULTS["model"],
        help="Model name (= experiment sub-folder under EXPERIMENTS_DIRECTORY/<dataset>/).",
    )
    p.add_argument(
        "--dataset", default=EVALUATE_DEFAULTS["dataset"],
        choices=list(DATASET_SETTINGS.keys()),
        help="Dataset the model was trained on.",
    )
    p.add_argument(
        "--exp_dir", default=None,
        help="Path to the model's experiment folder containing fold sub-dirs. "
             "Default: EXPERIMENTS_DIRECTORY/<dataset>/<model>/",
    )
    p.add_argument(
        "--sol_targets", default=None,
        help="Path to sol_targets.json from compute_sol_targets.py. "
             "Default: sol_experiments/data/sol_targets_<dataset>.json",
    )
    p.add_argument(
        "--out", default=None,
        help="Output JSON path for SOL metrics. "
             "Default: sol_experiments/results/sol_<model>_<dataset>.json",
    )
    p.add_argument(
        "--require_consecutive", type=int,
        default=EVALUATE_DEFAULTS["require_consecutive"],
        help="Min consecutive non-wake epochs to confirm SOL.",
    )
    return p


def main(args: argparse.Namespace) -> None:
    resolved_exp_dir  = args.exp_dir     or default_exp_dir(args.dataset, args.model)
    resolved_targets  = args.sol_targets or default_targets_path(args.dataset)
    resolved_out      = args.out         or default_results_path(args.dataset, args.model)

    print_config("evaluate_sol.py", {
        "model":               args.model,
        "dataset":             args.dataset,
        "exp_dir":             resolved_exp_dir,
        "sol_targets":         resolved_targets,
        "out":                 resolved_out,
        "require_consecutive": args.require_consecutive,
    })

    # ---- Validate inputs ----
    if not os.path.isdir(resolved_exp_dir):
        print(f"ERROR: exp_dir not found: {resolved_exp_dir}")
        print("  Run train_cnn_rnn.py (or run_base_experiments.py) first.")
        sys.exit(1)
    if not os.path.exists(resolved_targets):
        print(f"ERROR: sol_targets not found: {resolved_targets}")
        print("  Run compute_sol_targets.py first.")
        sys.exit(1)

    # ---- Load reference SOLs ----
    sol_targets = load_sol_targets(resolved_targets)
    ref_sols    = extract_consensus_sol(sol_targets)
    print(f"Loaded {len(ref_sols)} consensus SOL targets.")

    # ---- Gather hypnogram files ----
    hyp_files = find_hypnogram_files(resolved_exp_dir)
    if not hyp_files:
        print(f"ERROR: No hypnograms.json found under {resolved_exp_dir}")
        sys.exit(1)
    print(f"Found {len(hyp_files)} hypnograms.json file(s).\n")

    # ---- Collect predictions across all folds ----
    all_pred_sols: Dict[str, Optional[float]] = {}
    all_n1_f1:     Dict[str, Optional[float]] = {}
    staging_accs:  List[float] = []

    for hyp_path in hyp_files:
        pred_sols, _ = sol_from_hypnograms_json(
            hyp_path, require_consecutive=args.require_consecutive
        )
        all_pred_sols.update(pred_sols)
        all_n1_f1.update(n1_f1_from_hypnograms(hyp_path))
        acc = mean_staging_accuracy(hyp_path)
        if acc is not None:
            staging_accs.append(acc)

    # ---- SOL metrics ----
    metrics = compute_sol_metrics(all_pred_sols, ref_sols)

    # ---- Print report ----
    print(f"\n{'='*65}")
    print(f"  SOL EVALUATION  —  {args.model.upper()}  ({args.dataset.upper()})")
    print(f"{'='*65}")
    n = metrics.get("n_valid", 0)
    print(f"  Records with valid SOL pair : {n}")
    if "mae_min" in metrics:
        print(f"\n  MAE          : {metrics['mae_min']:.2f} min")
        print(f"  RMSE         : {metrics['rmse_min']:.2f} min")
        direction = "over" if metrics["bias_min"] > 0 else "under"
        print(f"  Bias         : {metrics['bias_min']:+.2f} min  ({direction}-estimate)")
        print(f"  SD of errors : {metrics['std_err_min']:.2f} min")
        print(f"  Pearson r    : {metrics['pearson_r']:.3f}")

    valid_n1 = [v for v in all_n1_f1.values() if v is not None]
    if valid_n1:
        print(f"\n  N1 F1        : {np.mean(valid_n1):.3f} ± {np.std(valid_n1):.3f}")
        print(f"  (Low N1 F1 → SOL errors driven by N1↔Wake mis-classification)")

    if staging_accs:
        print(f"\n  Staging accuracy (mean) : {np.mean(staging_accs):.3f}")

    pr = metrics.get("per_record", {})
    if pr:
        print(f"\n  Per-record results (sorted by |error|):")
        print(f"  {'Record':<20} {'Pred':>8} {'Ref':>8} {'Error':>10} {'N1 F1':>8}")
        print(f"  {'-'*57}")
        for rid, v in sorted(pr.items(),
                              key=lambda kv: (kv[1]["abs_error_min"] or 999)):
            pred_s = f"{v['predicted_min']:.1f}" if v["predicted_min"] is not None else "—"
            ref_s  = f"{v['reference_min']:.1f}"  if v["reference_min"]  is not None else "—"
            err_s  = f"{v['error_min']:+.1f}"     if v["error_min"]      is not None else "—"
            n1_s   = f"{all_n1_f1.get(rid):.3f}"  if all_n1_f1.get(rid) is not None else "—"
            print(f"  {rid:<20} {pred_s:>8} {ref_s:>8} {err_s:>10} {n1_s:>8}")
    print(f"{'='*65}\n")

    # ---- Save ----
    output = {
        "model":                args.model,
        "dataset":              args.dataset,
        "exp_dir":              resolved_exp_dir,
        "sol_targets":          resolved_targets,
        "require_consecutive":  args.require_consecutive,
        "sol_metrics":          metrics,
        "n1_f1_per_record":     all_n1_f1,
        "mean_staging_accuracy": float(np.mean(staging_accs)) if staging_accs else None,
    }
    os.makedirs(os.path.dirname(os.path.abspath(resolved_out)), exist_ok=True)
    with open(resolved_out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved → {resolved_out}\n")


if __name__ == "__main__":
    main(build_parser().parse_args())
