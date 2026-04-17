"""
evaluate_sol.py  —  Script 3
==============================
Evaluate any **pretrained** staging model's **LOOCV** outputs on **SOL estimation**
(vs expert-derived targets from ``compute_sol_targets.py``). Staging was trained
on **consensus** labels; this step scores **SOL** against per-scorer / consensus
reference, **per held-out fold** (same leave-one-subject-out split as pretraining).

Works on ``hypnograms.json`` under each run UUID in ``EXPERIMENTS_DIRECTORY``.

Minimal usage:
    python sol_experiments/evaluate_sol.py

Custom usage:
    python sol_experiments/evaluate_sol.py \\
        --model simple_sleep_net \\
        --dataset dodh \\
        --exp_dir /custom/experiment/dir/ \\
        --base-experiments-dir scripts/base_experiments

**Outputs** (under ``BASE_DIRECTORY/sol/evaluations/<dataset>/<model>/``):
  - ``fold_XX/sol_eval.json`` — metrics for that LOSO fold
  - ``summary.json`` — rollup across folds (unless ``--out`` overrides summary path)

Available ``--model`` values are folder names under ``EXPERIMENTS_DIRECTORY/<dataset>/``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dreem_learning_open.settings import REPO_ROOT
from dreem_learning_open.utils.experiment_fold_index import (
    build_loov_fold_map,
    load_memmap_description,
    recover_test_record_and_fold_idx,
)
from sol_experiments.sol_config import (
    DATASET_SETTINGS,
    EVALUATE_DEFAULTS,
    exp_dir as default_exp_dir,
    print_config,
    sol_eval_fold_dir,
    sol_eval_summary_path,
    sol_targets_path as default_targets_path,
    to_base_directory_relative,
)
from sol_experiments.utils.sol_metrics import (
    compute_sol_metrics,
    extract_consensus_sol,
    load_sol_targets,
    sol_from_hypnograms_json,
)


def find_hypnogram_files(exp_dir: str) -> List[str]:
    """Recursively find all hypnograms.json under exp_dir (one per LOOCV fold)."""
    found = []
    for root, _, files in os.walk(exp_dir):
        if "hypnograms.json" in files:
            found.append(os.path.join(root, "hypnograms.json"))
    return sorted(found)


def n1_f1_from_hypnograms(hyp_path: str) -> Dict[str, Optional[float]]:
    with open(hyp_path) as f:
        data = json.load(f)
    results = {}
    for rid, hyps in data.items():
        pred = np.array(hyps["predicted"], dtype=int)
        true = np.array(hyps["target"], dtype=int)
        mask = true >= 0
        if mask.sum() == 0:
            results[rid] = None
            continue
        try:
            f1 = f1_score(
                true[mask],
                pred[mask],
                labels=[1],
                average="macro",
                zero_division=0,
            )
            results[rid] = round(float(f1), 4)
        except Exception:
            results[rid] = None
    return results


def mean_staging_accuracy(hyp_path: str) -> Optional[float]:
    with open(hyp_path) as f:
        data = json.load(f)
    accs = []
    for hyps in data.values():
        pred = np.array(hyps["predicted"], dtype=int)
        true = np.array(hyps["target"], dtype=int)
        mask = true >= 0
        if mask.sum():
            accs.append((pred[mask] == true[mask]).mean())
    return float(np.mean(accs)) if accs else None


def _expert_interrater_summary(sol_targets: Dict) -> Optional[Dict[str, float]]:
    """Mean inter-rater SD (minutes) when present in target records."""
    stds = []
    for rec in sol_targets.values():
        if not isinstance(rec, dict):
            continue
        v = rec.get("interrater_std_min")
        if v is not None and isinstance(v, (int, float)) and not np.isnan(v):
            stds.append(float(v))
    if not stds:
        return None
    return {
        "mean_interrater_std_min": float(np.mean(stds)),
        "n_records_with_std": len(stds),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "LOSOCV SOL evaluation: model hypnograms vs expert SOL reference, "
            "one artifact tree per fold under BASE_DIRECTORY/sol/evaluations/."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model",
        default=EVALUATE_DEFAULTS["model"],
        help="Experiment folder name under EXPERIMENTS_DIRECTORY/<dataset>/.",
    )
    p.add_argument(
        "--dataset",
        default=EVALUATE_DEFAULTS["dataset"],
        choices=list(DATASET_SETTINGS.keys()),
    )
    p.add_argument(
        "--exp_dir",
        default=None,
        help="Pretrained experiment root (UUID run dirs). "
        "Default: EXPERIMENTS_DIRECTORY/<dataset>/<model>/",
    )
    p.add_argument(
        "--base-experiments-dir",
        default=os.path.join(REPO_ROOT, "scripts", "base_experiments"),
        help="Contains <model>/memmaps.json for LOOCV fold index (same as pretraining).",
    )
    p.add_argument(
        "--sol_targets",
        default=None,
        help="Expert SOL JSON. Default: BASE_DIRECTORY/sol/targets/<dataset>/sol_targets.json",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Optional path for rollup JSON (default: .../evaluations/.../summary.json). "
        "Per-fold files are always written under fold_XX/.",
    )
    p.add_argument(
        "--require_consecutive",
        type=int,
        default=EVALUATE_DEFAULTS["require_consecutive"],
        help="Min consecutive non-wake epochs to confirm SOL.",
    )
    return p


def main(args: argparse.Namespace) -> None:
    resolved_exp_dir = args.exp_dir or default_exp_dir(args.dataset, args.model)
    resolved_targets = args.sol_targets or default_targets_path(args.dataset)
    summary_out = args.out or sol_eval_summary_path(args.dataset, args.model)

    memmap_desc = load_memmap_description(
        args.base_experiments_dir, args.model, args.dataset
    )
    fold_map = build_loov_fold_map(DATASET_SETTINGS[args.dataset], memmap_desc)

    print_config(
        "evaluate_sol.py",
        {
            "model": args.model,
            "dataset": args.dataset,
            "exp_dir": resolved_exp_dir,
            "base_experiments_dir": args.base_experiments_dir,
            "sol_targets": resolved_targets,
            "summary_out": summary_out,
            "require_consecutive": args.require_consecutive,
        },
    )

    if not os.path.isdir(resolved_exp_dir):
        print("ERROR: exp_dir not found: {}".format(resolved_exp_dir))
        print("  Run scripts/run_cnn_rnn.py or run_base_experiments.py first.")
        sys.exit(1)
    if not os.path.exists(resolved_targets):
        print("ERROR: sol_targets not found: {}".format(resolved_targets))
        print("  Run compute_sol_targets.py first.")
        sys.exit(1)

    sol_targets = load_sol_targets(resolved_targets)
    ref_sols = extract_consensus_sol(sol_targets)
    print("Loaded {} consensus SOL targets.".format(len(ref_sols)))

    expert_hint = _expert_interrater_summary(sol_targets)
    if expert_hint:
        print(
            "Expert spread (mean inter-rater SD): {:.2f} min over {} records".format(
                expert_hint["mean_interrater_std_min"],
                expert_hint["n_records_with_std"],
            )
        )

    hyp_files = find_hypnogram_files(resolved_exp_dir)
    if not hyp_files:
        print("ERROR: No hypnograms.json found under {}".format(resolved_exp_dir))
        sys.exit(1)
    print("Found {} hypnograms.json file(s).\n".format(len(hyp_files)))

    all_pred_sols: Dict[str, Optional[float]] = {}
    all_n1_f1: Dict[str, Optional[float]] = {}
    staging_accs: List[float] = []
    per_fold_meta: List[Dict] = []

    for order_idx, hyp_path in enumerate(hyp_files):
        run_dir = os.path.dirname(hyp_path)
        fold_idx: Optional[int] = None
        test_record: Optional[str] = None
        desc_path = os.path.join(run_dir, "description.json")
        if os.path.isfile(desc_path):
            try:
                with open(desc_path, "r", encoding="utf-8") as f:
                    desc = json.load(f)
                test_record, fold_idx = recover_test_record_and_fold_idx(desc, fold_map)
            except Exception:
                pass
        if fold_idx is None:
            fold_idx = order_idx
            print(
                "WARNING: could not resolve fold_idx from {}; using order index {}".format(
                    run_dir,
                    fold_idx,
                )
            )

        pred_sols, _ = sol_from_hypnograms_json(
            hyp_path, require_consecutive=args.require_consecutive
        )
        n1_f1 = n1_f1_from_hypnograms(hyp_path)
        acc = mean_staging_accuracy(hyp_path)
        fold_metrics = compute_sol_metrics(pred_sols, ref_sols)

        fold_dir = sol_eval_fold_dir(args.dataset, args.model, fold_idx)
        fold_payload = {
            "model": args.model,
            "dataset": args.dataset,
            "fold_idx": fold_idx,
            "test_record": test_record,
            "run_dir": to_base_directory_relative(run_dir),
            "hypnograms_json": to_base_directory_relative(hyp_path),
            "require_consecutive": args.require_consecutive,
            "sol_metrics": fold_metrics,
            "n1_f1_per_record": n1_f1,
            "mean_staging_accuracy": acc,
        }
        fold_json = os.path.join(fold_dir, "sol_eval.json")
        with open(fold_json, "w", encoding="utf-8") as f:
            json.dump(fold_payload, f, indent=2)
        print("  Wrote {}".format(fold_json))

        all_pred_sols.update(pred_sols)
        all_n1_f1.update(n1_f1)
        if acc is not None:
            staging_accs.append(acc)
        per_fold_meta.append(
            {
                "fold_idx": fold_idx,
                "test_record": test_record,
                "path": to_base_directory_relative(fold_json),
            }
        )

    metrics = compute_sol_metrics(all_pred_sols, ref_sols)

    print("\n{}".format("=" * 65))
    print("  SOL EVALUATION (LOSOCV rollup)  —  {}  ({})".format(
        args.model.upper(), args.dataset.upper()
    ))
    print("{}".format("=" * 65))
    n = metrics.get("n_valid", 0)
    print("  Records with valid SOL pair : {}".format(n))
    if "mae_min" in metrics:
        print("\n  MAE          : {:.2f} min".format(metrics["mae_min"]))
        print("  RMSE         : {:.2f} min".format(metrics["rmse_min"]))
        direction = "over" if metrics["bias_min"] > 0 else "under"
        print(
            "  Bias         : {:+.2f} min  ({}-estimate)".format(
                metrics["bias_min"], direction
            )
        )
        print("  SD of errors : {:.2f} min".format(metrics["std_err_min"]))
        print("  Pearson r    : {:.3f}".format(metrics["pearson_r"]))

    valid_n1 = [v for v in all_n1_f1.values() if v is not None]
    if valid_n1:
        print(
            "\n  N1 F1        : {:.3f} ± {:.3f}".format(
                np.mean(valid_n1), np.std(valid_n1)
            )
        )

    if staging_accs:
        print("\n  Staging accuracy (mean) : {:.3f}".format(np.mean(staging_accs)))

    print("{}\n".format("=" * 65))

    summary_payload = {
        "model": args.model,
        "dataset": args.dataset,
        "exp_dir": to_base_directory_relative(resolved_exp_dir),
        "sol_targets": to_base_directory_relative(resolved_targets),
        "require_consecutive": args.require_consecutive,
        "expert_interrater_summary": expert_hint,
        "sol_metrics_rollup": metrics,
        "n1_f1_per_record": all_n1_f1,
        "mean_staging_accuracy": float(np.mean(staging_accs)) if staging_accs else None,
        "folds": per_fold_meta,
    }
    os.makedirs(os.path.dirname(os.path.abspath(summary_out)), exist_ok=True)
    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)
    print("Saved rollup → {}\n".format(summary_out))


if __name__ == "__main__":
    main(build_parser().parse_args())
