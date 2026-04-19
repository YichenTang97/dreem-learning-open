"""
evaluate_sol.py  —  Script 3
==============================
Evaluate any **pretrained** staging model's **LOOCV** outputs on **SOL estimation**
(vs expert-derived targets from ``compute_sol_targets.py``). Staging was trained
on **consensus** labels; this step scores **SOL** against per-scorer / consensus
reference, **per held-out fold** (same leave-one-subject-out split as pretraining).

Works on ``hypnograms.json`` under each run UUID in ``EXPERIMENTS_DIRECTORY``.

Minimal usage (pretrained staging runs under ``EXPERIMENTS_DIRECTORY``):
    python sol_experiments/evaluate_sol.py

Finetuned models under the default layout (``sol/finetuned/<dataset>/<subfolder>/``):
    python sol_experiments/evaluate_sol.py --dataset dodh \\
        --model simple_sleep_net_ft_c10m_a0.50 --finetuned

    ``--model`` is the finetuned subfolder name; ``--exp_dir`` defaults to that path.
    Override ``--exp_dir`` if the run lives elsewhere.

Custom usage:
    python sol_experiments/evaluate_sol.py \\
        --model simple_sleep_net \\
        --dataset dodh \\
        --exp_dir /custom/experiment/dir/ \\
        --base-experiments-dir scripts/base_experiments

**Outputs** (under ``BASE_DIRECTORY/sol/evaluations/<dataset>/<name>/``; ``<name>`` is
``--model`` for pretrained runs, or the finetuned folder name when ``--finetuned``):
  - ``fold_XX/sol_eval.json`` — metrics for that LOSO fold
  - ``summary.json`` — rollup across folds (unless ``--out`` overrides summary path)

Available ``--model`` values are folder names under ``EXPERIMENTS_DIRECTORY/<dataset>/``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
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
    finetuned_run_dir,
    normalize_base_model_for_finetune_tag,
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


def _resolve_record_dir(record_key: str) -> Optional[str]:
    """Resolve a hypnograms.json record key to an on-disk memmap record directory."""
    candidates = []
    if os.path.isabs(record_key):
        candidates.append(record_key)
    else:
        candidates.append(os.path.join(REPO_ROOT, record_key))
        if record_key.startswith("data/"):
            candidates.append(os.path.join(REPO_ROOT, record_key.replace("data/", "", 1)))

    for c in candidates:
        if os.path.isdir(c):
            return c
    return None


def _align_target_to_prediction(pred: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align target and prediction lengths.
    Base runs often trim padding at both ends before saving predictions.
    """
    pred = np.asarray(pred, dtype=int)
    target = np.asarray(target, dtype=int)
    if len(target) > len(pred):
        diff = len(target) - len(pred)
        left = diff // 2
        right = diff - left
        target = target[left: len(target) - right]
    n = min(len(pred), len(target))
    return pred[:n], target[:n]


UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)

def _staging_model_name(model_arg: str, finetuned: bool) -> str:
    """
    Folder name under EXPERIMENTS_DIRECTORY / base_experiments for memmaps and fold metadata.
    If --finetuned and model_arg is the finetuned subfolder name, strip ``_ft_c…m_a…`` suffix(es).
    """
    if not finetuned:
        return model_arg
    return normalize_base_model_for_finetune_tag(model_arg)


def _extract_record_id(key: str) -> Optional[str]:
    """
    Best-effort extraction of a record UUID from a record key/path.
    """
    if not key:
        return None
    normalized = key.replace("\\", "/").rstrip("/")
    parts = [p for p in normalized.split("/") if p]
    for token in reversed(parts):
        if UUID_RE.match(token):
            return token
    if UUID_RE.match(normalized):
        return normalized
    return None


def _normalize_per_record(
    per_record: Dict[str, Optional[float]],
    test_record: Optional[str] = None,
) -> Dict[str, Optional[float]]:
    """
    Normalize model record keys to canonical UUIDs used by SOL targets.
    """
    normalized: Dict[str, Optional[float]] = {}
    fallback_values: List[Optional[float]] = []
    for key, value in per_record.items():
        rid = _extract_record_id(key)
        if rid is not None:
            normalized[rid] = value
        else:
            fallback_values.append(value)

    if test_record and test_record not in normalized:
        if len(normalized) == 0 and len(fallback_values) == 1:
            normalized[test_record] = fallback_values[0]
        elif len(normalized) == 1:
            normalized[test_record] = next(iter(normalized.values()))
    return normalized


def _load_target_hypnogram(record_key: str) -> Optional[np.ndarray]:
    record_dir = _resolve_record_dir(record_key)
    if record_dir is None:
        return None
    hyp_path = os.path.join(record_dir, "hypno.mm")
    if not os.path.exists(hyp_path):
        return None
    try:
        return np.memmap(hyp_path, mode="r", dtype="float32").astype(int)
    except Exception:
        return None


def n1_f1_from_hypnograms(hyp_path: str) -> Dict[str, Optional[float]]:
    with open(hyp_path) as f:
        data = json.load(f)
    results = {}
    for rid, hyps in data.items():
        pred = np.array(hyps["predicted"], dtype=int) if isinstance(hyps, dict) else np.array(hyps, dtype=int)
        true = _load_target_hypnogram(rid)
        if true is None:
            results[rid] = None
            continue
        pred, true = _align_target_to_prediction(pred, true)
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
    for rid, hyps in data.items():
        pred = np.array(hyps["predicted"], dtype=int) if isinstance(hyps, dict) else np.array(hyps, dtype=int)
        true = _load_target_hypnogram(rid)
        if true is None:
            continue
        pred, true = _align_target_to_prediction(pred, true)
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


def _model_vs_stored_human_benchmark(
    sol_targets: Dict,
    model_pred_sols: Dict[str, Optional[float]],
) -> Optional[Dict]:
    """
    Read stored human scorer benchmark from sol_targets and append model comparison.
    """
    stored = sol_targets.get("_scorer_vs_mean_benchmark")
    if not isinstance(stored, dict):
        return None
    mean_ref = stored.get("per_record_mean_sol_min")
    if not isinstance(mean_ref, dict):
        return None
    model_vs_human_mean = compute_sol_metrics(model_pred_sols, mean_ref)
    return {
        **stored,
        "model_vs_human_mean": model_vs_human_mean,
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
        help=(
            "Without --finetuned: pretrained folder under EXPERIMENTS_DIRECTORY/<dataset>/ "
            "(memmaps / fold index). With --finetuned: subfolder name under "
            "sol/finetuned/<dataset>/ (e.g. simple_sleep_net_ft_c10m_a0.50); the leading "
            "base name before _ft_c… is used for memmaps. Use --exp_dir to override location."
        ),
    )
    p.add_argument(
        "--dataset",
        default=EVALUATE_DEFAULTS["dataset"],
        choices=list(DATASET_SETTINGS.keys()),
    )
    p.add_argument(
        "--finetuned",
        action="store_true",
        help=(
            "Hypnograms from SOL fine-tuning (fold_XX/hypnograms.json). Default exp_dir is "
            "sol/finetuned/<dataset>/<model>/ where --model is the finetuned subfolder "
            "(same layout as finetune_sol default --out_dir). Pass --exp_dir to use another path."
        ),
    )
    p.add_argument(
        "--eval_model",
        default=None,
        metavar="NAME",
        help=(
            "Folder name under sol/evaluations/<dataset>/ for this run. "
            "Default: --model for pretrained; for --finetuned, the basename of --exp_dir."
        ),
    )
    p.add_argument(
        "--exp_dir",
        default=None,
        help=(
            "Root directory containing fold_XX/ trees with hypnograms.json. "
            "Default without --finetuned: EXPERIMENTS_DIRECTORY/<dataset>/<model>/. "
            "With --finetuned: default is sol/finetuned/<dataset>/<model>/ (see sol_config)."
        ),
    )
    p.add_argument(
        "--base-experiments-dir",
        default=os.path.join(REPO_ROOT, "scripts", "base_experiments"),
        help="Contains <model>/memmaps.json for LOOCV fold index (same as pretraining).",
    )
    p.add_argument(
        "--sol_targets",
        default=None,
        help="SOL targets JSON (detailed or consensus-only). "
        "Default: BASE_DIRECTORY/sol/targets/<dataset>/sol_targets.json",
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
    staging_model = _staging_model_name(args.model, args.finetuned)
    if args.finetuned:
        resolved_exp_dir = args.exp_dir or finetuned_run_dir(args.dataset, args.model)
    else:
        resolved_exp_dir = args.exp_dir or default_exp_dir(args.dataset, args.model)

    if args.eval_model:
        eval_model_name = args.eval_model
    elif args.finetuned:
        eval_model_name = os.path.basename(os.path.normpath(resolved_exp_dir))
    else:
        eval_model_name = args.model

    resolved_targets = args.sol_targets or default_targets_path(args.dataset)
    summary_out = args.out or sol_eval_summary_path(args.dataset, eval_model_name)

    memmap_desc = load_memmap_description(
        args.base_experiments_dir, staging_model, args.dataset
    )
    fold_map = build_loov_fold_map(DATASET_SETTINGS[args.dataset], memmap_desc)

    print_config(
        "evaluate_sol.py",
        {
            "model (CLI)": args.model,
            "staging_model (memmaps)": staging_model,
            "eval_model (outputs)": eval_model_name,
            "dataset": args.dataset,
            "finetuned": args.finetuned,
            "exp_dir": resolved_exp_dir,
            "base_experiments_dir": args.base_experiments_dir,
            "sol_targets": resolved_targets,
            "summary_out": summary_out,
            "require_consecutive": args.require_consecutive,
        },
    )

    if not os.path.isdir(resolved_exp_dir):
        print("ERROR: exp_dir not found: {}".format(resolved_exp_dir))
        if args.finetuned:
            print(
                "  Check --model matches the finetuned subfolder under sol/finetuned/<dataset>/, "
                "or pass --exp_dir explicitly."
            )
        else:
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

        pred_sols_raw, _ = sol_from_hypnograms_json(
            hyp_path, require_consecutive=args.require_consecutive
        )
        pred_sols = _normalize_per_record(pred_sols_raw, test_record=test_record)
        n1_f1 = _normalize_per_record(n1_f1_from_hypnograms(hyp_path), test_record=test_record)
        acc = mean_staging_accuracy(hyp_path)
        fold_metrics = compute_sol_metrics(pred_sols, ref_sols)

        fold_dir = sol_eval_fold_dir(args.dataset, eval_model_name, fold_idx)
        fold_payload = {
            "model": eval_model_name,
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
    scorer_benchmark = _model_vs_stored_human_benchmark(sol_targets, all_pred_sols)

    print("\n{}".format("=" * 65))
    print("  SOL EVALUATION (LOSOCV rollup)  —  {}  ({})".format(
        eval_model_name.upper(), args.dataset.upper()
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

    if scorer_benchmark and scorer_benchmark.get("model_vs_human_mean", {}).get("mae_min") is not None:
        mv = scorer_benchmark["model_vs_human_mean"]
        # Default human baseline: LOO reference (same framework as per-scorer human benchmark).
        ha = scorer_benchmark.get("human_aggregate_vs_loo_mean") or {}
        print("\n  Side-by-side benchmark")
        print("  Model: vs MeanAll.  Human: vs leave-one-out mean (mean±std across scorers).")
        print("  {:<14} {:>12}  {:>22}".format("Metric", "Model", "Human (LOO)"))
        for key, label in (
            ("mae_min", "MAE (min)"),
            ("rmse_min", "RMSE (min)"),
            ("bias_min", "Bias (min)"),
            ("std_err_min", "SD(err) (min)"),
            ("pearson_r", "Pearson r"),
        ):
            mv_val = mv.get(key)
            hm = ha.get("mean_{}".format(key))
            hs = ha.get("std_{}".format(key))
            if key == "pearson_r":
                m_str = "{:.3f}".format(mv_val) if mv_val is not None else "—"
                if hm is not None and hs is not None:
                    h_str = "{:.3f} ± {:.3f}".format(hm, hs)
                elif hm is not None:
                    h_str = "{:.3f}".format(hm)
                else:
                    h_str = "—"
            else:
                m_str = "{:.2f}".format(mv_val) if mv_val is not None else "—"
                if hm is not None and hs is not None:
                    h_str = "{:.2f} ± {:.2f}".format(hm, hs)
                elif hm is not None:
                    h_str = "{:.2f}".format(hm)
                else:
                    h_str = "—"
            print("  {:<14} {:>12}  {:>22}".format(label, m_str, h_str))

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
        "model": eval_model_name,
        "base_model": staging_model,
        "model_arg": args.model,
        "finetuned": bool(args.finetuned),
        "dataset": args.dataset,
        "exp_dir": to_base_directory_relative(resolved_exp_dir),
        "sol_targets": to_base_directory_relative(resolved_targets),
        "require_consecutive": args.require_consecutive,
        "expert_interrater_summary": expert_hint,
        "scorer_vs_mean_benchmark": scorer_benchmark,
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
