"""
Summarise LOSOCV pretraining performance (consensus hypnogram metrics) for a base model.

Per-fold metrics are read from each run's ``description.json`` under
``EXPERIMENTS_DIRECTORY/<dataset>/<model>/<run_uuid>/`` (with ``settings.py``, that is typically
``<BASE_DIRECTORY>/experiments/<dataset>/<model>/``, e.g. ``data/experiments/dodh/simple_sleep_net/``).

The summary JSON is written **in that same runs directory** (next to ``fold_summary.json`` /
``completed_folds.jsonl`` if you use ``index_experiments``). Override with ``--output-dir`` only
if you need a different path.

Each fold includes the full ``performance_on_test_set`` dict from the best run's
``description.json``. The JSON also contains ``aggregate_by_metric`` (mean, stdev, min, max)
for every metric that appears with a numeric value on **all** folds.

Exits with non-zero status if any fold is missing a completed run with the selection metric.

Utility modules are loaded by file path so this script does not import ``dreem_learning_open.utils``
(which pulls in numpy via the package ``__init__``).

Usage:
    python scripts/run_summary_pretrained.py --model simple_sleep_net
    python scripts/run_summary_pretrained.py --model cnn_rnn --dataset dodh --metric cohen_kappa
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dreem_learning_open.settings import DODO_SETTINGS, DODH_SETTINGS, EXPERIMENTS_DIRECTORY  # noqa: E402


def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load {!r} from {!r}".format(name, path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_PKG_UTILS = os.path.join(_REPO_ROOT, "dreem_learning_open", "utils")
_experiment_fold_index = _load_module(
    "_dreem_experiment_fold_index",
    os.path.join(_PKG_UTILS, "experiment_fold_index.py"),
)
_indexed_run_complete = _load_module(
    "_dreem_indexed_run_complete",
    os.path.join(_PKG_UTILS, "indexed_run_complete.py"),
)

build_loov_fold_map = _experiment_fold_index.build_loov_fold_map
load_memmap_description = _experiment_fold_index.load_memmap_description
recover_test_record_and_fold_idx = _experiment_fold_index.recover_test_record_and_fold_idx
check_indexed_run_complete = _indexed_run_complete.check_indexed_run_complete

DATASET_SETTINGS = {
    "dodh": DODH_SETTINGS,
    "dodo": DODO_SETTINGS,
}


def _numeric_performance(perf: Any) -> Dict[str, float]:
    """Keep only numeric entries from performance_on_test_set for JSON-safe aggregates."""
    if not isinstance(perf, dict):
        return {}
    out: Dict[str, float] = {}
    for k, v in perf.items():
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            out[str(k)] = float(v)
    return out


def parse_run(run_root: str, run_id: str, fold_map: Dict[str, int], metric: str) -> dict:
    """Same logic as ``scripts/experiment_utils/index_experiments.parse_run``, plus full test perf."""
    run_dir = os.path.join(run_root, run_id)
    try:
        run_path_for_output = os.path.relpath(run_dir, os.getcwd())
    except Exception:
        run_path_for_output = run_dir
    run_path_for_output = run_path_for_output.replace("\\", "/")
    record = {
        "run_id": run_id,
        "run_path": run_path_for_output,
        "status": "failed",
        "reason": None,
        "fold_idx": None,
        "test_record": None,
        "metric_name": metric,
        "metric_value": None,
        "performance_on_test_set": None,
        "metadata_end": None,
        "timestamp": int(time.time()),
    }

    description_path = os.path.join(run_dir, "description.json")
    if not os.path.isfile(description_path):
        record["reason"] = "missing_description"
        return record

    try:
        with open(description_path, "r") as f:
            description = json.load(f)
    except Exception as exc:
        record["reason"] = "description_parse_error:{}".format(exc)
        return record

    complete, reason = check_indexed_run_complete(run_dir, description)
    if not complete:
        record["reason"] = reason
        record["metadata_end"] = description.get("metadata", {}).get("end")
        tr, fi = recover_test_record_and_fold_idx(description, fold_map)
        record["test_record"] = tr
        record["fold_idx"] = fi
        return record

    test_record = description["records_split"]["test_records"][0]
    if test_record not in fold_map:
        dataset_parameters = description.get("dataset_parameters")
        if not isinstance(dataset_parameters, dict):
            dataset_parameters = {}
        split = dataset_parameters.get("split")
        if not isinstance(split, dict):
            split = {}
        split_test = split.get("test")
        if isinstance(split_test, list) and len(split_test) == 1 and isinstance(split_test[0], str):
            recovered = os.path.basename(os.path.normpath(split_test[0]))
            if recovered in fold_map:
                test_record = recovered
    perf = description.get("performance_on_test_set")
    perf_num = _numeric_performance(perf)
    record["performance_on_test_set"] = perf_num if perf_num else None

    metric_value = perf_num.get(metric) if perf_num else None
    if metric_value is None:
        record["reason"] = "missing_metric:{}".format(metric)
        return record

    record["status"] = "completed"
    record["reason"] = None
    record["test_record"] = test_record
    record["fold_idx"] = fold_map.get(test_record)
    if record["fold_idx"] is None:
        record["reason"] = "unknown_test_record_in_fold_map"
        record["status"] = "failed"
        record["performance_on_test_set"] = None
    record["metric_value"] = metric_value
    record["metadata_end"] = description.get("metadata", {}).get("end")
    return record


def choose_best_per_fold(records: List[dict]) -> Dict[int, dict]:
    best: Dict[int, dict] = {}
    for record in records:
        if record["status"] != "completed":
            continue
        fold_idx = record["fold_idx"]
        if fold_idx is None:
            continue
        prev = best.get(fold_idx)
        if prev is None or record["metric_value"] > prev["metric_value"]:
            best[fold_idx] = record
    return best


def _fold_summary_rows(
    dataset: str,
    model: str,
    metric: str,
    base_experiments_dir: str,
    runs_root: Optional[str],
) -> List[Dict[str, Any]]:
    if dataset not in DATASET_SETTINGS:
        raise ValueError(
            "Unsupported dataset {!r}; expected one of: {}".format(
                dataset, sorted(DATASET_SETTINGS.keys())
            )
        )
    dataset_setting = DATASET_SETTINGS[dataset]
    memmap_description = load_memmap_description(base_experiments_dir, model, dataset)
    fold_map = build_loov_fold_map(dataset_setting, memmap_description)

    root = runs_root or os.path.join(EXPERIMENTS_DIRECTORY, dataset, model)
    if not os.path.isdir(root):
        raise FileNotFoundError("Runs root does not exist: {!r}".format(root))

    run_ids = [
        name
        for name in os.listdir(root)
        if os.path.isdir(os.path.join(root, name)) and not name.startswith(".")
    ]
    run_ids.sort()
    records = [parse_run(root, run_id, fold_map, metric) for run_id in run_ids]
    best_by_fold = choose_best_per_fold(records)

    summary_rows: List[Dict[str, Any]] = []
    for test_record, fold_idx in sorted(fold_map.items(), key=lambda x: x[1]):
        best = best_by_fold.get(fold_idx)
        if best is None:
            summary_rows.append(
                {
                    "fold_idx": fold_idx,
                    "test_record": test_record,
                    "run_id": None,
                    "run_path": None,
                    "metric_name": metric,
                    "metric_value": None,
                    "performance_on_test_set": None,
                    "metadata_end": None,
                }
            )
        else:
            summary_rows.append(
                {
                    "fold_idx": fold_idx,
                    "test_record": test_record,
                    "run_id": best["run_id"],
                    "run_path": best["run_path"],
                    "metric_name": best["metric_name"],
                    "metric_value": best["metric_value"],
                    "performance_on_test_set": best.get("performance_on_test_set"),
                    "metadata_end": best["metadata_end"],
                }
            )
    return summary_rows


def _aggregate(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {
            "mean": None,
            "stdev": None,
            "min": None,
            "max": None,
        }
    out: Dict[str, Optional[float]] = {
        "mean": float(statistics.mean(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }
    if len(values) >= 2:
        out["stdev"] = float(statistics.stdev(values))
    else:
        out["stdev"] = None
    return out


def _aggregate_all_metrics(
    fold_rows: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Optional[float]]]:
    """Mean/stdev/min/max per metric name over folds (keys must appear on every fold)."""
    per_fold_perfs: List[Dict[str, float]] = []
    for row in fold_rows:
        p = row.get("performance_on_test_set")
        if isinstance(p, dict) and p:
            per_fold_perfs.append(dict(p))
        else:
            per_fold_perfs.append({})

    if not any(per_fold_perfs):
        return {}

    common_keys: Set[str] = set(per_fold_perfs[0].keys())
    for p in per_fold_perfs[1:]:
        common_keys &= set(p.keys())

    out: Dict[str, Dict[str, Optional[float]]] = {}
    for mk in sorted(common_keys):
        values = [float(p[mk]) for p in per_fold_perfs]
        out[mk] = _aggregate(values)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model folder name under EXPERIMENTS_DIRECTORY/<dataset>/ (e.g. simple_sleep_net).",
    )
    parser.add_argument(
        "--dataset",
        default="dodh",
        help="Dataset name (default: dodh).",
    )
    parser.add_argument(
        "--metric",
        default="cohen_kappa",
        help="Metric used to pick the best run per fold (default: cohen_kappa).",
    )
    parser.add_argument(
        "--base-experiments-dir",
        default="scripts/base_experiments",
        help="Directory containing per-model memmaps.json (default: scripts/base_experiments).",
    )
    parser.add_argument(
        "--runs-root",
        default=None,
        help="Override experiment runs root (default: EXPERIMENTS_DIRECTORY/<dataset>/<model>).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory for the summary JSON. Default: same as the runs root "
            "(EXPERIMENTS_DIRECTORY/<dataset>/<model>/), i.e. where per-fold runs live."
        ),
    )
    parser.add_argument(
        "--output-name",
        default="losocv_pretrained_summary.json",
        help="Summary filename (default: losocv_pretrained_summary.json).",
    )
    args = parser.parse_args()

    runs_root = args.runs_root or os.path.join(
        EXPERIMENTS_DIRECTORY, args.dataset, args.model
    )
    try:
        summary_rows = _fold_summary_rows(
            args.dataset,
            args.model,
            args.metric,
            args.base_experiments_dir,
            args.runs_root,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print("Error: {}".format(exc), file=sys.stderr)
        sys.exit(1)

    missing: List[int] = []
    for row in summary_rows:
        if row.get("metric_value") is None:
            fi = row.get("fold_idx")
            if isinstance(fi, int):
                missing.append(fi)

    values = [
        float(row["metric_value"])
        for row in summary_rows
        if row.get("metric_value") is not None
    ]

    if missing:
        print(
            "Error: incomplete LOSOCV: {} fold(s) lack a completed run with metric {!r}: {}".format(
                len(missing),
                args.metric,
                sorted(missing),
            ),
            file=sys.stderr,
        )
        print(
            "Runs indexed from: {!r}".format(os.path.abspath(runs_root)),
            file=sys.stderr,
        )
        sys.exit(1)

    n_folds = len(summary_rows)
    if len(values) != n_folds:
        print(
            "Error: internal check failed: expected {} metric values, got {}".format(
                n_folds, len(values)
            ),
            file=sys.stderr,
        )
        sys.exit(1)

    out_dir = args.output_dir or runs_root
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, args.output_name)

    try:
        runs_rel = os.path.relpath(os.path.abspath(runs_root), os.getcwd())
    except Exception:
        runs_rel = os.path.abspath(runs_root)
    runs_rel = runs_rel.replace("\\", "/")

    aggregate_by_metric = _aggregate_all_metrics(summary_rows)
    sel_agg = aggregate_by_metric.get(args.metric) or _aggregate(values)

    payload: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "model": args.model,
        "selection_metric": args.metric,
        "metric": args.metric,
        "output_directory": os.path.abspath(out_dir).replace("\\", "/"),
        "n_folds": n_folds,
        "runs_root": runs_rel,
        "aggregate_by_metric": aggregate_by_metric,
        "aggregate": sel_agg,
        "folds": summary_rows,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    stdev_s = "{:.6f}".format(sel_agg["stdev"]) if sel_agg["stdev"] is not None else "n/a"
    print("Wrote: {}".format(os.path.abspath(out_path)))
    print(
        "LOSOCV {}: mean={:.6f} stdev={} min={:.6f} max={:.6f} (n={})".format(
            args.metric,
            sel_agg["mean"] if sel_agg["mean"] is not None else float("nan"),
            stdev_s,
            sel_agg["min"] if sel_agg["min"] is not None else float("nan"),
            sel_agg["max"] if sel_agg["max"] is not None else float("nan"),
            n_folds,
        )
    )
    if aggregate_by_metric:
        other = [k for k in sorted(aggregate_by_metric.keys()) if k != args.metric]
        if other:
            print("Also aggregated across all folds: {}".format(", ".join(other)))


if __name__ == "__main__":
    main()
