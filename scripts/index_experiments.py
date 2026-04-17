"""
Index experiment runs under EXPERIMENTS_DIRECTORY/<dataset>/<algo>/ and export
completion summaries.

Outputs (under data/experiments/dodh/simple_sleep_net by default):
- completed_folds.jsonl : one JSON record per discovered run with status
- fold_summary.csv      : one row per fold with best completed run (by metric)
- fold_summary.json     : JSON version of fold_summary.csv

A run is ``completed`` only if ``description.json``, ``hypnograms.json``, and
``best_model.gz`` exist and ``performance_on_test_set``, ``performance_per_records``,
and ``records_split`` are all non-null with non-empty content (LOOV: exactly one
``test_records``). ``metadata.end`` is not required, so recovered or interrupted
runs still count as pending for ``run_*_parallel`` until outputs exist.

Usage:
    python scripts/index_experiments.py
    python scripts/index_experiments.py --metric cohen_kappa
    python scripts/index_experiments.py --dataset dodh --algo simple_sleep_net
    python scripts/index_experiments.py --dataset dodh --algo cnn_rnn \\
        --base-experiments-dir sol_experiments/configs
"""
import argparse
import csv
import json
import os
import time
from typing import Dict, List, Optional, Tuple

from dreem_learning_open.settings import DODO_SETTINGS, DODH_SETTINGS, EXPERIMENTS_DIRECTORY
from dreem_learning_open.utils.experiment_fold_index import (
    build_loov_fold_map,
    load_memmap_description,
    recover_test_record_and_fold_idx,
)
from dreem_learning_open.utils.indexed_run_complete import check_indexed_run_complete

DATASET_SETTINGS = {
    "dodh": DODH_SETTINGS,
    "dodo": DODO_SETTINGS,
}


def parse_run(run_root: str, run_id: str, fold_map: Dict[str, int], metric: str) -> dict:
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
    # logger.py currently stores records_split with os.path.split(record)[-2],
    # which on Windows often yields the parent directory instead of record id.
    # Recover robustly from dataset_parameters.split.test (full memmap path).
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
    perf = description["performance_on_test_set"]
    metric_value = perf.get(metric)
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


def write_jsonl(path: str, rows: List[dict]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def write_summary_csv(path: str, rows: List[dict]) -> None:
    if not rows:
        # create header-only file for tooling consistency
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "fold_idx",
                    "test_record",
                    "run_id",
                    "run_path",
                    "metric_name",
                    "metric_value",
                    "metadata_end",
                ],
            )
            writer.writeheader()
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="dodh", help="Dataset name (default: dodh)")
    parser.add_argument("--algo", default="simple_sleep_net", help="Algorithm folder name")
    parser.add_argument(
        "--base-experiments-dir",
        default="scripts/base_experiments",
        help="Directory containing experiment configs",
    )
    parser.add_argument(
        "--runs-root",
        default=None,
        help="Override experiment runs root (default: from settings EXPERIMENTS_DIRECTORY)",
    )
    parser.add_argument(
        "--metric",
        default="cohen_kappa",
        help="Metric from performance_on_test_set used to select best run per fold",
    )
    args = parser.parse_args()

    if args.dataset not in DATASET_SETTINGS:
        raise ValueError(
            "Unsupported dataset {!r}; expected one of: {}".format(
                args.dataset, sorted(DATASET_SETTINGS.keys())
            )
        )

    dataset_setting = DATASET_SETTINGS[args.dataset]
    memmap_description = load_memmap_description(
        args.base_experiments_dir, args.algo, args.dataset
    )
    fold_map = build_loov_fold_map(dataset_setting, memmap_description)

    runs_root = args.runs_root or os.path.join(EXPERIMENTS_DIRECTORY, args.dataset, args.algo)
    if not os.path.isdir(runs_root):
        raise FileNotFoundError("Runs root does not exist: {!r}".format(runs_root))

    run_ids = [
        name
        for name in os.listdir(runs_root)
        if os.path.isdir(os.path.join(runs_root, name)) and not name.startswith(".")
    ]
    run_ids.sort()
    records = [parse_run(runs_root, run_id, fold_map, args.metric) for run_id in run_ids]

    jsonl_path = os.path.join(runs_root, "completed_folds.jsonl")
    write_jsonl(jsonl_path, records)

    best_by_fold = choose_best_per_fold(records)
    summary_rows = []
    for test_record, fold_idx in sorted(fold_map.items(), key=lambda x: x[1]):
        best = best_by_fold.get(fold_idx)
        if best is None:
            summary_rows.append(
                {
                    "fold_idx": fold_idx,
                    "test_record": test_record,
                    "run_id": None,
                    "run_path": None,
                    "metric_name": args.metric,
                    "metric_value": None,
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
                    "metadata_end": best["metadata_end"],
                }
            )

    csv_path = os.path.join(runs_root, "fold_summary.csv")
    write_summary_csv(csv_path, summary_rows)

    json_path = os.path.join(runs_root, "fold_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary_rows, f, indent=2)

    completed_count = sum(1 for row in summary_rows if row["run_id"] is not None)
    folds_still_open = len(summary_rows) - completed_count
    runs_not_completed = sum(1 for r in records if r.get("status") != "completed")
    print("Indexed runs root:", runs_root)
    print("Discovered run folders:", len(run_ids))
    print(
        "Run folders with status != completed (failed / incomplete artifacts): {}".format(
            runs_not_completed
        )
    )
    print(
        "Folds with no completed best run yet: {}/{}".format(
            folds_still_open, len(summary_rows)
        )
    )
    print("Completed folds (best run found): {}/{}".format(completed_count, len(summary_rows)))
    print("Wrote:", jsonl_path)
    print("Wrote:", csv_path)
    print("Wrote:", json_path)


if __name__ == "__main__":
    main()
