"""
Index Simple Sleep Net experiment runs and export completion summaries.

Outputs (under data/experiments/dodh/simple_sleep_net by default):
- completed_folds.jsonl : one JSON record per discovered run with status
- fold_summary.csv      : one row per fold with best completed run (by metric)
- fold_summary.json     : JSON version of fold_summary.csv

Usage:
    python scripts/index_experiments.py
    python scripts/index_experiments.py --metric cohen_kappa
    python scripts/index_experiments.py --dataset dodh --algo simple_sleep_net
"""
import argparse
import csv
import hashlib
import json
import os
import random as rd
import time
from typing import Dict, List, Optional, Tuple

from dreem_learning_open.settings import DODH_SETTINGS, EXPERIMENTS_DIRECTORY


def memmap_hash(memmap_description: dict) -> str:
    return hashlib.sha1(json.dumps(memmap_description).encode()).hexdigest()[:10]


def load_dodh_memmap_description(base_experiments_dir: str, algo: str) -> dict:
    memmaps_path = os.path.join(base_experiments_dir, algo, "memmaps.json")
    with open(memmaps_path, "r") as f:
        memmaps_description = json.load(f)

    for description in memmaps_description:
        if description.get("dataset") == "dodh":
            description = dict(description)
            del description["dataset"]
            return description
    raise RuntimeError("No DODH memmap_description found in {}".format(memmaps_path))


def build_loov_fold_map(dataset_setting: dict, memmap_description: dict) -> Dict[str, int]:
    dataset_dir = os.path.join(dataset_setting["memmap_directory"], memmap_hash(memmap_description))
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError("Memmap directory does not exist: {!r}".format(dataset_dir))

    records = [
        os.path.join(dataset_dir, record_name)
        for record_name in os.listdir(dataset_dir)
        if ".json" not in record_name
    ]
    rd.seed(2019)
    rd.shuffle(records)

    # dodh path in run_experiments is LOOV: folds = [[record] for record in records]
    return {os.path.basename(record): idx for idx, record in enumerate(records)}


def check_run_complete(run_dir: str, description: dict) -> Tuple[bool, Optional[str]]:
    required_files = [
        os.path.join(run_dir, "description.json"),
        os.path.join(run_dir, "hypnograms.json"),
        os.path.join(run_dir, "best_model.gz"),
        os.path.join(run_dir, "training", "best_net"),
    ]
    for file_path in required_files:
        if not os.path.isfile(file_path):
            return False, "missing_file:{}".format(os.path.relpath(file_path, run_dir))

    meta = description.get("metadata", {})
    if not meta.get("end"):
        return False, "metadata.end_missing"

    test_records = description.get("records_split", {}).get("test_records", [])
    if not isinstance(test_records, list) or len(test_records) != 1:
        return False, "invalid_test_records"

    perf = description.get("performance_on_test_set")
    if not isinstance(perf, dict) or len(perf) == 0:
        return False, "empty_performance_on_test_set"

    return True, None


def parse_run(run_root: str, run_id: str, fold_map: Dict[str, int], metric: str) -> dict:
    run_dir = os.path.join(run_root, run_id)
    record = {
        "run_id": run_id,
        "run_path": run_dir,
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

    complete, reason = check_run_complete(run_dir, description)
    if not complete:
        record["reason"] = reason
        record["metadata_end"] = description.get("metadata", {}).get("end")
        return record

    test_record = description["records_split"]["test_records"][0]
    # logger.py currently stores records_split with os.path.split(record)[-2],
    # which on Windows often yields the parent directory instead of record id.
    # Recover robustly from dataset_parameters.split.test (full memmap path).
    if test_record not in fold_map:
        split_test = description.get("dataset_parameters", {}).get("split", {}).get("test")
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

    if args.dataset != "dodh":
        raise ValueError("This indexing script currently supports dataset=dodh only")

    dataset_setting = DODH_SETTINGS
    memmap_description = load_dodh_memmap_description(args.base_experiments_dir, args.algo)
    fold_map = build_loov_fold_map(dataset_setting, memmap_description)

    runs_root = args.runs_root or os.path.join(EXPERIMENTS_DIRECTORY, args.dataset, args.algo)
    if not os.path.isdir(runs_root):
        raise FileNotFoundError("Runs root does not exist: {!r}".format(runs_root))

    run_ids = [name for name in os.listdir(runs_root) if os.path.isdir(os.path.join(runs_root, name))]
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
    print("Indexed runs root:", runs_root)
    print("Discovered run folders:", len(run_ids))
    print("Completed folds (best run found): {}/{}".format(completed_count, len(summary_rows)))
    print("Wrote:", jsonl_path)
    print("Wrote:", csv_path)
    print("Wrote:", json_path)


if __name__ == "__main__":
    main()
