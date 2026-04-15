"""
Cleanup incomplete/failed experiment UUID folders.

Default mode is dry-run (no deletion). Use --apply to actually remove folders.

Examples:
  python scripts/cleanup_incomplete_experiments.py
  python scripts/cleanup_incomplete_experiments.py --apply
  python scripts/cleanup_incomplete_experiments.py --dataset dodh --algo simple_sleep_net --apply
"""
import argparse
import json
import os
import shutil
from typing import Optional, Tuple

from dreem_learning_open.settings import EXPERIMENTS_DIRECTORY


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


def classify_run(run_dir: str) -> Tuple[str, str]:
    """
    Return (status, reason) where status in {"completed", "incomplete"}.
    """
    description_path = os.path.join(run_dir, "description.json")
    if not os.path.isfile(description_path):
        return "incomplete", "missing_description"

    try:
        with open(description_path, "r") as f:
            description = json.load(f)
    except Exception as exc:
        return "incomplete", "description_parse_error:{}".format(exc)

    complete, reason = check_run_complete(run_dir, description)
    if complete:
        return "completed", "ok"
    return "incomplete", reason or "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="dodh", help="Dataset under EXPERIMENTS_DIRECTORY")
    parser.add_argument("--algo", default="simple_sleep_net", help="Algorithm folder under dataset")
    parser.add_argument(
        "--runs-root",
        default=None,
        help="Override runs root (default: EXPERIMENTS_DIRECTORY/<dataset>/<algo>)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete incomplete folders. Default is dry-run.",
    )
    parser.add_argument(
        "--keep-last",
        type=int,
        default=0,
        help="Keep newest N incomplete runs (by folder mtime) and do not delete them.",
    )
    args = parser.parse_args()

    runs_root = args.runs_root or os.path.join(EXPERIMENTS_DIRECTORY, args.dataset, args.algo)
    if not os.path.isdir(runs_root):
        raise FileNotFoundError("Runs root does not exist: {!r}".format(runs_root))

    run_ids = [name for name in os.listdir(runs_root) if os.path.isdir(os.path.join(runs_root, name))]
    run_ids.sort()

    incomplete = []
    completed = []
    for run_id in run_ids:
        run_dir = os.path.join(runs_root, run_id)
        status, reason = classify_run(run_dir)
        item = {
            "run_id": run_id,
            "run_dir": run_dir,
            "reason": reason,
            "mtime": os.path.getmtime(run_dir),
        }
        if status == "completed":
            completed.append(item)
        else:
            incomplete.append(item)

    incomplete_sorted_newest = sorted(incomplete, key=lambda x: x["mtime"], reverse=True)
    keep_ids = set()
    if args.keep_last > 0:
        for item in incomplete_sorted_newest[: args.keep_last]:
            keep_ids.add(item["run_id"])

    candidates = [item for item in incomplete_sorted_newest if item["run_id"] not in keep_ids]

    print("Runs root:", runs_root)
    print("Completed:", len(completed))
    print("Incomplete:", len(incomplete))
    if keep_ids:
        print("Keeping newest incomplete (--keep-last={}): {}".format(args.keep_last, len(keep_ids)))
    print("Candidates for deletion:", len(candidates))

    if not candidates:
        print("Nothing to delete.")
        return 0

    for item in candidates:
        print("- {}  [{}]".format(item["run_id"], item["reason"]))

    if not args.apply:
        print("\nDry-run only. Re-run with --apply to delete these folders.")
        return 0

    deleted = 0
    failed = 0
    for item in candidates:
        try:
            shutil.rmtree(item["run_dir"])
            deleted += 1
        except Exception as exc:
            failed += 1
            print("Failed to delete {}: {}".format(item["run_id"], exc))

    print("\nDeleted:", deleted)
    print("Delete failures:", failed)
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
