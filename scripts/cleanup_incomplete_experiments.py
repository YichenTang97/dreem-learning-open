"""
Cleanup incomplete/failed experiment UUID folders.

"Incomplete" matches ``check_indexed_run_complete`` (same as ``index_experiments``).

By default, only incomplete runs whose LOOCV fold already has at least one
*completed* run in the same ``runs_root`` are candidates for deletion (e.g. stale
duplicate attempts). Incomplete runs that are the only folder for their test
subject are kept so you do not lose the only attempt at a fold.

Use ``--all-incomplete`` to delete every incomplete folder (previous behavior),
subject to ``--keep-last``.

Default mode is dry-run (no deletion). Use ``--apply`` to actually remove folders.

Examples:
  python scripts/cleanup_incomplete_experiments.py
  python scripts/cleanup_incomplete_experiments.py --apply
  python scripts/cleanup_incomplete_experiments.py --dataset dodh --algo simple_sleep_net --apply
  python scripts/cleanup_incomplete_experiments.py --all-incomplete --apply
"""
import argparse
import json
import os
import shutil
from typing import List, Optional, Set, Tuple

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


def classify_run(run_dir: str) -> Tuple[str, str]:
    """
    Return (status, reason) where status in {"completed", "incomplete"}.
    """
    description_path = os.path.join(run_dir, "description.json")
    if not os.path.isfile(description_path):
        return "incomplete", "missing_description"

    try:
        with open(description_path, "r", encoding="utf-8-sig") as f:
            description = json.load(f)
    except Exception as exc:
        return "incomplete", "description_parse_error:{}".format(exc)

    if not isinstance(description, dict):
        return "incomplete", "invalid_description_type"

    complete, reason = check_indexed_run_complete(run_dir, description)
    if complete:
        return "completed", "ok"
    return "incomplete", reason or "unknown"


def _load_description_dict(run_dir: str) -> Optional[dict]:
    description_path = os.path.join(run_dir, "description.json")
    if not os.path.isfile(description_path):
        return None
    try:
        with open(description_path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


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
        "--base-experiments-dir",
        default="scripts/base_experiments",
        help="Directory containing <algo>/memmaps.json (must match indexing)",
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
        help="Keep newest N deletion candidates (by folder mtime) and do not delete them.",
    )
    parser.add_argument(
        "--all-incomplete",
        action="store_true",
        help="Delete all incomplete runs, not only those with a completed run for the same fold.",
    )
    args = parser.parse_args()

    if args.dataset not in DATASET_SETTINGS:
        raise ValueError(
            "Unsupported dataset {!r}; expected one of: {}".format(
                args.dataset, sorted(DATASET_SETTINGS.keys())
            )
        )

    runs_root = args.runs_root or os.path.join(EXPERIMENTS_DIRECTORY, args.dataset, args.algo)
    if not os.path.isdir(runs_root):
        raise FileNotFoundError("Runs root does not exist: {!r}".format(runs_root))

    memmap_description = load_memmap_description(
        args.base_experiments_dir, args.algo, args.dataset
    )
    fold_map = build_loov_fold_map(DATASET_SETTINGS[args.dataset], memmap_description)

    run_ids = [
        name
        for name in os.listdir(runs_root)
        if os.path.isdir(os.path.join(runs_root, name)) and not name.startswith(".")
    ]
    run_ids.sort()

    incomplete: List[dict] = []
    completed: List[dict] = []
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

    test_records_with_completed: Set[str] = set()
    for item in completed:
        desc = _load_description_dict(item["run_dir"])
        if desc is None:
            continue
        tr, _ = recover_test_record_and_fold_idx(desc, fold_map)
        if tr is not None:
            test_records_with_completed.add(tr)

    if args.all_incomplete:
        deletion_pool = list(incomplete)
    else:
        deletion_pool = []
        skipped_protected = 0
        for item in incomplete:
            desc = _load_description_dict(item["run_dir"])
            if desc is None:
                skipped_protected += 1
                continue
            tr, _ = recover_test_record_and_fold_idx(desc, fold_map)
            if tr is None:
                skipped_protected += 1
                continue
            if tr in test_records_with_completed:
                deletion_pool.append(item)
            else:
                skipped_protected += 1
        print(
            "Incomplete skipped (sole attempt for fold or cannot resolve test subject):",
            skipped_protected,
        )

    incomplete_sorted_newest = sorted(deletion_pool, key=lambda x: x["mtime"], reverse=True)
    keep_ids = set()
    if args.keep_last > 0:
        for item in incomplete_sorted_newest[: args.keep_last]:
            keep_ids.add(item["run_id"])

    candidates = [item for item in incomplete_sorted_newest if item["run_id"] not in keep_ids]

    print("Runs root:", runs_root)
    print("Completed:", len(completed))
    print("Incomplete (total):", len(incomplete))
    if not args.all_incomplete:
        print(
            "Deletion policy: only incomplete runs whose test fold has another completed run "
            "in this directory (use --all-incomplete for every incomplete folder)."
        )
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
