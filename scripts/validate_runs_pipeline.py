"""
End-to-end pipeline: optional rclone bisync, recover incomplete validation dumps,
cleanup redundant incomplete run folders, re-index, final bisync, then a clear
summary of folds still to complete (printed last).

Steps:
  1. Rclone bisync of the **local data directory only** (default: ``<repo-root>/data``)
     against **``r2:research-data/dreem/data``** on the cloud. Skipped with
     ``--skip-rclone``. **Default** is incremental bisync (no ``--resync``). Pass
     ``--resync`` when you need the first-time bisync from ``rclone_sync_commands.md``.
     Paths are absolute and work on Windows and Linux.
  2. ``recover_final_validation_dump`` (dry-run first unless ``-y``).
  3. ``cleanup_incomplete_experiments`` (dry-run first unless ``-y``), then
     ``--apply`` with default policy (only extra failed tries when fold complete).
  4. ``index_experiments`` (refreshes ``fold_summary.json`` under the runs root).
  5. **Final** rclone bisync (same paths and ``--resync`` behavior as step 1) to
     push local changes. Confirmation is required unless you passed ``-y`` / ``--yes``.
  6. A **prominent summary** of folds that still need a completed best run (printed
     last so it is easy to spot).

The **first** rclone bisync **always** asks for confirmation (``-y`` does not skip
it), after printing paths and the full command. ``-y`` skips the **final** bisync
prompt (runs immediately) and skips recover/cleanup prompts as before. Without
``-y``, recovery runs ``--dry-run`` first then prompts; cleanup dry-runs then
prompts before ``--apply``. Declining either bisync skips that sync only; the
pipeline continues.

Examples:
  python scripts/validate_runs_pipeline.py
  python scripts/validate_runs_pipeline.py -y
  python scripts/validate_runs_pipeline.py --resync
  python scripts/validate_runs_pipeline.py --dataset dodh --algo cnn_rnn \\
      --base-experiments-dir sol_experiments/configs
  python scripts/validate_runs_pipeline.py --skip-rclone --yes
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from typing import List, Optional, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dreem_learning_open.settings import EXPERIMENTS_DIRECTORY

SUMMARY_LINE_RE = re.compile(
    r"Summary:\s*ok=(\d+)\s+dry_run=(\d+)\s+skipped=(\d+)\s+no_completed_best_model=(\d+)\s+errors=(\d+)"
)
CANDIDATES_RE = re.compile(r"Candidates for deletion:\s*(\d+)")


def _subprocess_env(repo_root: str) -> dict:
    env = os.environ.copy()
    prev = env.get("PYTHONPATH")
    env["PYTHONPATH"] = repo_root if not prev else repo_root + os.pathsep + prev
    return env


def _confirm(message: str, assume_yes: bool) -> bool:
    if assume_yes:
        return True
    try:
        reply = input("{} [y/N]: ".format(message)).strip().lower()
    except EOFError:
        return False
    return reply in ("y", "yes")


def _parse_recover_summary(text: str) -> Optional[Tuple[int, int, int, int, int]]:
    for line in text.splitlines():
        m = SUMMARY_LINE_RE.search(line)
        if m:
            return tuple(int(m.group(i)) for i in range(1, 6))
    return None


def _parse_cleanup_candidates(text: str) -> Optional[int]:
    for line in text.splitlines():
        m = CANDIDATES_RE.search(line)
        if m:
            return int(m.group(1))
    return None


def _normalize_rclone_local_path(path: str) -> str:
    """Absolute, normalized local filesystem path (Windows and POSIX)."""
    return os.path.normpath(os.path.abspath(path))


def _rclone_bisync_argv(
    local_path: str,
    remote: str,
    *,
    use_resync: bool,
) -> Tuple[Optional[str], str, List[str]]:
    """
    Build rclone bisync argv. Returns (rclone_exe_or_none, normalized_local, argv).
    If rclone is not on PATH, argv is empty.
    """
    rclone = shutil.which("rclone")
    local_norm = _normalize_rclone_local_path(local_path)
    if not rclone:
        return None, local_norm, []

    common_suffix: List[str] = [
        "--conflict-resolve",
        "newer",
        "--resilient",
        "--recover",
        "--modify-window",
        "2s",
        "--compare",
        "size,modtime",
        "--create-empty-src-dirs",
        "--retries",
        "10",
        "--low-level-retries",
        "20",
        "--timeout",
        "1m",
        "-vP",
    ]

    if use_resync:
        cmd: List[str] = [
            rclone,
            "bisync",
            local_norm,
            remote,
            "--resync",
        ] + common_suffix
    else:
        cmd = [
            rclone,
            "bisync",
            local_norm,
            remote,
            "--max-lock",
            "2m",
        ] + common_suffix

    return rclone, local_norm, cmd


def _rclone_bisync_phase(
    repo_root: str,
    rclone_local: str,
    rclone_remote: str,
    use_resync: bool,
    *,
    heading: str,
    confirm_message: str,
    confirm_assume_yes: bool,
    skip_message: str,
) -> int:
    """
    Print bisync details, optionally confirm, then run rclone.
    Returns 0 on success or skip, non-zero if rclone exits with an error.
    """
    os.makedirs(rclone_local, exist_ok=True)
    rclone_exe, local_norm, bisync_argv = _rclone_bisync_argv(
        rclone_local, rclone_remote, use_resync=use_resync
    )
    print(heading)
    print("  repo root:       ", repo_root)
    print("  local data dir:  ", local_norm)
    print("  remote:          ", rclone_remote)
    if use_resync:
        print("  mode:            bisync with --resync (you passed --resync)")
    else:
        print("  mode:            incremental bisync (default; no --resync)")
    print("  rclone binary:   ", rclone_exe or "(not found on PATH)")
    if bisync_argv:
        print("  full command:")
        print("   ", " ".join(bisync_argv))
    if not rclone_exe:
        print(
            "Skipping rclone bisync: install rclone or use --skip-rclone",
            file=sys.stderr,
        )
        return 0
    if not _confirm(confirm_message, confirm_assume_yes):
        print(skip_message)
        return 0
    rc = subprocess.run(bisync_argv, shell=False).returncode
    if rc != 0:
        print("rclone bisync failed with code", rc, file=sys.stderr)
    return rc


def _print_remaining_folds_banner(
    runs_root: str,
    summary_path: str,
    pending: List[dict],
    total_folds: int,
) -> None:
    """Last thing the user sees: obvious list of folds still to complete."""
    width = 78
    line = "=" * width
    runs_disp = runs_root.replace("\\", "/")
    summary_disp = summary_path.replace("\\", "/")

    print()
    print(line)
    print(line)
    print(
        "  >>>  REMAINING WORK  —  FOLDS STILL TO COMPLETE  "
        "(no completed best run yet)"
    )
    print(line)
    print(line)
    print()
    print("  Where to look:")
    print("    runs_root   ", runs_disp)
    print("    fold table  ", summary_disp)
    print()

    if not pending:
        print("  ***  All folds done: every LOOCV fold has a completed best run.  ***")
        print()
        print("  Pending folds: 0/{}".format(total_folds))
    else:
        print(
            "  ***  {} fold(s) STILL NEED a finished run — train / fix / recover these:  ***".format(
                len(pending)
            )
        )
        print()
        for r in pending:
            print(
                "      •  fold {:>3}     test record (held-out subject):  {}".format(
                    r.get("fold_idx"),
                    r.get("test_record"),
                )
            )
        print()
        print(
            "  Pending folds: {}/{}  —  see fold_summary.csv / fold_summary.json in runs_root".format(
                len(pending),
                total_folds,
            )
        )

    print()
    print(line)
    print(line)
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="dodh", help="Dataset name")
    parser.add_argument("--algo", default="simple_sleep_net", help="Algorithm folder name")
    parser.add_argument(
        "--base-experiments-dir",
        default=os.path.join("scripts", "base_experiments"),
        help="Experiment configs (memmaps); must match index/cleanup",
    )
    parser.add_argument(
        "--runs-root",
        default=None,
        help="Runs directory (default: EXPERIMENTS_DIRECTORY/<dataset>/<algo>)",
    )
    parser.add_argument(
        "--repo-root",
        default=_REPO_ROOT,
        help="Repository root (recover path resolution)",
    )
    parser.add_argument(
        "--metric",
        default="cohen_kappa",
        help="Metric for index_experiments best-fold selection",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help=(
            "Skip the final rclone bisync confirmation and recover/cleanup prompts "
            "(the first rclone bisync still always asks)"
        ),
    )
    parser.add_argument(
        "--skip-rclone",
        action="store_true",
        help="Skip rclone bisync",
    )
    parser.add_argument(
        "--resync",
        action="store_true",
        help=(
            "Run bisync with --resync (first-time style per rclone_sync_commands.md). "
            "Default is incremental bisync without --resync."
        ),
    )
    parser.add_argument(
        "--rclone-local",
        default=None,
        help=(
            "Local **data** directory to bisync (default: <repo-root>/data). "
            "Only this tree is paired with --rclone-remote."
        ),
    )
    parser.add_argument(
        "--rclone-remote",
        default="r2:research-data/dreem/data",
        help="Rclone remote path for the cloud **dreem/data** tree (default: r2:research-data/dreem/data)",
    )
    args = parser.parse_args()

    repo_root = os.path.abspath(args.repo_root)
    runs_root = args.runs_root or os.path.join(
        os.path.abspath(EXPERIMENTS_DIRECTORY), args.dataset, args.algo
    )
    rclone_local = _normalize_rclone_local_path(
        args.rclone_local or os.path.join(repo_root, "data")
    )

    env = _subprocess_env(repo_root)
    recover_script = os.path.join(repo_root, "scripts", "recover_final_validation_dump.py")
    cleanup_script = os.path.join(repo_root, "scripts", "cleanup_incomplete_experiments.py")
    index_script = os.path.join(repo_root, "scripts", "index_experiments.py")
    use_resync = bool(args.resync)

    # --- 1. Rclone (local data/  <->  r2:.../dreem/data only) ---
    if not args.skip_rclone:
        rc = _rclone_bisync_phase(
            repo_root,
            rclone_local,
            args.rclone_remote,
            use_resync,
            heading="--- Rclone bisync (start; always confirm; -y does not skip) ---",
            confirm_message="Run rclone bisync with the paths and command above?",
            confirm_assume_yes=False,
            skip_message="Skipping rclone bisync (continuing with local data only).",
        )
        if rc != 0:
            return rc
    else:
        print("Skipping rclone (--skip-rclone).")

    # --- 2. Recover ---
    recover_base = [
        sys.executable,
        recover_script,
        "--repo-root",
        repo_root,
        "--runs-root",
        runs_root,
    ]

    if args.yes:
        rc = subprocess.run(recover_base, cwd=repo_root, env=env).returncode
        if rc != 0:
            return rc
    else:
        dry_cmd = recover_base + ["--dry-run"]
        proc = subprocess.run(
            dry_cmd, cwd=repo_root, env=env, capture_output=True, text=True
        )
        sys.stdout.write(proc.stdout)
        if proc.stderr:
            sys.stderr.write(proc.stderr)
        if proc.returncode != 0:
            print("recover_final_validation_dump --dry-run failed", file=sys.stderr)
            return proc.returncode
        summary = _parse_recover_summary(proc.stdout)
        dry_n = summary[1] if summary else None
        err_n = summary[4] if summary else 0
        if dry_n is None:
            print(
                "Could not parse recover summary; run recover manually if needed.",
                file=sys.stderr,
            )
        elif dry_n == 0 and err_n == 0:
            print("Recover dry-run: nothing to recover; skipping full recovery.")
        else:
            if not _confirm(
                "Run recover_final_validation_dump for real (writes description/hypnograms)?",
                False,
            ):
                print("Skipping recovery.")
            else:
                rc = subprocess.run(recover_base, cwd=repo_root, env=env).returncode
                if rc != 0:
                    return rc

    # --- 3. Cleanup ---
    cleanup_base = [
        sys.executable,
        cleanup_script,
        "--dataset",
        args.dataset,
        "--algo",
        args.algo,
        "--base-experiments-dir",
        args.base_experiments_dir,
        "--runs-root",
        runs_root,
        "--keep-last",
        "0",
    ]

    if args.yes:
        cleanup_base_apply = cleanup_base + ["--apply"]
        rc = subprocess.run(cleanup_base_apply, cwd=repo_root, env=env).returncode
        if rc != 0:
            return rc
    else:
        proc = subprocess.run(
            cleanup_base, cwd=repo_root, env=env, capture_output=True, text=True
        )
        sys.stdout.write(proc.stdout)
        if proc.stderr:
            sys.stderr.write(proc.stderr)
        if proc.returncode != 0:
            print("cleanup_incomplete_experiments dry-run failed", file=sys.stderr)
            return proc.returncode
        n_del = _parse_cleanup_candidates(proc.stdout)
        if n_del is not None and n_del == 0:
            print("Cleanup: no candidates; skipping --apply.")
        else:
            if not _confirm(
                "Delete listed incomplete folders with cleanup_incomplete_experiments --apply?",
                False,
            ):
                print("Skipping cleanup --apply.")
            else:
                rc = subprocess.run(
                    cleanup_base + ["--apply"], cwd=repo_root, env=env
                ).returncode
                if rc != 0:
                    return rc

    # --- 4. Index ---
    index_cmd = [
        sys.executable,
        index_script,
        "--dataset",
        args.dataset,
        "--algo",
        args.algo,
        "--base-experiments-dir",
        args.base_experiments_dir,
        "--runs-root",
        runs_root,
        "--metric",
        args.metric,
    ]
    rc = subprocess.run(index_cmd, cwd=repo_root, env=env).returncode
    if rc != 0:
        return rc

    summary_path = os.path.join(runs_root, "fold_summary.json")
    if not os.path.isfile(summary_path):
        print("fold_summary.json missing after index", file=sys.stderr)
        return 1
    with open(summary_path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    pending = [r for r in rows if r.get("run_id") is None]

    # --- 5. Final rclone bisync (upload local changes; confirm unless -y) ---
    if not args.skip_rclone:
        rc = _rclone_bisync_phase(
            repo_root,
            rclone_local,
            args.rclone_remote,
            use_resync,
            heading="--- Rclone bisync (final; confirm unless you passed -y / --yes) ---",
            confirm_message=(
                "Run final rclone bisync to push local data changes to the remote?"
            ),
            confirm_assume_yes=args.yes,
            skip_message="Skipping final rclone bisync.",
        )
        if rc != 0:
            return rc

    # --- 6. Remaining folds (printed last on purpose) ---
    _print_remaining_folds_banner(runs_root, summary_path, pending, len(rows))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
