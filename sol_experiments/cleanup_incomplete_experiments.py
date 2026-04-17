"""
Wrapper around scripts/cleanup_incomplete_experiments.py with SOL-friendly defaults.

Uses ``sol_experiments/configs`` for memmaps (same fold identity as ``index_experiments``).

Usage:
    python sol_experiments/cleanup_incomplete_experiments.py
    python sol_experiments/cleanup_incomplete_experiments.py --apply
    python sol_experiments/cleanup_incomplete_experiments.py --dataset dodh --algo cnn_rnn --apply
    python sol_experiments/cleanup_incomplete_experiments.py --runs-root /custom/experiments/dodh/cnn_rnn
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import List


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="dodh", help="Dataset name")
    parser.add_argument("--algo", default="cnn_rnn", help="Algorithm folder name")
    parser.add_argument(
        "--base-experiments-dir",
        default=os.path.join("sol_experiments", "configs"),
        help="Directory containing <algo>/memmaps.json (must match indexing)",
    )
    parser.add_argument(
        "--runs-root",
        default=None,
        help="Override experiment runs root",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete folders (default is dry-run)",
    )
    parser.add_argument(
        "--keep-last",
        type=int,
        default=0,
        help="Keep newest N deletion candidates by folder mtime",
    )
    parser.add_argument(
        "--all-incomplete",
        action="store_true",
        help="Delete every incomplete run, not only redundant fold attempts",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cleanup_script = os.path.join(repo_root, "scripts", "cleanup_incomplete_experiments.py")

    command: List[str] = [
        sys.executable,
        cleanup_script,
        "--dataset",
        args.dataset,
        "--algo",
        args.algo,
        "--base-experiments-dir",
        args.base_experiments_dir,
        "--keep-last",
        str(args.keep_last),
    ]
    if args.runs_root:
        command.extend(["--runs-root", args.runs_root])
    if args.apply:
        command.append("--apply")
    if args.all_incomplete:
        command.append("--all-incomplete")

    env = os.environ.copy()
    prev_py = env.get("PYTHONPATH")
    env["PYTHONPATH"] = repo_root if not prev_py else repo_root + os.pathsep + prev_py

    completed = subprocess.run(command, cwd=repo_root, env=env)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
