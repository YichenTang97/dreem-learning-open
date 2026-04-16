"""
Wrapper around scripts/index_experiments.py with SOL-friendly defaults.

Usage:
    python sol_experiments/index_experiments.py
    python sol_experiments/index_experiments.py --dataset dodh --algo cnn_rnn
    python sol_experiments/index_experiments.py --runs-root /custom/experiments/dodh/cnn_rnn
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
        help="Directory containing experiment configs",
    )
    parser.add_argument(
        "--runs-root",
        default=None,
        help="Override experiment runs root",
    )
    parser.add_argument(
        "--metric",
        default="cohen_kappa",
        help="Metric used to select best run per fold",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    index_script = os.path.join(repo_root, "scripts", "index_experiments.py")

    command: List[str] = [
        sys.executable,
        index_script,
        "--dataset",
        args.dataset,
        "--algo",
        args.algo,
        "--base-experiments-dir",
        args.base_experiments_dir,
        "--metric",
        args.metric,
    ]
    if args.runs_root:
        command.extend(["--runs-root", args.runs_root])

    completed = subprocess.run(command, cwd=repo_root)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
