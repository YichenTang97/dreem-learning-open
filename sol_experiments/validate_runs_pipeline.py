"""
Wrapper around scripts/validate_runs_pipeline.py with SOL-friendly defaults.

Injects ``--algo cnn_rnn`` and ``--base-experiments-dir sol_experiments/configs``
before your arguments so CLI overrides still work (last duplicate flag wins).

Usage:
    python sol_experiments/validate_runs_pipeline.py
    python sol_experiments/validate_runs_pipeline.py -y
    python sol_experiments/validate_runs_pipeline.py --skip-rclone
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import List


def main() -> int:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = os.path.join(repo_root, "scripts", "validate_runs_pipeline.py")

    command: List[str] = [
        sys.executable,
        script,
        "--algo",
        "cnn_rnn",
        "--base-experiments-dir",
        os.path.join("sol_experiments", "configs"),
    ]
    command.extend(sys.argv[1:])

    env = os.environ.copy()
    prev = env.get("PYTHONPATH")
    env["PYTHONPATH"] = repo_root if not prev else repo_root + os.pathsep + prev

    return subprocess.run(command, cwd=repo_root, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
