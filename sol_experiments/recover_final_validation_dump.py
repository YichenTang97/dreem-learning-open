"""
Wrapper around scripts/recover_final_validation_dump.py with SOL-friendly defaults.

Defaults restrict recovery to ``--dataset dodh`` and ``--algo cnn_rnn`` unless you
override on the command line (last ``--algo`` / ``--dataset`` wins).

Usage:
    python sol_experiments/recover_final_validation_dump.py --dry-run
    python sol_experiments/recover_final_validation_dump.py
    python sol_experiments/recover_final_validation_dump.py --runs-root /path/to/dodh/cnn_rnn
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import List


def main() -> int:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = os.path.join(repo_root, "scripts", "recover_final_validation_dump.py")

    command: List[str] = [
        sys.executable,
        script,
        "--dataset",
        "dodh",
        "--algo",
        "cnn_rnn",
        "--repo-root",
        repo_root,
    ]
    command.extend(sys.argv[1:])

    env = os.environ.copy()
    prev = env.get("PYTHONPATH")
    env["PYTHONPATH"] = repo_root if not prev else repo_root + os.pathsep + prev

    return subprocess.run(command, cwd=repo_root, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
