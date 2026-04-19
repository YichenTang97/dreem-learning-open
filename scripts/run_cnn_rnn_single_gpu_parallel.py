"""
Run CNN-RNN (``scripts/run_cnn_rnn.py``) LOOCV folds on a single GPU with
configurable worker count.

Mirrors ``scripts/run_simple_sleep_net_single_gpu_parallel.py``:
- retries in rounds until all folds complete (or --max-rounds)
- bootstraps memmaps with a single fold when cache is missing
- re-indexes runs via ``scripts/experiment_utils/index_experiments.py`` each round
- streams worker logs to files and stdout

Usage:
    python scripts/run_cnn_rnn_single_gpu_parallel.py --workers 3

Examples:
    python scripts/run_cnn_rnn_single_gpu_parallel.py --workers 2 --dataset dodh
    python scripts/run_cnn_rnn_single_gpu_parallel.py --workers 4 --cuda-device 1 --dataset dodo
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random as rd
import subprocess
import sys
import threading
from typing import List, Tuple

from dreem_learning_open.settings import DODO_SETTINGS, DODH_SETTINGS, EXPERIMENTS_DIRECTORY
from dreem_learning_open.utils.memmap_eeg import filter_memmap_signals_eeg_only, with_eeg_model_suffix

ALGO_BASE = "cnn_rnn"
CONFIG_ROOT = os.path.join("scripts", "base_experiments", ALGO_BASE)

DATASET_SETTINGS = {
    "dodh": DODH_SETTINGS,
    "dodo": DODO_SETTINGS,
}


def memmap_hash(memmap_description: dict) -> str:
    return hashlib.sha1(json.dumps(memmap_description).encode()).hexdigest()[:10]


def load_memmap_description(dataset: str, *, eeg_only: bool) -> dict:
    path = os.path.join(CONFIG_ROOT, "memmaps.json")
    with open(path, "r") as f:
        memmaps = json.load(f)
    for desc in memmaps:
        if desc.get("dataset") == dataset:
            out = dict(desc)
            out.pop("dataset", None)
            if eeg_only:
                out = filter_memmap_signals_eeg_only(out)
            return out
    raise RuntimeError("No memmap block for dataset={!r} in {}".format(dataset, path))


def get_dataset_dir(dataset: str, memmap_description: dict) -> str:
    return os.path.join(
        DATASET_SETTINGS[dataset]["memmap_directory"],
        memmap_hash(memmap_description),
    )


def memmap_ready(dataset_dir: str) -> bool:
    required = [
        os.path.join(dataset_dir, "groups_description.json"),
        os.path.join(dataset_dir, "features_description.json"),
        os.path.join(dataset_dir, "memmap_description.json"),
    ]
    return os.path.isdir(dataset_dir) and all(os.path.isfile(x) for x in required)


def compute_fold_count(dataset_dir: str) -> int:
    records = [name for name in os.listdir(dataset_dir) if ".json" not in name]
    rd.seed(2019)
    rd.shuffle(records)
    return len(records)


def runs_root_for(dataset: str, out_dir: str | None, algo: str) -> str:
    if out_dir:
        return out_dir
    return os.path.join(EXPERIMENTS_DIRECTORY, dataset, algo)


def load_completed_folds(runs_root: str) -> List[int]:
    jsonl = os.path.join(runs_root, "completed_folds.jsonl")
    if not os.path.isfile(jsonl):
        return []
    completed = set()
    with open(jsonl, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get("status") != "completed":
                continue
            idx = row.get("fold_idx")
            if isinstance(idx, int):
                completed.add(idx)
    return sorted(completed)


def partition_evenly(items: List[int], n_buckets: int) -> List[List[int]]:
    n_buckets = max(1, n_buckets)
    buckets = [[] for _ in range(n_buckets)]
    for i, item in enumerate(items):
        buckets[i % n_buckets].append(item)
    return [b for b in buckets if b]


def run_command(command: List[str], env: dict, log_path: str) -> int:
    with open(log_path, "w") as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        return proc.wait()


def start_streaming_process(
    command: List[str], env: dict, log_path: str
) -> Tuple[subprocess.Popen, threading.Thread]:
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        bufsize=0,
    )

    def _pump_output():
        assert proc.stdout is not None
        with open(log_path, "wb") as log:
            while True:
                chunk = proc.stdout.read(4096)
                if not chunk:
                    break
                log.write(chunk)
                log.flush()
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()

    t = threading.Thread(target=_pump_output, daemon=True)
    t.start()
    return proc, t


def run_workers(
    python_exec: str,
    workers: int,
    cuda_device: str,
    dataset: str,
    pending: List[int],
    round_idx: int,
    logs_dir: str,
    out_dir: str | None,
    eeg_only: bool,
) -> bool:
    batches = partition_evenly(pending, min(workers, len(pending)))
    procs: List[Tuple[subprocess.Popen, str, threading.Thread]] = []

    for worker_id, folds in enumerate(batches):
        cmd = [
            python_exec,
            os.path.join("scripts", "run_cnn_rnn.py"),
            "--dataset",
            dataset,
            "--folds",
            *[str(x) for x in folds],
        ]
        if out_dir:
            cmd += ["--out_dir", out_dir]
        if eeg_only:
            cmd.append("--eeg-only")

        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
        log_path = os.path.join(
            logs_dir, "worker_{}_round_{}.log".format(worker_id, round_idx)
        )
        print(
            "  worker {} -> folds {} (streaming to {})".format(
                worker_id, " ".join(str(x) for x in folds), log_path
            )
        )
        proc, pump_thread = start_streaming_process(cmd, env, log_path)
        procs.append((proc, log_path, pump_thread))

    ok = True
    for proc, log_path, pump_thread in procs:
        code = proc.wait()
        pump_thread.join()
        if code != 0:
            ok = False
            print("  worker failed (exit {}): {}".format(code, log_path))
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="dodh",
        choices=sorted(DATASET_SETTINGS.keys()),
        help="Dataset name",
    )
    parser.add_argument(
        "--workers", type=int, default=2, help="Number of parallel workers on one GPU"
    )
    parser.add_argument(
        "--cuda-device", default="0", help="CUDA device index to use (default: 0)"
    )
    parser.add_argument(
        "--max-rounds", type=int, default=50, help="Max scheduling rounds"
    )
    parser.add_argument(
        "--metric", default="cohen_kappa", help="Metric used by index_experiments"
    )
    parser.add_argument(
        "--python-exec",
        default=sys.executable,
        help="Python executable to run worker scripts (default: current interpreter)",
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="*",
        default=None,
        metavar="N",
        help="Optional subset of fold indices (default: all folds). Example: --folds 0 1 2",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Override experiment output dir (forwarded to run_cnn_rnn.py and indexing)",
    )
    parser.add_argument(
        "--rebuild-memmaps",
        action="store_true",
        help="Pass --rebuild-memmaps to the bootstrap run_cnn_rnn.py run only",
    )
    parser.add_argument(
        "--eeg-only",
        action="store_true",
        help="Train EEG-only models; runs live under cnn_rnn_eeg/ (same memmap hash as full CNN-RNN for dodh/dodo).",
    )
    args = parser.parse_args()

    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    logs_dir = "logs_cnn_rnn_single_gpu_parallel"
    os.makedirs(logs_dir, exist_ok=True)

    algo = with_eeg_model_suffix(ALGO_BASE, args.eeg_only)
    memmap_description = load_memmap_description(args.dataset, eeg_only=args.eeg_only)
    dataset_dir = get_dataset_dir(args.dataset, memmap_description)
    requested_folds = set(args.folds) if args.folds is not None else None
    runs_root = runs_root_for(args.dataset, args.out_dir, algo)

    for round_idx in range(1, args.max_rounds + 1):
        print("\n========== Round {}/{} ==========".format(round_idx, args.max_rounds))

        if not memmap_ready(dataset_dir):
            bootstrap_fold = 0
            if requested_folds:
                bootstrap_fold = min(requested_folds)
            print("Memmaps not ready -> bootstrap fold {}".format(bootstrap_fold))
            bootstrap_cmd = [
                args.python_exec,
                os.path.join("scripts", "run_cnn_rnn.py"),
                "--dataset",
                args.dataset,
                "--folds",
                str(bootstrap_fold),
            ]
            if args.out_dir:
                bootstrap_cmd += ["--out_dir", args.out_dir]
            if args.rebuild_memmaps:
                bootstrap_cmd += ["--rebuild-memmaps"]
            if args.eeg_only:
                bootstrap_cmd.append("--eeg-only")

            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
            bootstrap_log = os.path.join(logs_dir, "bootstrap_round_{}.log".format(round_idx))
            code = run_command(bootstrap_cmd, env, bootstrap_log)
            if code != 0:
                print("Bootstrap failed (exit {}). See {}".format(code, bootstrap_log))
                return 1
            continue

        index_cmd = [
            args.python_exec,
            "scripts/experiment_utils/index_experiments.py",
            "--dataset",
            args.dataset,
            "--algo",
            algo,
            "--base-experiments-dir",
            os.path.join("scripts", "base_experiments"),
            "--metric",
            args.metric,
            "--runs-root",
            runs_root,
        ]
        index_env = dict(os.environ)
        index_log = os.path.join(logs_dir, "index_round_{}.log".format(round_idx))
        code = run_command(index_cmd, index_env, index_log)
        if code != 0:
            print("Indexing failed (exit {}). See {}".format(code, index_log))
            return 1

        total_folds = compute_fold_count(dataset_dir)
        if requested_folds is not None:
            invalid = sorted(x for x in requested_folds if x < 0 or x >= total_folds)
            if invalid:
                raise ValueError(
                    "Requested folds out of range 0..{}: {}".format(total_folds - 1, invalid)
                )
        completed = load_completed_folds(runs_root)
        pending = [x for x in range(total_folds) if x not in set(completed)]
        if requested_folds is not None:
            pending = [x for x in pending if x in requested_folds]

        print("Total folds: {}".format(total_folds))
        if requested_folds is not None:
            print("Requested folds: {}".format(" ".join(str(x) for x in sorted(requested_folds))))
        print("Completed folds: {}".format(len(completed)))
        if not pending:
            print("All folds complete.")
            return 0
        print("Pending folds ({}): {}".format(len(pending), " ".join(str(x) for x in pending)))

        ok = run_workers(
            python_exec=args.python_exec,
            workers=args.workers,
            cuda_device=str(args.cuda_device),
            dataset=args.dataset,
            pending=pending,
            round_idx=round_idx,
            logs_dir=logs_dir,
            out_dir=args.out_dir,
            eeg_only=args.eeg_only,
        )
        if not ok:
            print("Some workers failed. Re-indexing next round and retrying remaining folds.")

    print("Reached --max-rounds={} before completion.".format(args.max_rounds))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
