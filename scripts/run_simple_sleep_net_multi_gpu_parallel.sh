#!/usr/bin/env bash
set -euo pipefail

# Auto-resume multi-GPU launcher for Simple Sleep Net.
#
# What it does:
# 1) Detect visible GPU count (or use NUM_GPUS override)
# 2) Auto-detect whether memmaps are ready for simple_sleep_net/dodh
# 3) Refresh experiment index (completed_folds.jsonl / fold_summary) when memmaps exist
# 4) Read completed fold indices from completed_folds.jsonl
# 5) Evenly distribute pending folds across worker slots
#    (worker slots = GPU_COUNT * WORKERS_PER_GPU)
# 6) Repeat until all folds are complete
#
# Usage:
#   bash scripts/run_simple_sleep_net_multi_gpu_parallel.sh
#
# Optional env vars:
#   PYTHON=python3
#   NUM_GPUS=4
#   WORKERS_PER_GPU=2
#   MAX_ROUNDS=50
#   INDEX_METRIC=cohen_kappa
#   FOLDS="0 1 2"          # optional subset of fold indices to run
#
# Notes:
# - Uses scripts/run_simple_sleep_net_only.py as worker entrypoint
# - Workers always run with --no-force to avoid deleting sibling runs
# - Workers use --skip-memmap-build when memmaps are detected
# - On a single GPU, set NUM_GPUS=1 and tune WORKERS_PER_GPU for CPU-bound workloads

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-python}"
MAX_ROUNDS="${MAX_ROUNDS:-50}"
INDEX_METRIC="${INDEX_METRIC:-cohen_kappa}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-1}"
FOLDS="${FOLDS:-}"

if ! [[ "${WORKERS_PER_GPU}" =~ ^[0-9]+$ ]] || (( WORKERS_PER_GPU < 1 )); then
  echo "WORKERS_PER_GPU must be an integer >= 1 (got: ${WORKERS_PER_GPU})" >&2
  exit 1
fi

mkdir -p logs_multi_gpu_parallel

if [[ ! -f "scripts/run_simple_sleep_net_only.py" ]]; then
  echo "Missing scripts/run_simple_sleep_net_only.py" >&2
  exit 1
fi
if [[ ! -f "scripts/experiment_utils/index_experiments.py" ]]; then
  echo "Missing scripts/experiment_utils/index_experiments.py" >&2
  exit 1
fi

detect_gpu_count() {
  if [[ -n "${NUM_GPUS:-}" ]]; then
    echo "${NUM_GPUS}"
    return
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    local n
    n="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
    if [[ "${n}" =~ ^[0-9]+$ ]] && (( n > 0 )); then
      echo "${n}"
      return
    fi
  fi

  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a devs <<< "${CUDA_VISIBLE_DEVICES}"
    local n="${#devs[@]}"
    if (( n > 0 )); then
      echo "${n}"
      return
    fi
  fi

  echo "1"
}

GPU_COUNT="$(detect_gpu_count)"
TOTAL_WORKERS=$((GPU_COUNT * WORKERS_PER_GPU))
echo "Detected GPUs: ${GPU_COUNT}"
echo "Workers per GPU: ${WORKERS_PER_GPU}"
echo "Total worker slots: ${TOTAL_WORKERS}"

# Returns shell assignments:
#   MEMMAP_READY=0|1
#   TOTAL_FOLDS=<int>
#   COMPLETED_COUNT=<int>
#   PENDING_FOLDS="space separated ints"
compute_state() {
  "${PYTHON}" - <<'PY'
import hashlib
import json
import os
import random as rd
import shlex

from dreem_learning_open.settings import DODH_SETTINGS, EXPERIMENTS_DIRECTORY


def memmap_hash(memmap_description: dict) -> str:
    return hashlib.sha1(json.dumps(memmap_description).encode()).hexdigest()[:10]


algo = "simple_sleep_net"
exp_dir = os.path.join("scripts", "base_experiments", algo)
memmaps_path = os.path.join(exp_dir, "memmaps.json")

if not os.path.isfile(memmaps_path):
    print("MEMMAP_READY=0")
    print("TOTAL_FOLDS=0")
    print("COMPLETED_COUNT=0")
    print("PENDING_FOLDS=''")
    raise SystemExit(0)

with open(memmaps_path, "r") as f:
    memmaps_description = json.load(f)

description = None
for x in memmaps_description:
    if x.get("dataset") == "dodh":
        description = dict(x)
        description.pop("dataset", None)
        break

if description is None:
    print("MEMMAP_READY=0")
    print("TOTAL_FOLDS=0")
    print("COMPLETED_COUNT=0")
    print("PENDING_FOLDS=''")
    raise SystemExit(0)

dataset_dir = os.path.join(DODH_SETTINGS["memmap_directory"], memmap_hash(description))
required = [
    os.path.join(dataset_dir, "groups_description.json"),
    os.path.join(dataset_dir, "features_description.json"),
    os.path.join(dataset_dir, "memmap_description.json"),
]
memmap_ready = os.path.isdir(dataset_dir) and all(os.path.isfile(p) for p in required)

if not memmap_ready:
    print("MEMMAP_READY=0")
    print("TOTAL_FOLDS=0")
    print("COMPLETED_COUNT=0")
    print("PENDING_FOLDS=''")
    raise SystemExit(0)

records = [
    os.path.join(dataset_dir, name)
    for name in os.listdir(dataset_dir)
    if ".json" not in name
]
rd.seed(2019)
rd.shuffle(records)
fold_count = len(records)  # dodh LOOV
all_folds = list(range(fold_count))

runs_root = os.path.join(EXPERIMENTS_DIRECTORY, "dodh", algo)
completed = set()
jsonl_path = os.path.join(runs_root, "completed_folds.jsonl")
if os.path.isfile(jsonl_path):
    with open(jsonl_path, "r") as f:
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

pending = [i for i in all_folds if i not in completed]

print("MEMMAP_READY={}".format(shlex.quote("1")))
print("TOTAL_FOLDS={}".format(shlex.quote(str(fold_count))))
print("COMPLETED_COUNT={}".format(shlex.quote(str(len(completed)))))
print("PENDING_FOLDS={}".format(shlex.quote(" ".join(str(x) for x in pending))))
PY
}

filter_requested_folds() {
  local total_folds="$1"
  shift
  local pending_list=("$@")

  # No filter: passthrough
  if [[ -z "${FOLDS// }" ]]; then
    printf '%s\n' "${pending_list[@]}"
    return 0
  fi

  local -A requested_map=()
  for f in ${FOLDS}; do
    if ! [[ "${f}" =~ ^[0-9]+$ ]]; then
      echo "Invalid fold index in FOLDS: '${f}'" >&2
      return 1
    fi
    if (( f < 0 || f >= total_folds )); then
      echo "Fold index out of range in FOLDS: ${f} (valid: 0..$((total_folds - 1)))" >&2
      return 1
    fi
    requested_map["${f}"]=1
  done

  for p in "${pending_list[@]}"; do
    if [[ -n "${requested_map[$p]+x}" ]]; then
      printf '%s\n' "${p}"
    fi
  done
}

distribute_and_run_batches() {
  local pending_list=("$@")
  local n="${#pending_list[@]}"
  if (( n == 0 )); then
    return 0
  fi

  local slots="${TOTAL_WORKERS}"
  if (( slots > n )); then
    slots="${n}"
  fi

  local base=$((n / slots))
  local rem=$((n % slots))
  local idx=0
  local pids=()

  for ((slot = 0; slot < slots; slot++)); do
    local cnt
    if (( slot < rem )); then
      cnt=$((base + 1))
    else
      cnt=$base
    fi
    if (( cnt == 0 )); then
      continue
    fi

    local folds=()
    for ((k = 0; k < cnt; k++)); do
      folds+=("${pending_list[$idx]}")
      idx=$((idx + 1))
    done

    local gpu=$((slot % GPU_COUNT))
    local worker_on_gpu=$((slot / GPU_COUNT))
    echo "  slot ${slot} (gpu ${gpu}, worker ${worker_on_gpu}): folds ${folds[*]}"

    local log_file="logs_multi_gpu_parallel/gpu_${gpu}_worker_${worker_on_gpu}_round_${ROUND}.log"
    echo "    streaming to ${log_file}"
    # Stream live output to terminal (for tqdm progress bars) and persist to per-worker log.
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" scripts/run_simple_sleep_net_only.py \
      --no-force --reuse-incomplete-uuids --skip-memmap-build --folds "${folds[@]}" 2>&1 | tee "${log_file}" &
    pids+=("$!")
  done

  local ec=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      ec=1
    fi
  done
  return "${ec}"
}

ROUND=1
while (( ROUND <= MAX_ROUNDS )); do
  echo ""
  echo "========== Round ${ROUND}/${MAX_ROUNDS} =========="

  state_before="$(compute_state)"
  eval "${state_before}"

  if [[ "${MEMMAP_READY}" != "1" ]]; then
    echo "Memmaps not ready -> bootstrap run (fold 0, no-force)"
    if ! CUDA_VISIBLE_DEVICES=0 "${PYTHON}" scripts/run_simple_sleep_net_only.py --no-force --reuse-incomplete-uuids --folds 0 \
      >"logs_multi_gpu_parallel/bootstrap_round_${ROUND}.log" 2>&1; then
      echo "Bootstrap failed. See logs_multi_gpu_parallel/bootstrap_round_${ROUND}.log" >&2
      exit 1
    fi
    ROUND=$((ROUND + 1))
    continue
  fi

  if ! "${PYTHON}" scripts/experiment_utils/index_experiments.py --metric "${INDEX_METRIC}" \
    >"logs_multi_gpu_parallel/index_round_${ROUND}.log" 2>&1; then
    echo "Indexing failed. See logs_multi_gpu_parallel/index_round_${ROUND}.log" >&2
    exit 1
  fi

  state_after="$(compute_state)"
  eval "${state_after}"

  echo "Total folds: ${TOTAL_FOLDS}"
  echo "Completed folds: ${COMPLETED_COUNT}"
  if [[ -n "${FOLDS// }" ]]; then
    echo "Requested folds: ${FOLDS}"
  fi

  if [[ -z "${PENDING_FOLDS// }" ]]; then
    echo "All folds complete."
    exit 0
  fi

  read -r -a pending_array <<< "${PENDING_FOLDS}"
  if ! filtered_text="$(filter_requested_folds "${TOTAL_FOLDS}" "${pending_array[@]}")"; then
    exit 1
  fi
  mapfile -t filtered_pending <<< "${filtered_text}"
  if [[ "${#filtered_pending[@]}" -eq 0 ]]; then
    echo "No pending folds in requested subset."
    exit 0
  fi
  echo "Pending folds (${#filtered_pending[@]}): ${filtered_pending[*]}"

  if ! distribute_and_run_batches "${filtered_pending[@]}"; then
    echo "One or more workers failed in round ${ROUND}. Will re-index and retry pending folds."
  fi

  ROUND=$((ROUND + 1))
done

echo "Reached MAX_ROUNDS=${MAX_ROUNDS} before completion." >&2
echo "Check logs in logs_multi_gpu_parallel/ and re-run to continue." >&2
exit 1
