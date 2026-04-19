"""
finetune_sol.py  —  Script 4
==============================
Fine-tune a **pretrained** ModuloNet (any ``--base_model`` folder under
``EXPERIMENTS_DIRECTORY/<dataset>/``) with a combined staging + SOL loss.
Uses the same **LOOCV** folds as pretraining; outputs live under
``BASE_DIRECTORY/sol/finetuned/.../fold_XX/``.

Minimal usage (default base_model CNN-RNN on DODH, 10 min cutoff, alpha=0.5):
    python sol_experiments/finetune_sol.py

Custom usage:
    python sol_experiments/finetune_sol.py \\
        --base_model  simple_sleep_net \\
        --dataset     dodh \\
        --cutoff_minutes 15 \\
        --alpha 0.3 \\
        --lr 5e-5 \\
        --epochs 40 \\
        --patience 10 \\
        --folds 0 1 2 \\
        --out_dir /custom/finetuned/

Training protocol
-----------------
* LOOCV (same folds as the base experiment).
* Per training record: only the first (true_SOL + cutoff_minutes) epochs
  are used — the *peri-onset window*.
* Loss = alpha × CE(staging) + (1 − alpha) × |soft_SOL − true_SOL|
* Differentiable soft-SOL uses a survival-analysis expected-value formulation.
* Best checkpoint selected on validation SOL MAE.
* Fine-tuned hypnograms.json saved in the same format as base experiments,
  so evaluate_sol.py can compare results directly.

Resume: by default, skips a fold when ``fold_<NN>/finetune_results.json`` and
``hypnograms.json`` already exist. Use ``--force`` to re-run those folds.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sol_experiments.sol_config import (
    DATASET_SETTINGS,
    FINETUNE_DEFAULTS,
    exp_dir as default_exp_dir,
    finetune_dir as default_finetune_dir,
    finetuned_run_dir,
    normalize_base_model_for_finetune_tag,
    sol_targets_path as default_targets_path,
    print_config,
    to_base_directory_relative,
)
from sol_experiments.utils.sol_metrics import (
    load_sol_targets,
    extract_consensus_sol,
    compute_sol,
    sol_finetune_loss,
    compute_sol_metrics,
    EPOCH_DURATION_S,
)

import torch
from torch.optim import Adam

from dreem_learning_open.models.modulo_net.net import ModuloNet
from dreem_learning_open.datasets.dataset import DreemDataset
from dreem_learning_open.utils.train_test_val_split import train_test_val_split
from dreem_learning_open.utils.run_experiments import memmap_hash


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def get_record_sequential(
    dataset: DreemDataset, record: str
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor]:
    """
    Collect all epochs from `record` in sequential order.
    Returns (groups_cat, features_cat, hyp_cat) where hyp_cat has shape (n_epochs, TC).
    """
    groups_tensors: Dict[str, List[torch.Tensor]] = {g: [] for g in dataset.groups}
    features_tensors: Dict[str, List[torch.Tensor]] = {}
    hyp_list: List[torch.Tensor] = []

    for batch in dataset.get_record(record, batch_size=128, mode="eval"):
        for g in dataset.groups:
            groups_tensors[g].append(batch["groups"][g])
        for fname, fval in batch.get("features", {}).items():
            features_tensors.setdefault(fname, []).append(fval)
        hyp_list.append(batch["hypnogram"])

    groups_cat = {g: torch.cat(groups_tensors[g], dim=0) for g in dataset.groups}
    features_cat = {
        fname: torch.cat(fvals, dim=0) for fname, fvals in features_tensors.items() if fvals
    }
    hyp_cat    = torch.cat(hyp_list, dim=0)
    return groups_cat, features_cat, hyp_cat


def evaluate_record(net: ModuloNet, dataset: DreemDataset,
                    record: str) -> Tuple[List[int], List[int]]:
    """Run inference on `record`, return (predicted_labels, target_labels)."""
    net.eval()
    predicted = net.predict_on_record(record, dataset, return_prob=False)
    target    = dataset.hypnogram[record].astype(int).tolist()
    padding   = 0
    for g in dataset.groups_description:
        padding = dataset.groups_description[g].get("padding", 0) // 30
    if padding > 0:
        predicted = predicted[padding:-padding]
        target    = target[padding:-padding]
    return predicted.tolist(), target


# ---------------------------------------------------------------------------
# Fine-tuning step (one gradient update per training record)
# ---------------------------------------------------------------------------

def finetune_step(
        net: ModuloNet,
        optimizer: torch.optim.Optimizer,
        dataset: DreemDataset,
        record: str,
        true_sol_min: Optional[float],
        cutoff_minutes: float,
        alpha: float,
        epoch_duration_s: int = EPOCH_DURATION_S,
) -> Optional[float]:
    """One gradient update on the peri-onset window of a single record."""
    if true_sol_min is None:
        return None

    # Epoch index for peri-onset window only; SOL loss uses float consensus_sol_min below.
    true_sol_epochs = int(round(float(true_sol_min) * 60.0 / epoch_duration_s))
    cutoff_epochs   = int(round(cutoff_minutes * 60.0 / epoch_duration_s))
    window_end      = true_sol_epochs + cutoff_epochs

    device = net.device
    net.train()

    groups_cat, features_cat, hyp_cat = get_record_sequential(dataset, record)
    n_total = hyp_cat.shape[0]

    if window_end <= 0 or true_sol_epochs >= n_total:
        return None
    actual_end = min(window_end, n_total)
    tc = dataset.temporal_context

    # Collect per-epoch logits over the peri-onset window
    all_logits, all_labels = [], []
    for ep_idx in range(actual_end):
        g_batch = {
            "signals": {
                g: groups_cat[g][ep_idx].unsqueeze(0).to(device)
                for g in dataset.groups
            },
            "features": {
                fname: features_cat[fname][ep_idx].unsqueeze(0).to(device)
                for fname in features_cat
            },
        }
        logits, _ = net.forward(g_batch)
        # Handle both "one" and "many" output modes.
        # For "many", ModuloNet flattens temporal outputs to shape (B*TC, C);
        # with B=1 we recover the central epoch logits to match the central label.
        if logits.dim() == 3:
            logits = logits[:, tc // 2, :]
        elif logits.dim() == 2 and logits.shape[0] == tc:
            logits = logits[tc // 2].unsqueeze(0)
        all_logits.append(logits.squeeze(0))
        all_labels.append(hyp_cat[ep_idx, tc // 2])

    logits_seq = torch.stack(all_logits, dim=0)             # (actual_end, n_class)
    labels_seq = torch.stack(all_labels, dim=0).to(device)  # (actual_end,)

    optimizer.zero_grad()
    loss = sol_finetune_loss(
        logits=logits_seq, hypnogram=labels_seq,
        true_sol_minutes=true_sol_min,
        true_sol_epochs=true_sol_epochs, cutoff_epochs=cutoff_epochs,
        alpha=alpha, epoch_duration_s=epoch_duration_s,
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
    optimizer.step()
    return float(loss.detach().cpu())


# ---------------------------------------------------------------------------
# Per-fold fine-tuning + evaluation
# ---------------------------------------------------------------------------

def finetune_fold(
        fold_idx: int,
        fold_test: List[str],
        train_records: List[str],
        val_records: List[str],
        base_fold_dir: str,
        sol_targets: Dict,
        cutoff_minutes: float,
        alpha: float,
        lr: float,
        epochs: int,
        patience: int,
        save_folder: str,
        epoch_duration_s: int = EPOCH_DURATION_S,
) -> Dict:
    """Fine-tune one LOOCV fold and return per-fold metrics."""
    desc_path       = os.path.join(base_fold_dir, "description.json")
    best_model_path = os.path.join(base_fold_dir, "best_model.gz")
    if not os.path.exists(desc_path) or not os.path.exists(best_model_path):
        print(f"  [Fold {fold_idx}] Missing files in {base_fold_dir}. Skipping.")
        return {}

    with open(desc_path) as f:
        desc = json.load(f)

    groups_desc   = desc["groups_description"]
    features_desc = desc.get("features_description", {})
    ds_params     = desc["dataset_parameters"]
    tc    = ds_params["temporal_context"]
    mode  = ds_params.get("temporal_context_mode", "sequential")

    net = ModuloNet.load(best_model_path)
    net.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    optimizer = Adam(net.parameters(), lr=lr)

    def make_ds(records):
        return DreemDataset(groups_desc, features_description=features_desc,
                            temporal_context=tc, temporal_context_mode=mode,
                            records=records)

    ds_train = make_ds(train_records)
    ds_val   = make_ds(val_records)
    ds_test  = make_ds(fold_test)

    consensus_sols = extract_consensus_sol(sol_targets)

    def record_id(path):
        return os.path.splitext(os.path.basename(path))[0]

    def true_sol(path):
        return consensus_sols.get(record_id(path))

    # ---- Fine-tuning loop ----
    best_val_mae = float("inf")
    best_state   = copy.deepcopy(net.state_dict())
    patience_ctr = 0
    history      = []

    print(f"\n  [Fold {fold_idx}] cutoff={cutoff_minutes}min  "
          f"alpha={alpha}  lr={lr}  "
          f"train={len(train_records)} val={len(val_records)}")

    stop_reason = "max_epochs_reached"
    epochs_ran = 0
    for epoch in range(epochs):
        losses = []
        random.shuffle(train_records)
        for rec in train_records:
            lv = finetune_step(net, optimizer, ds_train, rec,
                               true_sol(rec), cutoff_minutes, alpha, epoch_duration_s)
            if lv is not None:
                losses.append(lv)

        mean_loss = float(np.mean(losses)) if losses else float("nan")

        # Validation SOL MAE
        net.eval()
        v_pred, v_ref = {}, {}
        for rec in val_records:
            rid = record_id(rec)
            pred_hyp, _ = evaluate_record(net, ds_val, rec)
            v_pred[rid] = compute_sol(np.array(pred_hyp), epoch_duration_s)
            v_ref[rid]  = true_sol(rec)
        val_mae = compute_sol_metrics(v_pred, v_ref).get("mae_min", float("inf"))

        is_best = val_mae < best_val_mae
        print(f"    ep {epoch:3d} | loss={mean_loss:.4f}  val_MAE={val_mae:.2f} min"
              + ("  ← best" if is_best else ""))
        history.append({"epoch": epoch, "train_loss": round(mean_loss, 4),
                        "val_mae_min": round(val_mae, 3) if val_mae != float("inf") else None})
        epochs_ran = epoch + 1

        if is_best:
            best_val_mae = val_mae
            best_state   = copy.deepcopy(net.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"    Early stop at epoch {epoch}.")
                stop_reason = "early_stop"
                break

    # ---- Test evaluation ----
    net.load_state_dict(best_state)
    net.eval()
    hypnograms_out, t_pred, t_ref = {}, {}, {}
    for rec in fold_test:
        rid = record_id(rec)
        pred_hyp, _target_hyp = evaluate_record(net, ds_test, rec)
        # Keep the same format as base experiments: predicted hypnogram only.
        hypnograms_out[rec] = pred_hyp
        t_pred[rid] = compute_sol(np.array(pred_hyp), epoch_duration_s)
        t_ref[rid]  = true_sol(rec)
    test_metrics = compute_sol_metrics(t_pred, t_ref)

    # ---- Save ----
    fold_dir = os.path.join(save_folder, f"fold_{fold_idx:02d}")
    os.makedirs(fold_dir, exist_ok=True)
    with open(os.path.join(fold_dir, "hypnograms.json"), "w") as f:
        json.dump(hypnograms_out, f, indent=2)
    net.save(os.path.join(fold_dir, "finetuned_model.gz"))

    result = {
        "fold_idx": fold_idx,
        "base_fold_dir": to_base_directory_relative(base_fold_dir),
        "finished_training": True,
        "stop_reason": stop_reason,
        "epochs_ran": epochs_ran,
        "requested_epochs": epochs,
        "requested_patience": patience,
        "requested_lr": lr,
        "requested_cutoff_minutes": cutoff_minutes,
        "requested_alpha": alpha,
        "best_val_mae": round(best_val_mae, 3),
        "test_sol_metrics": test_metrics, "training_history": history,
    }
    with open(os.path.join(fold_dir, "finetune_results.json"), "w") as f:
        json.dump(result, f, indent=2)

    mae = test_metrics.get("mae_min", float("nan"))
    bias = test_metrics.get("bias_min", float("nan"))
    print(f"  [Fold {fold_idx}] Test MAE={mae:.2f} min  bias={bias:+.2f} min")
    return result


# ---------------------------------------------------------------------------
# Locate trained fold directories in the base experiment folder
# ---------------------------------------------------------------------------

def find_fold_dirs(base_exp_dir: str) -> Dict[str, str]:
    """
    Return {test_record_basename: fold_directory} for all trained folds
    found under base_exp_dir.
    """
    result = {}
    for name in sorted(os.listdir(base_exp_dir)):
        full = os.path.join(base_exp_dir, name)
        if not os.path.isdir(full):
            continue
        desc  = os.path.join(full, "description.json")
        model = os.path.join(full, "best_model.gz")
        if not (os.path.exists(desc) and os.path.exists(model)):
            continue
        with open(desc) as f:
            d = json.load(f)
        test_recs = d.get("dataset_parameters", {}).get("split", {}).get("test", [])
        if test_recs:
            result[os.path.basename(test_recs[0])] = full
    return result


def partition_evenly(items: List[int], n_buckets: int) -> List[List[int]]:
    n_buckets = max(1, n_buckets)
    buckets = [[] for _ in range(n_buckets)]
    for i, item in enumerate(items):
        buckets[i % n_buckets].append(item)
    return [b for b in buckets if b]


def _resolve_gpu_slots(requested: Optional[List[int]]) -> List[int]:
    if requested:
        return requested
    if torch.cuda.is_available():
        return list(range(torch.cuda.device_count()))
    return []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    d = FINETUNE_DEFAULTS
    p = argparse.ArgumentParser(
        description="SOL fine-tuning of a pretrained staging model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--base_model", default=d["base_model"],
        help="Name of the base model to fine-tune "
             "(= folder under EXPERIMENTS_DIRECTORY/<dataset>/).",
    )
    p.add_argument(
        "--dataset", default=d["dataset"],
        choices=list(DATASET_SETTINGS.keys()),
        help="Dataset name (must match what the base model was trained on).",
    )
    p.add_argument(
        "--base_exp_dir", default=None,
        help="Override base experiment folder path. "
             "Default: EXPERIMENTS_DIRECTORY/<dataset>/<base_model>/",
    )
    p.add_argument(
        "--sol_targets", default=None,
        help="Path to SOL targets JSON. Supports either detailed "
             "sol_targets.json or consensus-only consensus_sol_targets.json. "
             "Default: BASE_DIRECTORY/sol/targets/<dataset>/sol_targets.json",
    )
    p.add_argument(
        "--out_dir", default=None,
        help="Output directory root for fine-tuned fold_XX/ trees. "
             "Default: BASE_DIRECTORY/sol/finetuned/<dataset>/<base_model>_ft_c<C>m_a<A>/",
    )
    p.add_argument(
        "--cutoff_minutes", type=float, default=d["cutoff_minutes"],
        help="Minutes after true SOL to include in the peri-onset training window.",
    )
    p.add_argument(
        "--alpha", type=float, default=d["alpha"],
        help="CE staging loss weight; (1-alpha) applied to SOL loss.",
    )
    p.add_argument("--lr",      type=float, default=d["lr"],      help="Fine-tuning learning rate.")
    p.add_argument("--epochs",  type=int,   default=d["epochs"],  help="Max fine-tuning epochs.")
    p.add_argument("--patience",type=int,   default=d["patience"],help="Early-stop patience (val SOL MAE).")
    p.add_argument(
        "--folds", nargs="+", type=int, default=d["folds"],
        help="Specific fold indices to fine-tune (default: all).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-run folds even if fold_<NN>/ already has finetune_results.json and hypnograms.json.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel worker processes. >1 runs folds in parallel.",
    )
    p.add_argument(
        "--gpus",
        type=int,
        nargs="*",
        default=None,
        metavar="ID",
        help="Optional CUDA GPU ids to use (default: all visible GPUs).",
    )
    p.add_argument(
        "--_worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return p


def _is_completed_finetune_result(result: Dict, args: argparse.Namespace) -> bool:
    """
    A fold is reusable only if training finished cleanly and with matching config.
    """
    if not isinstance(result, dict):
        return False
    if result.get("finished_training") is not True:
        return False
    if result.get("stop_reason") not in {"early_stop", "max_epochs_reached"}:
        return False
    epochs_ran = result.get("epochs_ran")
    if not isinstance(epochs_ran, int) or epochs_ran <= 0:
        return False

    expected = {
        "requested_epochs": args.epochs,
        "requested_patience": args.patience,
        "requested_lr": args.lr,
        "requested_cutoff_minutes": args.cutoff_minutes,
        "requested_alpha": args.alpha,
    }
    for key, value in expected.items():
        if result.get(key) != value:
            return False
    return True


def run_finetune(args: argparse.Namespace) -> None:
    resolved_base = args.base_exp_dir or default_exp_dir(args.dataset, args.base_model)
    resolved_sol  = args.sol_targets  or default_targets_path(args.dataset)
    resolved_out  = args.out_dir      or default_finetune_dir(
        args.dataset, args.base_model, args.cutoff_minutes, args.alpha
    )

    print_config("finetune_sol.py", {
        "base_model":      args.base_model,
        "dataset":         args.dataset,
        "base_exp_dir":    resolved_base,
        "sol_targets":     resolved_sol,
        "out_dir":         resolved_out,
        "cutoff_minutes":  args.cutoff_minutes,
        "alpha (CE wt)":   args.alpha,
        "sol loss weight": round(1 - args.alpha, 2),
        "lr / epochs":     f"{args.lr} / {args.epochs}",
        "patience":        args.patience,
        "folds":           args.folds or "all",
        "force":           args.force,
    })

    # ---- Validate ----
    if not os.path.isdir(resolved_base):
        print(f"ERROR: base_exp_dir not found: {resolved_base}")
        print("  Run scripts/run_cnn_rnn.py (or run_base_experiments.py) first.")
        sys.exit(1)
    if not os.path.exists(resolved_sol):
        print(f"ERROR: sol_targets not found: {resolved_sol}")
        print("  Run compute_sol_targets.py first.")
        sys.exit(1)

    os.makedirs(resolved_out, exist_ok=True)
    sol_targets = load_sol_targets(resolved_sol)

    # ---- Discover trained fold directories and rebuild LOOCV splits ----
    fold_map = find_fold_dirs(resolved_base)
    if not fold_map:
        print(f"ERROR: No trained folds found in {resolved_base}")
        sys.exit(1)
    print(f"Found {len(fold_map)} trained fold(s).\n")

    # Reconstruct the global record list and LOOCV split from the first fold
    first_fold_dir = next(iter(fold_map.values()))
    with open(os.path.join(first_fold_dir, "description.json")) as f:
        first_desc = json.load(f)

    memmap_desc      = first_desc["memmap_description"]
    dataset_settings = DATASET_SETTINGS[args.dataset]
    memmaps_dir = os.path.join(
        dataset_settings["memmap_directory"], memmap_hash(memmap_desc)
    )
    if not os.path.isdir(memmaps_dir):
        print(f"ERROR: memmaps directory not found: {memmaps_dir}")
        sys.exit(1)

    all_records = [
        os.path.join(memmaps_dir, r)
        for r in os.listdir(memmaps_dir)
        if ".json" not in r
    ]
    random.seed(2019)
    random.shuffle(all_records)
    all_folds = [[r] for r in all_records]  # LOOCV

    # Match fold directories → fold indices by test-record name
    fold_dir_to_idx = {}
    for test_basename, fold_dir in fold_map.items():
        for i, fold in enumerate(all_folds):
            if os.path.basename(fold[0]) == test_basename:
                fold_dir_to_idx[fold_dir] = i
                break

    effective = set(args.folds) if args.folds else set(range(len(all_folds)))
    all_results = []

    for fold_dir, fold_idx in sorted(fold_dir_to_idx.items(), key=lambda x: x[1]):
        if fold_idx not in effective:
            continue
        fold_out = os.path.join(resolved_out, f"fold_{fold_idx:02d}")
        done_json = os.path.join(fold_out, "finetune_results.json")
        done_hyp  = os.path.join(fold_out, "hypnograms.json")
        if not args.force and os.path.isfile(done_json) and os.path.isfile(done_hyp):
            can_reuse = False
            done_payload = None
            try:
                with open(done_json) as f:
                    done_payload = json.load(f)
                can_reuse = _is_completed_finetune_result(done_payload, args)
            except Exception:
                can_reuse = False

            if can_reuse:
                print(f"  [Fold {fold_idx}] already complete → {fold_out}")
                all_results.append(done_payload)
                continue
            print(f"  [Fold {fold_idx}] existing artifacts are incomplete/stale; re-running.")

        test_fold = all_folds[fold_idx]
        other     = [r for r in all_records if r not in test_fold]
        random.seed(2019 + fold_idx)
        random.shuffle(other)
        train_recs, val_recs, _ = train_test_val_split(other, 0.8, 0.2, 0.0, seed=2019)

        result = finetune_fold(
            fold_idx=fold_idx, fold_test=test_fold,
            train_records=train_recs, val_records=val_recs,
            base_fold_dir=fold_dir,
            sol_targets=sol_targets,
            cutoff_minutes=args.cutoff_minutes, alpha=args.alpha,
            lr=args.lr, epochs=args.epochs, patience=args.patience,
            save_folder=resolved_out,
        )
        all_results.append(result)

    # ---- Aggregate summary ----
    valid_maes = [
        r["test_sol_metrics"]["mae_min"]
        for r in all_results
        if r.get("test_sol_metrics", {}).get("mae_min") is not None
    ]
    print(f"\n{'='*65}")
    print(f"  FINE-TUNING COMPLETE")
    print(f"{'='*65}")
    if valid_maes:
        print(f"  Test SOL MAE : {np.mean(valid_maes):.2f} ± {np.std(valid_maes):.2f} min "
              f"over {len(valid_maes)} fold(s)")
    print(f"  Results in   : {resolved_out}")
    sub_name = os.path.basename(os.path.normpath(resolved_out))
    default_ft = finetuned_run_dir(args.dataset, sub_name)
    print(f"\n  Next: evaluate fine-tuned hypnograms vs expert SOL (rollup):")
    if os.path.abspath(resolved_out) == os.path.abspath(default_ft):
        print(f"    python sol_experiments/evaluate_sol.py \\")
        print(f"        --dataset {args.dataset} \\")
        print(f"        --model {sub_name} \\")
        print(f"        --finetuned\n")
    else:
        print(f"    python sol_experiments/evaluate_sol.py \\")
        print(f"        --dataset {args.dataset} \\")
        print(f"        --model {sub_name} \\")
        print(f"        --finetuned \\")
        print(f"        --exp_dir {resolved_out}\n")

    summary = {
        "base_model": args.base_model, "dataset": args.dataset,
        "cutoff_minutes": args.cutoff_minutes, "alpha": args.alpha,
        "lr": args.lr, "epochs": args.epochs, "patience": args.patience,
        "workers": args.workers,
        "gpus": args.gpus,
        "mean_test_sol_mae_min": float(np.mean(valid_maes)) if valid_maes else None,
        "std_test_sol_mae_min":  float(np.std(valid_maes))  if valid_maes else None,
        "fold_results": all_results,
    }
    with open(os.path.join(resolved_out, "finetune_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


def main(args: argparse.Namespace) -> None:
    bm = normalize_base_model_for_finetune_tag(args.base_model)
    if bm != args.base_model:
        print(
            "NOTE: --base_model {!r} -> {!r} (use the pretrained folder name under "
            "EXPERIMENTS_DIRECTORY/<dataset>/, not the finetuned output subfolder).".format(
                args.base_model, bm
            )
        )
        args.base_model = bm

    # Worker mode runs the core logic directly.
    if args._worker:
        run_finetune(args)
        return

    # Single worker: keep current behavior.
    if args.workers <= 1:
        run_finetune(args)
        return

    # Multi-worker scheduler: partition folds and launch subprocess workers.
    if args.folds:
        all_target_folds = sorted(set(args.folds))
    else:
        # We need fold count to partition all folds deterministically.
        resolved_base = args.base_exp_dir or default_exp_dir(args.dataset, args.base_model)
        fold_map = find_fold_dirs(resolved_base)
        all_target_folds = sorted(range(len(fold_map)))

    gpu_slots = _resolve_gpu_slots(args.gpus)
    n_workers = min(args.workers, len(all_target_folds))
    fold_batches = partition_evenly(all_target_folds, n_workers)

    print("\nLaunching {} parallel worker(s) for {} fold(s).".format(
        len(fold_batches), len(all_target_folds)
    ))
    if gpu_slots:
        print("GPU slots: {}".format(" ".join(str(x) for x in gpu_slots)))
    else:
        print("No CUDA GPUs selected/available; workers run on CPU.")

    procs: List[Tuple[subprocess.Popen, List[int], Optional[int]]] = []
    for worker_idx, batch in enumerate(fold_batches):
        cmd = [
            sys.executable,
            "sol_experiments/finetune_sol.py",
            "--base_model", args.base_model,
            "--dataset", args.dataset,
            "--cutoff_minutes", str(args.cutoff_minutes),
            "--alpha", str(args.alpha),
            "--lr", str(args.lr),
            "--epochs", str(args.epochs),
            "--patience", str(args.patience),
            "--workers", "1",
            "--_worker",
            "--folds",
            *[str(x) for x in batch],
        ]
        if args.base_exp_dir is not None:
            cmd.extend(["--base_exp_dir", args.base_exp_dir])
        if args.sol_targets is not None:
            cmd.extend(["--sol_targets", args.sol_targets])
        if args.out_dir is not None:
            cmd.extend(["--out_dir", args.out_dir])
        if args.force:
            cmd.append("--force")

        env = dict(os.environ)
        assigned_gpu: Optional[int] = None
        if gpu_slots:
            assigned_gpu = gpu_slots[worker_idx % len(gpu_slots)]
            env["CUDA_VISIBLE_DEVICES"] = str(assigned_gpu)

        print("  worker {} -> folds {}{}".format(
            worker_idx,
            " ".join(str(x) for x in batch),
            "" if assigned_gpu is None else " | gpu {}".format(assigned_gpu),
        ))
        proc = subprocess.Popen(cmd, env=env)
        procs.append((proc, batch, assigned_gpu))

    failed = False
    for proc, batch, assigned_gpu in procs:
        code = proc.wait()
        if code != 0:
            failed = True
            print("Worker failed (exit={} folds={} gpu={})".format(
                code, batch, assigned_gpu
            ))

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main(build_parser().parse_args())
