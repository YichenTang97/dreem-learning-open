"""
sol_metrics.py
==============
Core utilities for Sleep Onset Latency (SOL) computation and evaluation.

Stage encoding (Dreem convention):
    -1  : Not scored / unlabelled
     0  : Wake
     1  : N1
     2  : N2
     3  : N3
     4  : REM

SOL definition: time from the first scored epoch to the first epoch whose
label is strictly greater than 0 (any non-wake sleep stage).
Returned in **minutes** throughout unless otherwise noted.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

# torch is only required for the differentiable loss functions used during
# fine-tuning.  Import lazily so the pure-numpy SOL utilities work without a
# GPU / torch installation in environments where only inference is needed.
try:
    import torch
    _TORCH_AVAILABLE = True
except (ImportError, OSError):
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EPOCH_DURATION_S: int = 30          # standard PSG epoch length in seconds
SLEEP_LABELS: Tuple[int, ...] = (1, 2, 3, 4)   # N1, N2, N3, REM


# ---------------------------------------------------------------------------
# SOL computation from a single hypnogram array
# ---------------------------------------------------------------------------

def compute_sol(hypnogram: np.ndarray,
                epoch_duration_s: int = EPOCH_DURATION_S,
                require_consecutive: int = 1) -> Optional[float]:
    """
    Compute SOL in minutes from a 1-D integer hypnogram array.

    Parameters
    ----------
    hypnogram : array-like of int
        Sequence of sleep stage labels (-1, 0, 1, 2, 3, 4).
    epoch_duration_s : int
        Duration of each epoch in seconds (default 30).
    require_consecutive : int
        Minimum number of consecutive non-wake epochs required to confirm
        sleep onset (default 1, i.e. the AASM single-epoch rule).
        Setting this to 2 avoids artefactual single-epoch N1 predictions.

    Returns
    -------
    float or None
        SOL in minutes, or None if no sleep is found.
    """
    hyp = np.asarray(hypnogram, dtype=int)

    # Boolean mask: True where the stage is a sleep stage (non-wake, scored)
    is_sleep = np.isin(hyp, SLEEP_LABELS)

    if require_consecutive == 1:
        indices = np.where(is_sleep)[0]
        if len(indices) == 0:
            return None
        return float(indices[0]) * epoch_duration_s / 60.0

    # Require `require_consecutive` adjacent sleep epochs
    for i in range(len(is_sleep) - require_consecutive + 1):
        if all(is_sleep[i: i + require_consecutive]):
            return float(i) * epoch_duration_s / 60.0
    return None


# ---------------------------------------------------------------------------
# SOL metrics between a prediction and a reference
# ---------------------------------------------------------------------------

def sol_error(predicted: Optional[float],
              reference: Optional[float]) -> Optional[float]:
    """Signed error (predicted - reference) in minutes, or None if either is None."""
    if predicted is None or reference is None:
        return None
    return predicted - reference


def sol_abs_error(predicted: Optional[float],
                  reference: Optional[float]) -> Optional[float]:
    """Absolute error in minutes, or None if either is None."""
    err = sol_error(predicted, reference)
    return None if err is None else abs(err)


def compute_sol_metrics(predictions: Dict[str, Optional[float]],
                        references: Dict[str, Optional[float]]) -> Dict:
    """
    Compute aggregate SOL metrics across a set of recordings.

    Parameters
    ----------
    predictions : dict  {record_id -> predicted_sol_minutes or None}
    references  : dict  {record_id -> reference_sol_minutes  or None}

    Returns
    -------
    dict with keys:
        n_valid, mae, rmse, bias, std_error, pearson_r,
        per_record: dict {record_id -> {predicted, reference, error, abs_error}}
    """
    errors, abs_errors, preds_valid, refs_valid = [], [], [], []
    per_record = {}

    common_keys = set(predictions) & set(references)
    for rid in sorted(common_keys):
        pred = predictions[rid]
        ref  = references[rid]
        err  = sol_error(pred, ref)
        aerr = sol_abs_error(pred, ref)
        per_record[rid] = {
            "predicted_min": round(pred, 3)  if pred is not None else None,
            "reference_min": round(ref,  3)  if ref  is not None else None,
            "error_min":     round(err,  3)  if err  is not None else None,
            "abs_error_min": round(aerr, 3)  if aerr is not None else None,
        }
        if err is not None:
            errors.append(err)
            abs_errors.append(aerr)
            preds_valid.append(pred)
            refs_valid.append(ref)

    if len(errors) == 0:
        return {"n_valid": 0, "per_record": per_record}

    errors     = np.array(errors)
    abs_errors = np.array(abs_errors)
    preds_arr  = np.array(preds_valid)
    refs_arr   = np.array(refs_valid)

    # Pearson r (handle constant arrays gracefully)
    if np.std(preds_arr) < 1e-8 or np.std(refs_arr) < 1e-8:
        r = float("nan")
    else:
        r = float(np.corrcoef(preds_arr, refs_arr)[0, 1])

    return {
        "n_valid":    len(errors),
        "mae_min":    float(np.mean(abs_errors)),
        "rmse_min":   float(np.sqrt(np.mean(errors ** 2))),
        "bias_min":   float(np.mean(errors)),        # positive = over-estimate
        "std_err_min":float(np.std(errors)),
        "pearson_r":  round(r, 4),
        "per_record": per_record,
    }


# ---------------------------------------------------------------------------
# Differentiable SOL loss (for fine-tuning with backprop)
# ---------------------------------------------------------------------------

def soft_sol_minutes(logits: "torch.Tensor",
                     epoch_duration_s: int = EPOCH_DURATION_S) -> torch.Tensor:
    """
    Compute a differentiable *expected* SOL from per-epoch class logits.

    Formulation:
        P(sleep | t) = 1 - softmax(logits[t])[0]            (wake class = 0)
        P(SOL = t)   = P(still awake at 0..t-1) × P(sleep at t)
        E[SOL]       = Σ_t  t × P(SOL = t)

    This is a standard survival / discrete-time hazard model and is
    fully differentiable with respect to `logits`.

    Parameters
    ----------
    logits : Tensor of shape (T, n_class)
        Raw (un-softmaxed) class scores for T consecutive epochs.
    epoch_duration_s : int
        Epoch length in seconds.

    Returns
    -------
    Tensor scalar: expected SOL in minutes.
    """
    probs       = torch.softmax(logits, dim=-1)        # (T, n_class)
    wake_prob   = probs[:, 0].clamp(min=1e-7, max=1.0)  # (T,)
    sleep_prob  = 1.0 - wake_prob                        # (T,)

    # Log-domain cumulative product for numerical stability
    log_cum_wake = torch.cumsum(torch.log(wake_prob), dim=0)   # (T,)
    # still_awake[t] = P(wake at 0, 1, ..., t)
    still_awake  = torch.exp(log_cum_wake)                     # (T,)
    # Shift: P(still awake *before* epoch t) = [1, wake_0, wake_0*wake_1, ...]
    ones = torch.ones(1, device=logits.device, dtype=logits.dtype)
    still_awake_before = torch.cat([ones, still_awake[:-1]], dim=0)  # (T,)

    sol_prob_at_t = still_awake_before * sleep_prob  # (T,)
    t = torch.arange(len(logits), device=logits.device, dtype=logits.dtype)
    expected_sol_epochs = (sol_prob_at_t * t).sum()

    return expected_sol_epochs * epoch_duration_s / 60.0   # minutes


def sol_finetune_loss(logits: "torch.Tensor",
                      hypnogram: "torch.Tensor",
                      true_sol_epochs: int,
                      cutoff_epochs: int,
                      alpha: float = 0.5,
                      epoch_duration_s: int = EPOCH_DURATION_S) -> torch.Tensor:
    """
    Combined staging cross-entropy + differentiable SOL loss for fine-tuning.

    Only epochs in [0, true_sol_epochs + cutoff_epochs) are used.

    Parameters
    ----------
    logits       : Tensor (T, n_class)   — model output for the window
    hypnogram    : Tensor (T,) long      — true sleep stage labels
    true_sol_epochs : int                — true SOL expressed in epochs
    cutoff_epochs   : int                — how many epochs past SOL to include
    alpha        : float                 — weight of the staging CE loss
                                           (1-alpha) for the SOL loss
    epoch_duration_s : int

    Returns
    -------
    Tensor scalar: combined loss
    """
    end = min(true_sol_epochs + cutoff_epochs, len(logits))
    window_logits = logits[:end]         # (end, n_class)
    window_hyp    = hypnogram[:end]      # (end,)

    # --- Staging cross-entropy (ignore unlabelled epochs) ---
    mask = window_hyp >= 0
    if mask.sum() == 0:
        ce_loss = torch.tensor(0.0, device=logits.device)
    else:
        ce_loss = torch.nn.functional.cross_entropy(
            window_logits[mask], window_hyp[mask])

    # --- Differentiable SOL loss ---
    pred_sol_min = soft_sol_minutes(window_logits, epoch_duration_s)
    true_sol_min = torch.tensor(
        true_sol_epochs * epoch_duration_s / 60.0,
        device=logits.device, dtype=logits.dtype)
    sol_loss = torch.abs(pred_sol_min - true_sol_min)

    return alpha * ce_loss + (1.0 - alpha) * sol_loss


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_sol_targets(sol_targets_path: str) -> Dict[str, Dict]:
    """Load the JSON produced by compute_sol_targets.py."""
    with open(sol_targets_path) as f:
        return json.load(f)


def extract_consensus_sol(sol_targets: Dict[str, Dict]) -> Dict[str, Optional[float]]:
    """Return {record_id -> consensus_sol_minutes} from the sol_targets dict."""
    return {
        rid: info.get("consensus_sol_min")
        for rid, info in sol_targets.items()
    }


def sol_from_hypnograms_json(hypnograms_json_path: str,
                             require_consecutive: int = 1
                             ) -> Tuple[Dict[str, Optional[float]],
                                        Dict[str, Optional[float]]]:
    """
    Read a hypnograms.json file (output of log_experiment) and compute
    predicted and target SOLs for every test record.

    Returns
    -------
    (predicted_sols, target_sols) as {record_id -> sol_minutes or None}
    """
    with open(hypnograms_json_path) as f:
        hypnograms = json.load(f)

    predicted_sols, target_sols = {}, {}
    for rid, data in hypnograms.items():
        pred_hyp   = np.array(data["predicted"], dtype=int)
        target_hyp = np.array(data["target"],    dtype=int)
        predicted_sols[rid] = compute_sol(pred_hyp,   require_consecutive=require_consecutive)
        target_sols[rid]    = compute_sol(target_hyp, require_consecutive=require_consecutive)

    return predicted_sols, target_sols
