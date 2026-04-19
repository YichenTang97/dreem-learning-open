"""
Helpers for EEG-only memmap pipelines (drop non-EEG signal paths from memmap JSON).

Used when training with ``--eeg-only``: same ``scripts/base_experiments/<algo>/``
configs, but filtered ``signals`` lists so only paths under ``signals/eeg/`` remain.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List

# Suffix for experiment output folders under EXPERIMENTS_DIRECTORY/<dataset>/.
EEG_MODEL_SUFFIX = "_eeg"


def _is_eeg_signal_path(path: str) -> bool:
    p = path.replace("\\", "/")
    return "/eeg/" in p or p.startswith("signals/eeg")


def is_eeg_signal_path(path: str) -> bool:
    """True if ``path`` is under ``signals/eeg/`` (excludes EMG, ECG, EOG, … in the eeg memmap group)."""
    return _is_eeg_signal_path(path)


def _filter_signal_entries(entries: List[Any]) -> List[Any]:
    """Recursively filter a memmap ``signals`` list to EEG-only paths."""
    out: List[Any] = []
    for item in entries:
        if isinstance(item, str):
            if _is_eeg_signal_path(item):
                out.append(item)
        elif isinstance(item, dict):
            sub = item.get("signals")
            if isinstance(sub, list):
                filtered_sub = _filter_signal_entries(sub)
                if filtered_sub:
                    newd = copy.deepcopy(item)
                    newd["signals"] = filtered_sub
                    out.append(newd)
    return out


def filter_memmap_signals_eeg_only(memmap_description: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a deep copy of ``memmap_description`` with non-EEG channel paths removed.

    - For each entry in ``signals`` (top-level groups), only EEG paths are kept.
    - ``features`` entries that list ``signals`` keep only EEG paths (unchanged if already EEG-only).
    """
    out = copy.deepcopy(memmap_description)
    for group in out.get("signals", []):
        if isinstance(group, dict) and "signals" in group:
            group["signals"] = _filter_signal_entries(group["signals"])
    for feat in out.get("features", []):
        if isinstance(feat, dict) and "signals" in feat:
            sigs = feat["signals"]
            if isinstance(sigs, list):
                feat["signals"] = [s for s in sigs if isinstance(s, str) and _is_eeg_signal_path(s)]
    return out


def with_eeg_model_suffix(model_name: str, eeg_only: bool) -> str:
    """Append ``_eeg`` when ``eeg_only`` and the name does not already end with it."""
    if not eeg_only:
        return model_name
    if model_name.endswith(EEG_MODEL_SUFFIX):
        return model_name
    return model_name + EEG_MODEL_SUFFIX
