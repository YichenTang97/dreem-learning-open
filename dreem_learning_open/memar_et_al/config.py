"""Load ``memar_et_al_config.json`` from ``scripts/base_experiments/memar_et_al/``."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from dreem_learning_open.settings import REPO_ROOT

_CONFIG_REL = os.path.join("scripts", "base_experiments", "memar_et_al", "memar_et_al_config.json")


def default_memar_et_al_config_path() -> str:
    return os.path.join(REPO_ROOT, _CONFIG_REL)


def load_memar_et_al_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Parameters
    ----------
    config_path
        If None, uses ``REPO_ROOT/scripts/base_experiments/memar_et_al/memar_et_al_config.json``.
    """
    path = config_path or default_memar_et_al_config_path()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "eeg_signal" not in data:
        raise ValueError("memar_et_al config must be a JSON object with key 'eeg_signal'")
    if not isinstance(data["eeg_signal"], str) or not data["eeg_signal"].strip():
        raise ValueError("eeg_signal must be a non-empty string")
    return data


def get_eeg_signal(config_path: Optional[str] = None) -> str:
    return str(load_memar_et_al_config(config_path)["eeg_signal"]).strip()
