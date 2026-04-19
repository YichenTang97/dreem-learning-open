"""Memar & Faradji (2018) classical EEG features + RF baseline (memar_et_al)."""

from dreem_learning_open.memar_et_al.config import get_eeg_signal, load_memar_et_al_config
from dreem_learning_open.memar_et_al.features import (
    FEATURE_DIM,
    build_feature_names,
    extract_memar_features_matrix,
    extract_memar_features_vector,
)

__all__ = [
    "FEATURE_DIM",
    "build_feature_names",
    "extract_memar_features_matrix",
    "extract_memar_features_vector",
    "get_eeg_signal",
    "load_memar_et_al_config",
]
