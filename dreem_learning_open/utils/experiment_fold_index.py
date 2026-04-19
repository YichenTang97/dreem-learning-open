"""
LOOCV fold map and test-subject recovery for experiment runs (shared by
``scripts/experiment_utils/index_experiments.py`` and
``scripts/experiment_utils/cleanup_incomplete_experiments.py``).
"""
from __future__ import annotations

import hashlib
import json
import os
import random as rd
from typing import Dict, Optional, Tuple

from dreem_learning_open.utils.memmap_eeg import EEG_MODEL_SUFFIX, filter_memmap_signals_eeg_only


def memmap_hash(memmap_description: dict) -> str:
    return hashlib.sha1(json.dumps(memmap_description).encode()).hexdigest()[:10]


def load_memmap_description(base_experiments_dir: str, algo: str, dataset: str) -> dict:
    """
    Load memmap JSON for ``algo`` and ``dataset``.

    If ``algo`` ends with ``_eeg`` (e.g. ``simple_sleep_net_eeg``), configs are read from
    the base name (``simple_sleep_net``) and the description is filtered to EEG-only
    signals so the memmap hash matches EEG-only training runs.
    """
    config_algo = algo
    eeg_only = False
    if algo.endswith(EEG_MODEL_SUFFIX) and len(algo) > len(EEG_MODEL_SUFFIX):
        config_algo = algo[: -len(EEG_MODEL_SUFFIX)]
        eeg_only = True

    memmaps_path = os.path.join(base_experiments_dir, config_algo, "memmaps.json")
    with open(memmaps_path, "r") as f:
        memmaps_description = json.load(f)

    for description in memmaps_description:
        if description.get("dataset") == dataset:
            description = dict(description)
            del description["dataset"]
            if eeg_only:
                description = filter_memmap_signals_eeg_only(description)
            return description
    raise RuntimeError(
        "No memmap block for dataset={!r} in {}".format(dataset, memmaps_path)
    )


def build_loov_fold_map(dataset_setting: dict, memmap_description: dict) -> Dict[str, int]:
    dataset_dir = os.path.join(
        dataset_setting["memmap_directory"], memmap_hash(memmap_description)
    )
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError("Memmap directory does not exist: {!r}".format(dataset_dir))

    records = [
        os.path.join(dataset_dir, record_name)
        for record_name in os.listdir(dataset_dir)
        if ".json" not in record_name
    ]
    rd.seed(2019)
    rd.shuffle(records)

    return {os.path.basename(record): idx for idx, record in enumerate(records)}


def recover_test_record_and_fold_idx(
    description: dict, fold_map: Dict[str, int]
) -> Tuple[Optional[str], Optional[int]]:
    """
    LOOV test subject id and fold index from description, even when the run is
    incomplete (no metrics / hypnograms yet).
    """
    test_record: Optional[str] = None
    rs = description.get("records_split")
    if isinstance(rs, dict):
        tr = rs.get("test_records")
        if isinstance(tr, list) and len(tr) == 1 and isinstance(tr[0], str):
            test_record = tr[0]

    if test_record is not None and (
        os.sep in test_record or "/" in test_record or "\\" in test_record
    ):
        test_record = os.path.basename(os.path.normpath(test_record))

    if test_record is None or test_record not in fold_map:
        dataset_parameters = description.get("dataset_parameters")
        if not isinstance(dataset_parameters, dict):
            dataset_parameters = {}
        split = dataset_parameters.get("split")
        if not isinstance(split, dict):
            split = {}
        split_test = split.get("test")
        if isinstance(split_test, list) and len(split_test) == 1 and isinstance(
            split_test[0], str
        ):
            recovered = os.path.basename(os.path.normpath(split_test[0]))
            if recovered in fold_map:
                test_record = recovered

    if test_record is None or test_record not in fold_map:
        return None, None
    return test_record, fold_map[test_record]
