"""Load 30 s EEG epochs from Dreem memmap records (F4_O2 or chosen channel)."""
from __future__ import annotations

import json
import os
from typing import Iterator, List, Tuple

import numpy as np

from dreem_learning_open.memar_et_al.config import get_eeg_signal
from dreem_learning_open.memar_et_al.features import (
    eeg_signal_order_from_memmap_desc,
    channel_index_for_signal,
    extract_memar_features_vector,
    load_bands_config,
    precompute_band_sos_list,
)


def load_hypnogram(record_path: str) -> np.ndarray:
    p = os.path.join(record_path, "hypno.mm")
    h = np.memmap(p, dtype="float32", mode="r")
    return np.asarray(h, dtype=np.int64)


def epoch_iterator(
    record_path: str,
    memmap_description: dict,
    channel_signal: str | None = None,
    config_path: str | None = None,
) -> Iterator[Tuple[int, np.ndarray, int]]:
    """
    Yields (epoch_index, epoch_eeg_1d, stage_label) for each epoch in the record.

    Channel selection: ``channel_signal`` overrides ``memar_et_al_config.json``;
    if both are omitted, :func:`get_eeg_signal` is used.
    """
    sig = channel_signal if channel_signal is not None else get_eeg_signal(config_path)
    order = eeg_signal_order_from_memmap_desc(memmap_description)
    ch = channel_index_for_signal(order, sig)
    prop_path = os.path.join(record_path, "properties.json")
    with open(prop_path, "r", encoding="utf-8") as f:
        props = json.load(f)
    eeg_meta = props["eeg"]
    fs = int(eeg_meta["fs"])
    wl = int(fs * 30)
    shape = tuple(eeg_meta["shape"])
    hyp = load_hypnogram(record_path)
    n_epochs = int(hyp.shape[0])
    if shape[0] != n_epochs * wl:
        raise ValueError(
            "eeg length {} != n_epochs*wl {} for {}".format(shape[0], n_epochs * wl, record_path)
        )
    mm = np.memmap(
        os.path.join(record_path, "signals", "eeg.mm"),
        dtype="float32",
        mode="r",
        shape=shape,
    )
    for i in range(n_epochs):
        seg = np.asarray(mm[i * wl : (i + 1) * wl, ch], dtype=np.float64)
        yield i, seg, int(hyp[i])


def _gather_labeled_epochs_one_record(
    record_path: str,
    memmap_description: dict,
    channel_signal: str | None,
    config_path: str | None,
    bands: dict,
    fs: float,
    band_sos_list: list,
) -> Tuple[np.ndarray, np.ndarray]:
    """One training subject; module-level for ``joblib`` pickling (Windows spawn)."""
    xs: list = []
    ys: list = []
    for _i, ep, lab in epoch_iterator(
        record_path, memmap_description, channel_signal=channel_signal, config_path=config_path
    ):
        if lab < 0:
            continue
        xs.append(extract_memar_features_vector(ep, fs, bands, band_sos_list=band_sos_list))
        ys.append(lab)
    if not xs:
        return np.zeros((0, 104), dtype=np.float64), np.zeros((0,), dtype=np.int64)
    return np.vstack(xs), np.asarray(ys, dtype=np.int64)


def gather_labeled_epochs(
    record_paths: List[str],
    memmap_description: dict,
    channel_signal: str | None = None,
    config_path: str | None = None,
    show_progress: bool = False,
    progress_desc: str = "Train subjects (features)",
    feat_n_jobs: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stack (n_samples, n_features) and labels for all scored epochs (label >= 0)."""
    bands = load_bands_config()
    fs = float(bands["fs"])
    band_sos_list = precompute_band_sos_list(fs, bands)

    if feat_n_jobs != int(feat_n_jobs) or feat_n_jobs < 1:
        raise ValueError("feat_n_jobs must be a positive integer")

    if feat_n_jobs == 1 or len(record_paths) <= 1:
        xs = []
        ys = []
        pbar = None
        if show_progress:
            try:
                from tqdm import tqdm

                n_total = 0
                for rec in record_paths:
                    hyp = load_hypnogram(rec)
                    n_total += int(np.sum(hyp >= 0))
                pbar = tqdm(total=n_total, desc=progress_desc, unit="ep", leave=True)
            except ImportError:
                pbar = None
        for rec in record_paths:
            rec_short = os.path.basename(rec)
            if pbar is not None:
                pbar.set_postfix_str(rec_short[:24], refresh=False)
            for _i, ep, lab in epoch_iterator(
                rec, memmap_description, channel_signal=channel_signal, config_path=config_path
            ):
                if lab < 0:
                    continue
                xs.append(extract_memar_features_vector(ep, fs, bands, band_sos_list=band_sos_list))
                ys.append(lab)
                if pbar is not None:
                    pbar.update(1)
        if pbar is not None:
            pbar.close()
        if not xs:
            return np.zeros((0, 104), dtype=np.float64), np.zeros((0,), dtype=np.int64)
        return np.vstack(xs), np.asarray(ys, dtype=np.int64)

    from joblib import Parallel, delayed

    delayed_calls = [
        delayed(_gather_labeled_epochs_one_record)(
            rec,
            memmap_description,
            channel_signal,
            config_path,
            bands,
            fs,
            band_sos_list,
        )
        for rec in record_paths
    ]
    n_subj = len(record_paths)
    if show_progress:
        try:
            from tqdm import tqdm
            from tqdm.contrib.joblib import tqdm_joblib

            with tqdm_joblib(
                tqdm(
                    total=n_subj,
                    desc=progress_desc,
                    unit="subj",
                    leave=True,
                )
            ):
                parts = Parallel(n_jobs=feat_n_jobs, verbose=0)(delayed_calls)
        except ImportError:
            # No tqdm / tqdm_joblib: joblib verbose batch lines
            parts = Parallel(n_jobs=feat_n_jobs, verbose=10)(delayed_calls)
    else:
        parts = Parallel(n_jobs=feat_n_jobs, verbose=0)(delayed_calls)
    xs_arr = [p[0] for p in parts]
    ys_arr = [p[1] for p in parts]
    if all(x.shape[0] == 0 for x in xs_arr):
        return np.zeros((0, 104), dtype=np.float64), np.zeros((0,), dtype=np.int64)
    return np.vstack(xs_arr), np.concatenate(ys_arr)
