"""
Memar & Faradji (IEEE TNSRE 2018) Section IV: 13 features × 8 bands = 104 features per epoch.

Single-channel subband signals x(n) are obtained with 14th-order Butterworth filters
(see bands.json). F4_O2 channel only upstream.
"""
from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from scipy import signal
from scipy.special import digamma, gamma

from dreem_learning_open.memar_et_al.config import get_eeg_signal
from dreem_learning_open.settings import REPO_ROOT
from dreem_learning_open.utils.memmap_eeg import is_eeg_signal_path

# Memar et al.: 13 features × 8 bands = 104. Hjorth Activity σ_x² equals SD², so we keep SD + HM + HC.
FEATURE_DIM = 104

# Kraskov uses fixed embedding dimension m=3 (paper); log(volume) is constant — compute once.
_VOL_UNIT_BALL_3 = math.pi ** 1.5 / gamma(2.5)


def _default_bands_path() -> str:
    return os.path.join(
        REPO_ROOT, "scripts", "base_experiments", "memar_et_al", "bands.json"
    )


def load_bands_config(path: str | None = None) -> Dict[str, Any]:
    p = path or _default_bands_path()
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def build_feature_names(bands_config: Dict[str, Any] | None = None) -> List[str]:
    cfg = bands_config or load_bands_config()
    names: List[str] = []
    base = [
        "sd",
        "hjorth_mobility",
        "hjorth_complexity",
        "mmd",
        "pfd",
        "nll",
        "ghe",
        "lrssv",
        "nse",
        "renyi_m2",
        "kraskov_entropy",
        "phase_mean",
        "phase_std",
    ]
    for b in cfg["bands_hz"]:
        bid = b["id"]
        for f in base:
            names.append("{}_{}".format(bid, f))
    assert len(names) == FEATURE_DIM
    return names


def _sos_filter_epoch(
    x: np.ndarray,
    fs: float,
    low_hz: float,
    high_hz: float,
    order: int,
    lowpass_only: bool,
    sos_precomputed: np.ndarray | None = None,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    if sos_precomputed is not None:
        return signal.sosfiltfilt(sos_precomputed, x)
    nyq = fs / 2.0
    if lowpass_only:
        wn = min(high_hz, nyq - 0.25)
        if wn <= 0:
            return np.zeros_like(x)
        sos = signal.butter(order, wn, btype="lowpass", fs=fs, output="sos")
    else:
        lo = max(low_hz, 1e-3)
        hi = min(high_hz, nyq - 0.25)
        if lo >= hi:
            return np.zeros_like(x)
        sos = signal.butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x)


def precompute_band_sos_list(fs: float, bands_config: Dict[str, Any]) -> List[np.ndarray | None]:
    """
    Build Butterworth SOS once per band (same filters for every epoch). ``None`` means
    degenerate band → zeros (matches :func:`_sos_filter_epoch`).
    """
    order = 14
    out: List[np.ndarray | None] = []
    nyq = fs / 2.0
    for b in bands_config["bands_hz"]:
        low_hz = float(b["low_hz"])
        high_hz = float(b["high_hz"])
        lowpass_only = bool(b.get("lowpass_only", False))
        if lowpass_only:
            wn = min(high_hz, nyq - 0.25)
            if wn <= 0:
                out.append(None)
                continue
            sos = signal.butter(order, wn, btype="lowpass", fs=fs, output="sos")
        else:
            lo = max(low_hz, 1e-3)
            hi = min(high_hz, nyq - 0.25)
            if lo >= hi:
                out.append(None)
                continue
            sos = signal.butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
        out.append(sos)
    return out


def _feat_sd(x: np.ndarray, sx_pre: float | None = None) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size < 2:
        return 0.0
    if sx_pre is not None:
        return float(sx_pre)
    return float(np.std(x, ddof=1))


def _feat_hjorth_mobility_complexity(
    x: np.ndarray,
    d1: np.ndarray | None = None,
    d2: np.ndarray | None = None,
    sx: float | None = None,
) -> Tuple[float, float]:
    """Hjorth HM and HC; Activity σ_x² omitted (redundant with SD²).

    Optional ``d1``, ``d2``, ``sx`` avoid redundant ``np.diff`` / ``np.std`` when the
    caller already computed them for the same band signal.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size < 3:
        return 0.0, 0.0
    if d1 is None:
        d1 = np.diff(x)
    if d2 is None:
        d2 = np.diff(d1)
    if sx is None:
        sx = float(np.std(x, ddof=1))
    else:
        sx = float(sx)
    s1 = float(np.std(d1, ddof=1)) if d1.size > 1 else 0.0
    s2 = float(np.std(d2, ddof=1)) if d2.size > 1 else 0.0
    hm = float(s1 / sx) if sx > 1e-15 else 0.0
    r1 = s2 / s1 if s1 > 1e-15 else 0.0
    r0 = s1 / sx if sx > 1e-15 else 0.0
    hc = float(r1 / r0) if r0 > 1e-15 else 0.0
    return hm, hc


def _feat_mmd(
    x: np.ndarray,
    win: int = 100,
    step: int = 50,
) -> float:
    """Sliding-window max–min Pythagorean distance sum (Eqs. 6–7)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    if n < 2:
        return 0.0
    if n <= win:
        idx_max = int(np.argmax(x))
        idx_min = int(np.argmin(x))
        dx = float(abs(idx_max - idx_min))
        dy = float(abs(x[idx_max] - x[idx_min]))
        return float(math.sqrt(dx * dx + dy * dy))
    n_stride = n - win + 1
    try:
        from numpy.lib.stride_tricks import sliding_window_view
    except ImportError:
        sliding_window_view = None  # type: ignore[assignment]
    if sliding_window_view is not None and n_stride >= 1:
        sw = sliding_window_view(x, win)
        W = sw[::step]
        if W.size == 0:
            return 0.0
        idx_max = np.argmax(W, axis=1)
        idx_min = np.argmin(W, axis=1)
        dx = np.abs(idx_max.astype(np.float64) - idx_min.astype(np.float64))
        rr = np.arange(W.shape[0], dtype=np.intp)
        dy = np.abs(W[rr, idx_max] - W[rr, idx_min])
        return float(np.sqrt(dx * dx + dy * dy).sum())
    total = 0.0
    k = 0
    while True:
        sl = x[k : k + win]
        if sl.size < 2:
            break
        idx_max = int(np.argmax(sl))
        idx_min = int(np.argmin(sl))
        dx = float(abs(idx_max - idx_min))
        dy = float(abs(sl[idx_max] - sl[idx_min]))
        total += math.sqrt(dx * dx + dy * dy)
        k += step
        if k + win > n:
            break
    return float(total)


def _feat_pfd(x: np.ndarray, dx: np.ndarray | None = None) -> float:
    """Petrosian FD (Eq. 8). N = samples; M = sign changes in derivative."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    if n < 3:
        return 0.0
    if dx is None:
        dx = np.diff(x)
    sgn = np.sign(dx)
    sgn = sgn[sgn != 0]
    if sgn.size < 2:
        m = 0
    else:
        m = int(np.sum(sgn[1:] * sgn[:-1] < 0))
    num = math.log10(float(n))
    den = num + math.log10(float(n) / (n + 0.4 * m))
    if den <= 0 or not math.isfinite(den):
        return 0.0
    return float(num / den)


def _feat_nll(x: np.ndarray, dx: np.ndarray | None = None) -> float:
    """NLL with M=1 and window length = epoch (Eq. 9)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size < 2:
        return 0.0
    if dx is None:
        dx = np.diff(x)
    return float(np.sum(np.abs(dx)))


def _feat_ghe(x: np.ndarray) -> float:
    """GHE with m=1, cumulative-sum time series, d in [5,19] (Eq. 10)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    cs = np.cumsum(x)
    n = cs.size
    den = float(np.mean(np.abs(cs))) + 1e-15
    ds = list(range(5, 20))
    ratios = []
    log_ds = []
    for d in ds:
        if n <= d:
            continue
        a = cs[d:] - cs[:-d]
        num = np.mean(np.abs(a))
        r = num / den
        if r > 0:
            ratios.append(math.log(r))
            log_ds.append(math.log(float(d)))
    if len(ratios) < 2:
        return 0.0
    slope, _ = np.polyfit(log_ds, ratios, 1)
    return float(slope)


def _feat_lrssv(x: np.ndarray, dx: np.ndarray | None = None) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size < 2:
        return 0.0
    if dx is None:
        dx = np.diff(x)
    s = float(np.sum(dx * dx))
    if s <= 0:
        return -10.0
    return float(math.log10(math.sqrt(s)))


def _feat_nse(x: np.ndarray, fs: float, f_lo: float, f_hi: float) -> float:
    """Normalized spectral entropy on Welch PSD; bins restricted to band (Eq. 12)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    if n < 32:
        return 0.0
    nperseg = min(512, n)
    f, pxx = signal.welch(x, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    m = (f >= f_lo) & (f <= f_hi)
    s = pxx[m].astype(np.float64)
    s_sum = float(np.sum(s))
    if s_sum <= 0:
        return 0.0
    s = s / s_sum
    s = np.clip(s, 1e-15, 1.0)
    nf = int(np.sum(m))
    if nf <= 1:
        return 0.0
    h = float(np.sum(s * np.log2(1.0 / s)))
    return h / math.log2(float(nf))


def _feat_renyi_m2(x: np.ndarray, n_bins: int | None = None) -> float:
    """Rényi entropy order m=2 (Eq. 13): -log2(sum p_i^2)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    if n < 2:
        return 0.0
    nb = n_bins or max(5, int(min(64, round(math.sqrt(n)))))
    hist, _ = np.histogram(x, bins=nb, density=False)
    s = float(np.sum(hist))
    if s <= 0:
        return 0.0
    p = (hist.astype(np.float64) / s)[hist > 0]
    if p.size == 0:
        return 0.0
    return float(-math.log2(float(np.dot(p, p))))


def _volume_m_ball(m: int) -> float:
    return math.pi ** (m / 2.0) / gamma(m / 2.0 + 1.0)


def _feat_kraskov(x: np.ndarray, embed_m: int = 3, tau: int = 1) -> float:
    """Kraskov entropy (Eq. 14); k = round(sqrt(N)) with N = epoch sample count."""
    from scipy.spatial import cKDTree

    x = np.asarray(x, dtype=np.float64).ravel()
    n_len = x.size
    k = max(1, int(round(math.sqrt(float(n_len)))))
    i0 = (embed_m - 1) * tau
    if n_len < i0 + 10:
        return 0.0
    # Vectorized delay embedding (same as former Python loop over i in range(i0, n_len)).
    X = np.column_stack([x[(i0 - j * tau) : (n_len - j * tau)] for j in range(embed_m)])
    n, m = X.shape
    if n < k + 2:
        return 0.0
    tree = cKDTree(X)
    try:
        dists, _ = tree.query(X, k=k + 1, workers=1)
    except TypeError:
        # scipy < 1.6: no workers= kwarg
        dists, _ = tree.query(X, k=k + 1)
    r = dists[:, k]
    r = np.maximum(r, 1e-12)
    vm = _VOL_UNIT_BALL_3 if m == 3 else _volume_m_ball(m)
    ke = float(
        -digamma(k)
        + digamma(n)
        + math.log(vm)
        + (m / n) * float(np.sum(np.log(2.0 * r)))
    )
    return ke


def _feat_phase_mean_std(x: np.ndarray) -> Tuple[float, float]:
    """Hilbert phase mean and std (Eqs. 15–16); unwrap before moments."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size < 8:
        return 0.0, 0.0
    z = signal.hilbert(x)
    phi = np.unwrap(np.angle(z))
    return float(np.mean(phi)), float(np.std(phi, ddof=1))


def extract_memar_features_vector(
    epoch_eeg: np.ndarray,
    fs: float,
    bands_config: Dict[str, Any] | None = None,
    band_sos_list: List[np.ndarray | None] | None = None,
) -> np.ndarray:
    """
    Parameters
    ----------
    epoch_eeg : 1d array
        One 30 s epoch of F4_O2 (or chosen channel) at `fs` Hz.
    band_sos_list
        If provided (from :func:`precompute_band_sos_list`), skips repeated
        ``scipy.signal.butter`` design for every epoch.
    """
    cfg = bands_config or load_bands_config()
    x0 = np.asarray(epoch_eeg, dtype=np.float64).ravel()
    out = np.zeros(FEATURE_DIM, dtype=np.float64)
    o = 0
    order = 14
    bands = cfg["bands_hz"]
    if band_sos_list is not None and len(band_sos_list) != len(bands):
        raise ValueError("band_sos_list length must match bands_hz")
    for bi, b in enumerate(bands):
        low_hz = float(b["low_hz"])
        high_hz = float(b["high_hz"])
        lowpass_only = bool(b.get("lowpass_only", False))
        sos_pre = band_sos_list[bi] if band_sos_list is not None else None
        if sos_pre is None and band_sos_list is not None:
            xs = np.zeros_like(x0)
        else:
            xs = _sos_filter_epoch(x0, fs, low_hz, high_hz, order, lowpass_only, sos_precomputed=sos_pre)

        x = np.asarray(xs, dtype=np.float64).ravel()
        n = x.size
        if n >= 2:
            d1 = np.diff(x)
            d2 = np.diff(d1) if n >= 3 else None
            sx_pre = float(np.std(x, ddof=1))
        else:
            d1 = np.empty(0, dtype=np.float64)
            d2 = None
            sx_pre = None

        out[o + 0] = _feat_sd(x, sx_pre=sx_pre)
        hm, hc = _feat_hjorth_mobility_complexity(x, d1=d1, d2=d2, sx=sx_pre if n >= 3 else None)
        out[o + 1] = hm
        out[o + 2] = hc
        out[o + 3] = _feat_mmd(x)
        out[o + 4] = _feat_pfd(x, dx=d1)
        out[o + 5] = _feat_nll(x, dx=d1)
        out[o + 6] = _feat_ghe(x)
        out[o + 7] = _feat_lrssv(x, dx=d1)
        out[o + 8] = _feat_nse(x, fs, low_hz, high_hz)
        out[o + 9] = _feat_renyi_m2(x)
        out[o + 10] = _feat_kraskov(x)
        pm, ps = _feat_phase_mean_std(x)
        out[o + 11] = pm
        out[o + 12] = ps
        o += 13
    assert o == FEATURE_DIM
    return out


def channel_tag_for_feature_name(signal_path: str) -> str:
    """Stable token for feature names (no slashes)."""
    return signal_path.replace("\\", "/").replace("/", "__")


def build_feature_names_multichannel(
    bands_config: Dict[str, Any],
    eeg_channel_paths: Sequence[str],
) -> List[str]:
    """
    One Memar 104-vector per EEG channel, concatenated. Names: ``{ch}__{band}_{feat}``.
    """
    base_names = build_feature_names(bands_config)
    names: List[str] = []
    for sig in eeg_channel_paths:
        tag = channel_tag_for_feature_name(sig)
        for bn in base_names:
            names.append("{}__{}".format(tag, bn))
    assert len(names) == FEATURE_DIM * len(eeg_channel_paths)
    return names


def total_memar_feature_dim(n_eeg_channels: int) -> int:
    return FEATURE_DIM * int(n_eeg_channels)


def extract_memar_features_multichannel(
    epoch_eeg: np.ndarray,
    fs: float,
    bands_config: Dict[str, Any] | None = None,
    band_sos_list: List[np.ndarray | None] | None = None,
) -> np.ndarray:
    """
    If ``epoch_eeg`` is 1-D, same as :func:`extract_memar_features_vector`.
    If 2-D ``(n_samples, n_channels)``, compute 104 features per column and concatenate.
    """
    a = np.asarray(epoch_eeg, dtype=np.float64)
    if a.ndim == 1:
        return extract_memar_features_vector(a, fs, bands_config, band_sos_list)
    if a.ndim != 2:
        raise ValueError("epoch_eeg must be 1d or 2d, got shape {}".format(a.shape))
    cfg = bands_config or load_bands_config()
    parts = [
        extract_memar_features_vector(a[:, c], fs, cfg, band_sos_list=band_sos_list)
        for c in range(a.shape[1])
    ]
    return np.concatenate(parts, axis=0)


def extract_memar_features_matrix(
    epochs: np.ndarray,
    fs: float,
    bands_config: Dict[str, Any] | None = None,
) -> np.ndarray:
    """epochs: shape (n_epochs, n_samples_per_epoch)"""
    epochs = np.asarray(epochs, dtype=np.float64)
    rows = [extract_memar_features_vector(epochs[i], fs, bands_config) for i in range(epochs.shape[0])]
    return np.vstack(rows)


def channel_index_for_signal(memmap_signals_order: Sequence[str], signal_path: str) -> int:
    for i, p in enumerate(memmap_signals_order):
        if p == signal_path:
            return i
    raise ValueError("Signal {!r} not found in memmap order".format(signal_path))


def eeg_signal_order_from_memmap_desc(memmap_description: dict) -> List[str]:
    for grp in memmap_description.get("signals", []):
        if grp.get("name") == "eeg":
            return list(grp["signals"])
    raise ValueError("No eeg group in memmap_description")


def memar_multichannel_eeg_paths(memmap_description: dict) -> List[str]:
    """
    Paths used for ``--all-eeg-channels``: only ``signals/eeg/...`` entries.

    The memmap ``eeg`` group often lists EMG/ECG/EOG paths that are stacked into ``eeg.mm``;
    those columns must not receive Memar EEG features.
    """
    paths = [p for p in eeg_signal_order_from_memmap_desc(memmap_description) if is_eeg_signal_path(p)]
    if not paths:
        raise ValueError("No signals/eeg/* paths in memmap eeg group (multichannel Memar mode)")
    return paths


# Default single-channel path; override via scripts/base_experiments/memar_et_al/memar_et_al_config.json
EEG_CHANNEL_DEFAULT = get_eeg_signal()
