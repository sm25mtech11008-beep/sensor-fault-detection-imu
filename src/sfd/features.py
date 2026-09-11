"""Time-domain feature extraction.

Fully vectorised over the window axis: the original notebook looped in
Python and called ``np.polyfit`` per window, which dominated runtime.

Feature-set policy differs from v1. The original dropped any feature with
|r| > 0.90 against another, which removed ``max_diff`` and ``mean_abs_diff``
-- precisely the two statistics that isolate an impulsive spike. Correlation
is a statement about marginal redundancy, not about conditional usefulness
to a non-linear classifier, so pruning on it discarded signal. Here the full
set is kept by default and pruning is an explicit, measurable option.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, skew

FEATURE_NAMES: tuple[str, ...] = (
    "mean",
    "std",
    "variance",
    "rms",
    "min",
    "max",
    "range",
    "skewness",
    "kurtosis",
    "mean_abs_diff",
    "max_diff",
    "trend",
    "unique_ratio",
    "zero_cross_rate",
    "iqr",
    "energy_hf",
)

FEATURE_DOC: dict[str, str] = {
    "mean": "Average level - shifts under bias/drift",
    "std": "Spread - near zero in flatline, elevated under noise",
    "variance": "Squared spread - amplifies std differences",
    "rms": "Signal energy",
    "min": "Minimum - pulled down by negative spikes",
    "max": "Maximum - pulled up by positive spikes",
    "range": "max - min - zero in flatline, large under noise",
    "skewness": "Asymmetry from one-sided spikes",
    "kurtosis": "Tail heaviness - impulsive noise",
    "mean_abs_diff": "Mean step size - zero in flatline, large under noise",
    "max_diff": "Largest single step - isolates the spike instant",
    "trend": "Least-squares slope - non-zero under bias",
    "unique_ratio": "Fraction of distinct values - ~0 in flatline/stuck",
    "zero_cross_rate": "Mean crossings - collapses when the signal freezes",
    "iqr": "Robust spread - insensitive to a few spikes, unlike std",
    "energy_hf": "High-frequency energy via second difference",
}


def _trend(w: np.ndarray) -> np.ndarray:
    """Least-squares slope per row, in closed form (no per-window polyfit)."""
    n = w.shape[1]
    x = np.arange(n, dtype=np.float64)
    x_c = x - x.mean()
    denom = (x_c ** 2).sum()
    return (w - w.mean(axis=1, keepdims=True)) @ x_c / denom


def _unique_ratio(w: np.ndarray, decimals: int = 4) -> np.ndarray:
    """Fraction of distinct rounded values per row, without a Python loop."""
    r = np.sort(np.round(w, decimals), axis=1)
    changed = np.diff(r, axis=1) != 0
    return (1 + changed.sum(axis=1)) / w.shape[1]


def extract(windows: np.ndarray) -> np.ndarray:
    """Return an (n_windows, n_features) matrix in ``FEATURE_NAMES`` order."""
    w = np.asarray(windows, dtype=np.float64)
    if w.ndim == 1:
        w = w[None, :]

    mean = w.mean(axis=1)
    std = w.std(axis=1)
    mn = w.min(axis=1)
    mx = w.max(axis=1)
    d1 = np.abs(np.diff(w, axis=1))
    d2 = np.diff(w, n=2, axis=1)
    centred = w - mean[:, None]

    # Compute shape statistics only where the window actually varies: on a
    # constant window scipy warns about catastrophic cancellation and the
    # moment is undefined anyway. Flatline/stuck windows land here by design.
    safe = std > 1e-12
    sk = np.zeros(len(w))
    ku = np.zeros(len(w))
    if safe.any():
        sk[safe] = skew(w[safe], axis=1)
        ku[safe] = kurtosis(w[safe], axis=1)

    q75, q25 = np.percentile(w, [75, 25], axis=1)

    feats = np.column_stack(
        [
            mean,
            std,
            w.var(axis=1),
            np.sqrt((w ** 2).mean(axis=1)),
            mn,
            mx,
            mx - mn,
            sk,
            ku,
            d1.mean(axis=1),
            d1.max(axis=1),
            _trend(w),
            _unique_ratio(w),
            (np.diff(np.signbit(centred), axis=1) != 0).mean(axis=1),
            q75 - q25,
            (d2 ** 2).mean(axis=1),
        ]
    )
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def to_frame(windows: np.ndarray):
    import pandas as pd

    return pd.DataFrame(extract(windows), columns=list(FEATURE_NAMES))
