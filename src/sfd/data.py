"""Loading and windowing of PVS drive data."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import DATASET_META, SIDES, Config


@dataclass
class WindowSet:
    """Windows plus the provenance needed for group-aware splitting.

    ``groups`` is what makes an honest evaluation possible: it records which
    drive each window came from, so whole drives (not random rows) can be
    held out.
    """

    windows: np.ndarray      # (n_windows, window_size) float32
    dataset_id: np.ndarray   # (n_windows,) e.g. 'PVS3'
    side: np.ndarray         # (n_windows,) 'left' | 'right'
    vehicle: np.ndarray      # (n_windows,) e.g. 'Fiat Bravo'
    scenario: np.ndarray     # (n_windows,)
    start_idx: np.ndarray    # (n_windows,) sample offset within the drive
    # Geometry is stored explicitly rather than read off ``windows``: the
    # dataset cache round-trips features only, so ``windows`` may be empty.
    window_size: int = 0
    window_step: int = 0

    def __len__(self) -> int:
        return len(self.windows)

    @property
    def drive(self) -> np.ndarray:
        """Unique id per (dataset, side) recording — the finest safe group."""
        return np.char.add(np.char.add(self.dataset_id, "/"), self.side)


def sliding_windows(signal: np.ndarray, size: int, step: int) -> np.ndarray:
    """Zero-copy sliding window view, materialised as a contiguous array.

    Vectorised replacement for the per-window Python loop in the original
    Phase 2 notebook.
    """
    if len(signal) < size:
        return np.empty((0, size), dtype=np.float32)
    view = np.lib.stride_tricks.sliding_window_view(signal, size)[::step]
    return np.ascontiguousarray(view, dtype=np.float32)


def load_channel(cfg: Config, dataset_id: str, side: str, channel: str) -> np.ndarray:
    """Read a single sensor column from one drive."""
    path = cfg.drive_path(dataset_id, side)
    df = pd.read_csv(path, usecols=[channel], dtype={channel: "float32"})
    return df[channel].dropna().to_numpy(dtype=np.float32)


def build_windows(cfg: Config, channel: str | None = None, datasets=None) -> WindowSet:
    """Window every drive for one channel, carrying provenance through."""
    channel = channel or cfg.channel
    if channel is None:
        raise ValueError("channel must be set on the Config or passed explicitly")
    datasets = datasets or list(DATASET_META)

    chunks, d_ids, sides, vehicles, scenarios, starts = [], [], [], [], [], []
    for dataset_id in datasets:
        meta = DATASET_META[dataset_id]
        for side in SIDES:
            sig = load_channel(cfg, dataset_id, side, channel)
            w = sliding_windows(sig, cfg.window_size, cfg.window_step)
            if len(w) == 0:
                continue
            n = len(w)
            chunks.append(w)
            d_ids.append(np.full(n, dataset_id))
            sides.append(np.full(n, side))
            vehicles.append(np.full(n, meta["vehicle"]))
            scenarios.append(np.full(n, meta["scenario"]))
            starts.append(np.arange(n, dtype=np.int64) * cfg.window_step)

    return WindowSet(
        windows=np.concatenate(chunks),
        dataset_id=np.concatenate(d_ids),
        side=np.concatenate(sides),
        vehicle=np.concatenate(vehicles),
        scenario=np.concatenate(scenarios),
        start_idx=np.concatenate(starts),
        window_size=cfg.window_size,
        window_step=cfg.window_step,
    )


def signal_statistics(cfg: Config, channel: str) -> dict[str, float]:
    """Global and within-window statistics used to calibrate fault amplitudes.

    The distinction matters: the original pipeline scaled faults by the
    *global* std while the natural variation inside a 1 s window is ~10x
    smaller, which made the injected faults far more obvious than intended.
    """
    all_vals, win_stds, lsbs = [], [], []
    for dataset_id in DATASET_META:
        for side in SIDES:
            sig = load_channel(cfg, dataset_id, side, channel)
            all_vals.append(sig)
            n = (len(sig) // cfg.window_size) * cfg.window_size
            if n:
                win_stds.append(sig[:n].reshape(-1, cfg.window_size).std(axis=1))
            # Per drive: pooling drives first would blend their grids into a
            # finer, fictitious step.
            g = np.diff(np.unique(sig.astype(np.float64)))
            g = g[g > 1e-9]
            if g.size:
                lsbs.append(float(np.median(g)))

    vals = np.concatenate(all_vals)
    stds = np.concatenate(win_stds)

    # Effective quantisation step. The MPU-9250 thermometer has a 0.002995 degC
    # LSB, but the logger's IIR smoothing puts values on a finer dyadic grid,
    # so the step is measured rather than assumed. Preserving it matters: real
    # readings are discrete, and injecting continuous Gaussian noise would
    # otherwise make "noise was added" trivially detectable via unique_ratio.
    lsb = float(np.median(lsbs)) if lsbs else 0.0

    return {
        "lsb": lsb,
        "global_mean": float(vals.mean()),
        "global_std": float(vals.std()),
        "window_std_median": float(np.median(stds)),
        "window_std_p95": float(np.percentile(stds, 95)),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "n_samples": int(vals.size),
    }
