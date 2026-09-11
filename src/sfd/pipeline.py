"""End-to-end dataset construction: window -> inject -> featurise, with caching."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from .config import Config
from .data import WindowSet, build_windows, signal_statistics
from .faults import FaultConfig, inject_dataset
from .features import FEATURE_NAMES, extract


@dataclass
class Dataset:
    X: np.ndarray            # (n, n_features)
    y: np.ndarray            # (n,)
    ws: WindowSet
    feature_names: list[str]
    stats: dict
    fault_config: dict

    def __len__(self) -> int:
        return len(self.y)


def _cache_key(cfg: Config, channel: str, severity: float, n_classes: int) -> str:
    payload = {
        "recipe": 4,          # bump when injection/feature semantics change
        "channel": channel,
        "window": cfg.window_size,
        "step": cfg.window_step,
        "severity": severity,
        "n_classes": n_classes,
        "seed": cfg.seed,
        "features": list(FEATURE_NAMES),
    }
    raw = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha1(raw).hexdigest()[:12]


def build_dataset(
    cfg: Config,
    channel: str | None = None,
    severity: float = 1.0,
    n_classes: int | None = None,
    use_cache: bool = True,
    verbose: bool = True,
) -> Dataset:
    """Construct the full supervised dataset for one channel."""
    channel = channel or cfg.channel
    n_classes = n_classes or cfg.n_classes
    key = _cache_key(cfg, channel, severity, n_classes)
    cache_dir = cfg.results_root / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"ds_{key}.npz"

    if use_cache and cache_file.exists():
        if verbose:
            print(f"  [cache] {cache_file.name}")
        z = np.load(cache_file, allow_pickle=True)
        ws = WindowSet(
            windows=np.empty((len(z["y"]), 0), dtype=np.float32),
            dataset_id=z["dataset_id"], side=z["side"], vehicle=z["vehicle"],
            scenario=z["scenario"], start_idx=z["start_idx"],
            window_size=int(z["window_size"]), window_step=int(z["window_step"]),
        )
        return Dataset(
            X=z["X"], y=z["y"], ws=ws, feature_names=list(FEATURE_NAMES),
            stats=json.loads(str(z["stats"])), fault_config=json.loads(str(z["fault_config"])),
        )

    if verbose:
        print(f"  [1/4] signal statistics for {channel} ...")
    stats = signal_statistics(cfg, channel)

    if verbose:
        print(f"  [2/4] windowing ({cfg.window_size} @ step {cfg.window_step}) ...")
    ws = build_windows(cfg, channel)

    fc = FaultConfig(window_sigma=stats["window_std_median"], severity=severity,
                     lsb=stats.get("lsb", 0.0))
    if verbose:
        print(f"  [3/4] injecting {n_classes} classes, severity={severity} ...")
    faulted, y = inject_dataset(ws.windows, fc, n_classes=n_classes, seed=cfg.seed)

    if verbose:
        print(f"  [4/4] extracting {len(FEATURE_NAMES)} features from {len(y):,} windows ...")
    X = extract(faulted)

    fc_dump = {k: v for k, v in fc.__dict__.items()}
    if use_cache:
        np.savez_compressed(
            cache_file, X=X, y=y,
            dataset_id=ws.dataset_id, side=ws.side, vehicle=ws.vehicle,
            scenario=ws.scenario, start_idx=ws.start_idx,
            window_size=cfg.window_size, window_step=cfg.window_step,
            stats=json.dumps(stats), fault_config=json.dumps(fc_dump),
        )
    ws.windows = faulted
    return Dataset(X=X, y=y, ws=ws, feature_names=list(FEATURE_NAMES),
                   stats=stats, fault_config=fc_dump)
