"""Evaluation protocols, from optimistic to defensible.

The original pipeline used ``train_test_split(X, y, stratify=y)`` on windows
generated with 50 % overlap. Two distinct leaks follow from that:

* **Overlap leak** - window *i* and window *i+1* share 50 of their 100
  samples. A random split puts one in train and its near-duplicate in test.
* **Drive leak** - windows from the same vehicle, driver and road segment
  appear on both sides, so the model can memorise drive-specific signal
  character instead of learning what a fault looks like.

``random`` is retained so the inflation can be *measured* rather than merely
asserted. ``grouped`` and ``vehicle`` are the honest protocols.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import (
    GroupKFold,
    LeaveOneGroupOut,
    train_test_split,
)

from .data import WindowSet

PROTOCOLS = ("random", "disjoint", "grouped", "vehicle")


@dataclass
class Split:
    train_idx: np.ndarray
    test_idx: np.ndarray
    name: str = ""

    def apply(self, *arrays):
        out = []
        for a in arrays:
            out.extend([a[self.train_idx], a[self.test_idx]])
        return tuple(out)


def _overlap_offsets(window_size: int, window_step: int) -> list[int]:
    """Start-offsets, relative to a window, that share at least one sample.

    Window starts are all multiples of ``window_step``, so overlap can be
    tested against a small fixed candidate set instead of scanning every
    other window.
    """
    k = int(np.ceil(window_size / window_step)) - 1
    return [o * window_step for o in range(-k, k + 1)]


def _overlap_stride(window_size: int, window_step: int) -> int:
    """How many consecutive windows share samples with a given window."""
    return max(1, int(np.ceil(window_size / window_step)))


def random_split(y: np.ndarray, test_size: float = 0.2, seed: int = 42) -> Split:
    """v1 behaviour: stratified random split. Leaks. Kept as the baseline."""
    tr, te = train_test_split(
        np.arange(len(y)), test_size=test_size, random_state=seed, stratify=y
    )
    return Split(tr, te, "random (leaky)")


def disjoint_split(
    ws: WindowSet, y: np.ndarray, window_size: int, window_step: int,
    test_size: float = 0.2, seed: int = 42,
) -> Split:
    """Random split, then drop test windows overlapping any training window.

    Isolates the *overlap* leak from the *drive* leak: the same drives still
    appear on both sides, so any remaining gap versus ``grouped`` is
    attributable to drive memorisation.
    """
    base = random_split(y, test_size, seed)
    offsets = _overlap_offsets(window_size, window_step)

    train_pos = set(zip(ws.drive[base.train_idx].tolist(),
                        ws.start_idx[base.train_idx].tolist(), strict=True))
    te_drives = ws.drive[base.test_idx].tolist()
    te_starts = ws.start_idx[base.test_idx].tolist()

    keep = [
        idx
        for idx, d, s in zip(base.test_idx, te_drives, te_starts, strict=True)
        if not any((d, s + o) in train_pos for o in offsets)
    ]
    return Split(base.train_idx, np.asarray(keep, dtype=np.int64), "disjoint windows")


def grouped_split(ws: WindowSet, n_splits: int = 5, fold: int = 0) -> Split:
    """Hold out whole recordings (drive = dataset x side). No shared samples."""
    groups = ws.drive
    gkf = GroupKFold(n_splits=n_splits)
    for k, (tr, te) in enumerate(gkf.split(np.zeros(len(groups)), groups=groups)):
        if k == fold:
            return Split(tr, te, f"grouped drives (fold {fold})")
    raise ValueError(f"fold {fold} out of range for n_splits={n_splits}")


def leave_one_vehicle_out(ws: WindowSet) -> Iterator[Split]:
    """Strictest protocol: train on two vehicles, test on the unseen third.

    This is the number to quote. It answers the question an interviewer will
    actually ask - does this work on a car it has never seen?
    """
    logo = LeaveOneGroupOut()
    vehicles = ws.vehicle
    for tr, te in logo.split(np.zeros(len(vehicles)), groups=vehicles):
        held = str(np.unique(vehicles[te])[0])
        yield Split(tr, te, f"leave-out {held}")


def make_split(protocol: str, ws: WindowSet, y: np.ndarray, cfg) -> Split:
    """Dispatch by protocol name; ``vehicle`` returns its first fold."""
    if protocol == "random":
        return random_split(y, seed=cfg.seed)
    if protocol == "disjoint":
        return disjoint_split(ws, y, cfg.window_size, cfg.window_step, seed=cfg.seed)
    if protocol == "grouped":
        return grouped_split(ws)
    if protocol == "vehicle":
        return next(iter(leave_one_vehicle_out(ws)))
    raise ValueError(f"unknown protocol {protocol!r}; expected one of {PROTOCOLS}")


def leakage_report(ws: WindowSet, split: Split) -> dict[str, float]:
    """Quantify how much a split leaks - useful as evidence in the write-up."""
    tr_drives = set(ws.drive[split.train_idx].tolist())
    te_drives = set(ws.drive[split.test_idx].tolist())
    shared = tr_drives & te_drives

    size = ws.window_size or ws.windows.shape[1]
    step = ws.window_step or 1
    if not size:
        raise ValueError("WindowSet has no window geometry; cannot audit overlap")
    offsets = _overlap_offsets(size, step)

    train_pos = set(zip(ws.drive[split.train_idx].tolist(),
                        ws.start_idx[split.train_idx].tolist(), strict=True))
    overlapping = sum(
        any((d, s + o) in train_pos for o in offsets)
        for d, s in zip(ws.drive[split.test_idx].tolist(),
                        ws.start_idx[split.test_idx].tolist(), strict=True)
    )
    return {
        "shared_drives": len(shared),
        "test_windows": len(split.test_idx),
        "test_overlapping_train": overlapping,
        "pct_test_overlapping": 100.0 * overlapping / max(1, len(split.test_idx)),
    }
