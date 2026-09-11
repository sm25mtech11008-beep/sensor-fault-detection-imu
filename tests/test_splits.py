"""Split protocols - these guard the honesty of every reported number."""
import numpy as np
import pytest

from sfd.data import WindowSet, sliding_windows
from sfd.splits import (
    _overlap_offsets,
    disjoint_split,
    grouped_split,
    leakage_report,
    leave_one_vehicle_out,
    random_split,
)


@pytest.fixture
def ws():
    """Two vehicles x two drives, 100 windows each, 50 % overlap."""
    n_per = 100
    drives = [("PVS1", "left", "VW"), ("PVS2", "right", "VW"),
              ("PVS4", "left", "Fiat"), ("PVS5", "right", "Fiat")]
    d_id, side, veh, scen, start = [], [], [], [], []
    for did, sd, vh in drives:
        d_id += [did] * n_per
        side += [sd] * n_per
        veh += [vh] * n_per
        scen += ["S1"] * n_per
        start += list(np.arange(n_per) * 50)
    n = len(d_id)
    return WindowSet(
        windows=np.zeros((n, 100), dtype=np.float32),
        dataset_id=np.array(d_id), side=np.array(side), vehicle=np.array(veh),
        scenario=np.array(scen), start_idx=np.array(start),
        window_size=100, window_step=50,
    )


@pytest.fixture
def y(ws):
    return np.random.default_rng(0).integers(0, 4, len(ws))


def test_sliding_windows_are_contiguous_and_overlapping():
    sig = np.arange(500, dtype=np.float32)
    w = sliding_windows(sig, 100, 50)
    assert w.shape == (9, 100)
    np.testing.assert_array_equal(w[0][50:], w[1][:50])  # 50 % overlap


def test_sliding_windows_handles_short_signal():
    assert sliding_windows(np.arange(10, dtype=np.float32), 100, 50).shape == (0, 100)


def test_overlap_offsets_for_50pct():
    assert sorted(_overlap_offsets(100, 50)) == [-50, 0, 50]


def test_leakage_audit_survives_an_emptied_window_array(ws, y):
    """The dataset cache drops raw windows; the audit must still work."""
    import dataclasses
    stripped = dataclasses.replace(ws, windows=np.empty((len(ws), 0), dtype=np.float32))
    assert leakage_report(stripped, random_split(y, seed=0))["pct_test_overlapping"] > 80


def test_drive_id_is_dataset_and_side(ws):
    assert set(ws.drive) == {"PVS1/left", "PVS2/right", "PVS4/left", "PVS5/right"}


def test_random_split_leaks_heavily(ws, y):
    """The v1 protocol: near-total overlap between train and test."""
    rep = leakage_report(ws, random_split(y, seed=0))
    assert rep["shared_drives"] == 4
    assert rep["pct_test_overlapping"] > 80


def test_disjoint_split_removes_overlap(ws, y):
    """No test window may share samples with a training window.

    Drives are still shared with train (that is the point of this protocol -
    it isolates the overlap leak from the drive leak), but on a fixture this
    small the surviving test set is only a handful of windows.
    """
    sp = disjoint_split(ws, y, 100, 50, seed=0)
    rep = leakage_report(ws, sp)
    assert rep["pct_test_overlapping"] == 0
    assert len(sp.test_idx) > 0
    assert rep["shared_drives"] >= 1
    assert set(ws.drive[sp.test_idx]).issubset(set(ws.drive[sp.train_idx]))


def test_grouped_split_shares_no_drive(ws):
    sp = grouped_split(ws, n_splits=4, fold=0)
    assert leakage_report(ws, sp)["shared_drives"] == 0
    assert set(ws.drive[sp.train_idx]).isdisjoint(set(ws.drive[sp.test_idx]))


def test_leave_one_vehicle_out_holds_out_whole_vehicle(ws):
    folds = list(leave_one_vehicle_out(ws))
    assert len(folds) == 2
    for sp in folds:
        tr_v, te_v = set(ws.vehicle[sp.train_idx]), set(ws.vehicle[sp.test_idx])
        assert tr_v.isdisjoint(te_v)
        assert len(te_v) == 1
        assert leakage_report(ws, sp)["pct_test_overlapping"] == 0


def test_splits_partition_all_indices(ws, y):
    for sp in [random_split(y, seed=0), grouped_split(ws, n_splits=4, fold=0)]:
        assert len(np.intersect1d(sp.train_idx, sp.test_idx)) == 0


def test_apply_slices_consistently(ws, y):
    sp = grouped_split(ws, n_splits=4, fold=0)
    X = np.arange(len(y) * 3).reshape(len(y), 3)
    Xtr, Xte, ytr, yte = sp.apply(X, y)
    assert len(Xtr) == len(ytr) and len(Xte) == len(yte)
    np.testing.assert_array_equal(Xtr, X[sp.train_idx])
