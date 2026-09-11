"""Fault injection must produce the signatures it claims to."""
import numpy as np
import pytest

from sfd.faults import (
    FaultConfig,
    add_adc_noise,
    inject_bias,
    inject_dataset,
    inject_dropout,
    inject_flatline,
    inject_noise,
    inject_scale,
    inject_stuck,
)

SIGMA = 0.1


@pytest.fixture
def fc():
    return FaultConfig(window_sigma=SIGMA)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def clean():
    return np.random.default_rng(1).normal(35.0, SIGMA, 100)


def test_noise_floor_is_independent_of_severity():
    """Severity must change fault-to-noise ratio, not rescale the problem."""
    a, b = FaultConfig(window_sigma=SIGMA, severity=0.1), FaultConfig(window_sigma=SIGMA, severity=2.0)
    assert a.floor() == pytest.approx(b.floor())
    assert b.fault_to_noise > a.fault_to_noise


def test_flatline_has_near_zero_variance(fc, rng, clean):
    assert inject_flatline(clean, fc, rng).std() < 0.05 * clean.std()


def test_noise_raises_max_step(fc, rng, clean):
    base = add_adc_noise(clean, fc, rng)
    out = inject_noise(clean, fc, rng)
    assert np.abs(np.diff(out)).max() > np.abs(np.diff(base)).max()


def test_bias_introduces_slope(fc, rng, clean):
    out = inject_bias(clean, fc, rng)
    slope = np.polyfit(np.arange(100), out, 1)[0]
    assert abs(slope) > abs(np.polyfit(np.arange(100), clean, 1)[0])


def test_stuck_freezes_the_tail(fc, rng, clean):
    out = inject_stuck(clean, fc, rng)
    assert np.diff(out)[-5:].max() == pytest.approx(0.0, abs=1e-12)


def test_dropout_holds_a_constant_run(fc, rng, clean):
    out = inject_dropout(clean, fc, rng)
    runs = np.diff(out) == 0
    assert runs.sum() >= 9


def test_scale_changes_spread_not_centre(fc, rng, clean):
    out = inject_scale(clean, fc, rng)
    assert out.mean() == pytest.approx(add_adc_noise(clean, fc, np.random.default_rng(0)).mean(), abs=0.05)
    assert out.std() != pytest.approx(clean.std(), rel=0.05)


def test_all_windows_preserved_and_labelled(fc):
    w = np.random.default_rng(2).normal(35, SIGMA, (500, 100)).astype(np.float32)
    X, y = inject_dataset(w, fc, n_classes=4, seed=3)
    assert X.shape == w.shape
    assert set(np.unique(y)) <= {0, 1, 2, 3}
    assert np.isfinite(X).all()


def test_injection_is_reproducible(fc):
    w = np.random.default_rng(2).normal(35, SIGMA, (200, 100)).astype(np.float32)
    a, ya = inject_dataset(w, fc, n_classes=4, seed=7)
    b, yb = inject_dataset(w, fc, n_classes=4, seed=7)
    np.testing.assert_array_equal(ya, yb)
    np.testing.assert_allclose(a, b)


# ── Regression: injection must not leak "was this synthesised?" ──────────
# A quantised sensor emits few distinct values per window. If injection adds
# continuous noise, unique_ratio separates injected from real windows rather
# than faulty from healthy - and the model then calls raw healthy data
# 'flatline'. These tests pin the fix.

LSB = 0.012


@pytest.fixture
def quantised_signal():
    """A realistic window: quantised, slowly varying, already noisy."""
    rng = np.random.default_rng(11)
    raw = 35.0 + np.cumsum(rng.normal(0, 0.02, 100))
    return np.round(raw / LSB) * LSB


def _unique_ratio(w):
    return len(np.unique(np.round(w, 4))) / len(w)


def test_quantise_snaps_to_the_grid(quantised_signal):
    from sfd.faults import quantise
    fc = FaultConfig(window_sigma=SIGMA, lsb=LSB)
    out = quantise(quantised_signal + 0.003, fc)
    np.testing.assert_allclose(out / LSB, np.round(out / LSB), atol=1e-9)


def test_quantise_is_a_noop_without_lsb(quantised_signal):
    from sfd.faults import quantise
    fc = FaultConfig(window_sigma=SIGMA, lsb=0.0)
    np.testing.assert_array_equal(quantise(quantised_signal, fc), quantised_signal)


def test_normal_class_preserves_source_discreteness(quantised_signal):
    """The headline regression: injected 'normal' must look like real data."""
    fc = FaultConfig(window_sigma=SIGMA, lsb=LSB)
    rng = np.random.default_rng(0)
    out = add_adc_noise(quantised_signal.copy(), fc, rng)
    assert _unique_ratio(out) == pytest.approx(_unique_ratio(quantised_signal), abs=0.05)


def test_every_fault_mode_stays_on_the_grid(quantised_signal):
    from sfd.faults import FAULT_FNS
    fc = FaultConfig(window_sigma=SIGMA, lsb=LSB)
    for label, fn in FAULT_FNS.items():
        out = fn(quantised_signal.copy(), fc, np.random.default_rng(label))
        off = np.abs(out / LSB - np.round(out / LSB)).max()
        assert off < 1e-9, f"class {label} left the sensor grid (residual {off})"


def test_noise_floor_uses_measured_variation_not_a_constant():
    """floor() must track the signal's own in-window std."""
    a = FaultConfig(window_sigma=0.1)
    b = FaultConfig(window_sigma=0.2)
    assert b.floor() == pytest.approx(2 * a.floor())
    assert a.floor() == pytest.approx(0.1)
