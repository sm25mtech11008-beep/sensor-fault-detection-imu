"""Feature extraction correctness - the numbers the whole project rests on."""
import numpy as np
import pytest
from scipy.stats import kurtosis, skew

from sfd.features import FEATURE_NAMES, extract


@pytest.fixture
def windows():
    rng = np.random.default_rng(0)
    return rng.normal(35.0, 0.1, (64, 100)).astype(np.float32)


def col(F, name):
    return F[:, FEATURE_NAMES.index(name)]


def test_shape_and_finiteness(windows):
    F = extract(windows)
    assert F.shape == (64, len(FEATURE_NAMES))
    assert np.isfinite(F).all()


def test_accepts_single_window():
    assert extract(np.zeros(100)).shape == (1, len(FEATURE_NAMES))


def test_basic_moments_match_numpy(windows):
    F = extract(windows)
    np.testing.assert_allclose(col(F, "mean"), windows.mean(1), rtol=1e-6)
    np.testing.assert_allclose(col(F, "std"), windows.std(1), rtol=1e-6)
    np.testing.assert_allclose(col(F, "min"), windows.min(1), rtol=1e-6)
    np.testing.assert_allclose(col(F, "max"), windows.max(1), rtol=1e-6)
    np.testing.assert_allclose(col(F, "range"), windows.max(1) - windows.min(1), rtol=1e-6)


def test_shape_stats_match_scipy(windows):
    """extract() upcasts to float64; compare against scipy at the same dtype."""
    F = extract(windows)
    w64 = windows.astype(np.float64)
    np.testing.assert_allclose(col(F, "skewness"), skew(w64, axis=1), rtol=1e-9)
    np.testing.assert_allclose(col(F, "kurtosis"), kurtosis(w64, axis=1), rtol=1e-9)


def test_trend_matches_polyfit(windows):
    """Closed-form slope must equal np.polyfit, which it replaces for speed."""
    F = extract(windows)
    expected = np.polyfit(np.arange(100), windows.T, 1)[0]
    np.testing.assert_allclose(col(F, "trend"), expected, atol=1e-12)


def test_trend_recovers_known_slope():
    slope = 0.037
    w = (np.arange(100) * slope)[None, :]
    assert col(extract(w), "trend")[0] == pytest.approx(slope, rel=1e-9)


def test_constant_window_is_degenerate():
    """A flatlined sensor must produce the signature the classifier keys on."""
    F = extract(np.full((1, 100), 31.5))
    assert col(F, "std")[0] == pytest.approx(0.0)
    assert col(F, "range")[0] == pytest.approx(0.0)
    assert col(F, "unique_ratio")[0] == pytest.approx(0.01)
    assert col(F, "skewness")[0] == 0.0   # guarded, not NaN
    assert col(F, "kurtosis")[0] == 0.0


def test_unique_ratio_counts_distinct_values():
    w = np.tile([1.0, 2.0], 50)[None, :]
    assert col(extract(w), "unique_ratio")[0] == pytest.approx(2 / 100)


def test_max_diff_isolates_a_spike():
    w = np.zeros((1, 100))
    w[0, 50] = 5.0
    assert col(extract(w), "max_diff")[0] == pytest.approx(5.0)
