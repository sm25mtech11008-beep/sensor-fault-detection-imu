"""Sensor fault models.

Two changes versus the original Phase 2 notebook:

1. Amplitudes are calibrated against the *within-window* standard deviation,
   not the global one. Global std for a temperature channel is dominated by
   slow warm-up drift over ~20 minutes; inside a 1 s window the signal varies
   ~10x less. Scaling by the global std made injected faults 15-40x larger
   than natural in-window variation, i.e. much easier than a real fault.

2. A single ``severity`` multiplier scales every fault, so difficulty is an
   explicit, reportable experimental axis rather than a hidden constant.

Three additional fault modes are provided that the original omitted and that
occur in real IMU/thermometer hardware: stuck-at-last-value, intermittent
dropout, and scale-factor (gain) error.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import CLASS_LABELS


@dataclass(frozen=True)
class FaultConfig:
    """Fault amplitudes expressed in units of within-window sigma."""

    window_sigma: float          # natural std inside one window
    severity: float = 1.0        # global difficulty knob
    lsb: float = 0.0             # sensor quantisation step; 0 disables snapping

    # Multipliers on window_sigma
    adc_noise: float = 0.0       # EXTRA synthetic noise; real data already has its own
    spike_min: float = 3.0
    spike_max: float = 8.0
    spike_count: tuple[int, int] = (3, 7)
    bias_min: float = 1.5
    bias_max: float = 5.0
    flat_jitter: float = 0.02
    dropout_frac: tuple[float, float] = (0.1, 0.4)
    scale_min: float = 1.15
    scale_max: float = 1.6

    def s(self, mult: float) -> float:
        """Resolve a fault sigma-multiplier into an absolute amplitude.

        Scaled by ``severity``. The noise floor deliberately is *not* - see
        :meth:`floor` - so that ``severity`` changes the fault-to-noise ratio
        rather than rescaling the whole problem (which would leave difficulty
        unchanged).
        """
        return mult * self.window_sigma * self.severity

    def floor(self) -> float:
        """The noise a fault has to beat, in absolute units.

        This is the signal's *measured* within-window standard deviation, not
        a synthetic constant. v1 added Gaussian noise on top of readings that
        already carry their own, which both double-counted the noise and
        destroyed the sensor's quantisation.
        """
        return self.window_sigma * (1.0 + self.adc_noise)

    @property
    def fault_to_noise(self) -> float:
        """Mean spike amplitude divided by the noise floor - the difficulty axis."""
        return self.s(0.5 * (self.spike_min + self.spike_max)) / self.floor()


def _rng(seed: int | np.random.Generator) -> np.random.Generator:
    return seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)


def quantise(w: np.ndarray, fc: FaultConfig) -> np.ndarray:
    """Snap a window back onto the sensor's own value grid.

    Every injector ends with this. Without it the injected classes carry
    continuous values while a genuinely flat sensor (and all real data) stays
    discrete, so ``unique_ratio`` ends up detecting *the injection procedure*
    rather than the fault - and the model then labels raw, healthy data as
    flatline.
    """
    if not fc.lsb:
        return w
    return np.round(w / fc.lsb) * fc.lsb


def add_adc_noise(w: np.ndarray, fc: FaultConfig, rng: np.random.Generator) -> np.ndarray:
    """Thermal/quantisation noise floor present on every real reading.

    Defaults to a no-op: the real signal already contains its own noise. Kept
    as a knob so the v1 behaviour can be reproduced for comparison.
    """
    if not fc.adc_noise:
        return quantise(w, fc)
    return quantise(w + rng.normal(0.0, fc.adc_noise * fc.window_sigma, w.shape), fc)


def inject_noise(w: np.ndarray, fc: FaultConfig, rng: np.random.Generator) -> np.ndarray:
    """Class 1 - impulsive spikes (loose connector, EMI)."""
    out = add_adc_noise(w.copy(), fc, rng)
    n = out.shape[-1]
    k = rng.integers(fc.spike_count[0], fc.spike_count[1] + 1)
    pos = rng.choice(n, size=k, replace=False)
    amp = rng.uniform(fc.s(fc.spike_min), fc.s(fc.spike_max), k)
    out[pos] += rng.choice([-1.0, 1.0], k) * amp
    return quantise(out, fc)


def inject_bias(w: np.ndarray, fc: FaultConfig, rng: np.random.Generator) -> np.ndarray:
    """Class 2 - progressive drift (ageing, thermal runaway)."""
    out = add_adc_noise(w.copy(), fc, rng)
    n = out.shape[-1]
    amount = rng.uniform(fc.s(fc.bias_min), fc.s(fc.bias_max)) * rng.choice([-1.0, 1.0])
    start = int(rng.uniform(0.0, 0.4) * n)
    ramp = np.zeros(n)
    ramp[start:] = np.linspace(0.0, amount, n - start)
    return quantise(out + ramp, fc)


def inject_flatline(w: np.ndarray, fc: FaultConfig, rng: np.random.Generator) -> np.ndarray:
    """Class 3 - output frozen at a constant (dead sensor, bus fault)."""
    n = w.shape[-1]
    return quantise(np.full(n, float(w.mean())) + rng.normal(0.0, fc.flat_jitter * fc.window_sigma, n), fc)


def inject_stuck(w: np.ndarray, fc: FaultConfig, rng: np.random.Generator) -> np.ndarray:
    """Class 4 - freezes mid-window at the last good value.

    Harder than flatline: the first part of the window is genuinely normal,
    so window-level statistics are only half-affected.
    """
    out = add_adc_noise(w.copy(), fc, rng)
    n = out.shape[-1]
    freeze = int(rng.uniform(0.2, 0.7) * n)
    out[freeze:] = out[freeze]
    return quantise(out, fc)


def inject_dropout(w: np.ndarray, fc: FaultConfig, rng: np.random.Generator) -> np.ndarray:
    """Class 5 - intermittent loss: samples replaced by zero-order hold."""
    out = add_adc_noise(w.copy(), fc, rng)
    n = out.shape[-1]
    frac = rng.uniform(*fc.dropout_frac)
    n_drop = max(1, int(frac * n))
    start = rng.integers(0, max(1, n - n_drop))
    out[start : start + n_drop] = out[max(0, start - 1)]
    return quantise(out, fc)


def inject_scale(w: np.ndarray, fc: FaultConfig, rng: np.random.Generator) -> np.ndarray:
    """Class 6 - gain error: readings scaled about the window mean."""
    out = add_adc_noise(w.copy(), fc, rng)
    gain = rng.uniform(fc.scale_min, fc.scale_max)
    if rng.random() < 0.5:
        gain = 1.0 / gain
    mu = out.mean()
    return quantise(mu + (out - mu) * gain, fc)


FAULT_FNS = {
    1: inject_noise,
    2: inject_bias,
    3: inject_flatline,
    4: inject_stuck,
    5: inject_dropout,
    6: inject_scale,
}


def inject_dataset(
    windows: np.ndarray,
    fc: FaultConfig,
    n_classes: int = 4,
    seed: int = 42,
    class_probs: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign each window a random class and apply the matching fault.

    Returns ``(faulted_windows, labels)``. Class 0 windows still receive the
    ADC noise floor so that 'normal' is not trivially the only smooth class.
    """
    rng = _rng(seed)
    n = len(windows)
    if class_probs is None:
        class_probs = np.full(n_classes, 1.0 / n_classes)
    labels = rng.choice(n_classes, size=n, p=class_probs)

    out = np.empty_like(windows, dtype=np.float32)
    for i in range(n):
        lbl = int(labels[i])
        if lbl == 0:
            out[i] = add_adc_noise(windows[i].astype(np.float64), fc, rng)
        else:
            out[i] = FAULT_FNS[lbl](windows[i].astype(np.float64), fc, rng)
    return out, labels.astype(np.int8)


def describe(n_classes: int) -> list[str]:
    return [CLASS_LABELS[i] for i in range(n_classes)]
