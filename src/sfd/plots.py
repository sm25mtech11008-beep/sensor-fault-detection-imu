"""Figure generation.

Palette is the validated four-slot categorical set (blue / orange / aqua /
yellow), passing every check in both light and dark mode. Two light-mode slots
fall below 3:1 against the surface, so the relief rule applies: every series
also carries a distinct marker shape, and each figure has a CSV table view
beside it in ``results/experiments/``.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Validated categorical palette, fixed slot order (never cycled).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8a85"
GRID = "#e5e5e1"
SURFACE = "#fcfcfb"

CLASS_SLOT = {
    "normal": SERIES[0],
    "noise_fault": SERIES[1],
    "bias_fault": SERIES[2],
    "flatline_fault": SERIES[3],
}


def _style(ax):
    """Recessive axes: no top/right spines, faint horizontal grid only."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(1.0)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    return ax


def _fig(w=9.0, h=5.4):
    fig, ax = plt.subplots(figsize=(w, h), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    return fig, _style(ax)


def _titles(ax, title: str, subtitle: str) -> None:
    """Title above subtitle, both clear of the axes and of each other."""
    ax.set_title(title, color=INK, fontsize=13, fontweight="700", loc="left", pad=30)
    ax.text(0, 1.035, subtitle, transform=ax.transAxes,
            color=INK_2, fontsize=9.5, va="bottom")


def detection_limit(csv: Path, out: Path, target: float = 90.0) -> Path:
    """The headline figure: per-class F1 against fault-to-noise ratio.

    ``normal`` and ``noise_fault`` track within ~3 points across the whole
    sweep, so direct labels cannot separate them. Each series therefore also
    carries a distinct marker shape, and ``detection_limit.csv`` is the table
    view - together these satisfy the relief rule for the two light-mode
    slots that fall below 3:1 against the surface.
    """
    df = pd.read_csv(csv)
    x = df["fault_to_noise"].to_numpy()

    fig, ax = _fig(9.4, 5.6)
    classes = [c for c in CLASS_SLOT if f"F1[{c}]" in df.columns]
    markers = {"normal": "o", "noise_fault": "s", "bias_fault": "^", "flatline_fault": "D"}

    ax.axhline(target, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.text(x[0] * 0.95, target + 1.6, f"F1 = {target:.0f}% detection threshold",
            color=MUTED, fontsize=8.5, va="bottom", ha="left")

    for cname in classes:
        ax.plot(x, df[f"F1[{cname}]"].to_numpy(), color=CLASS_SLOT[cname], lw=2.0,
                solid_capstyle="round", marker=markers[cname], markersize=5.0,
                markeredgecolor=SURFACE, markeredgewidth=1.3, zorder=3,
                label=cname.replace("_", " "))

    ax.set_xscale("log")
    ax.set_xlim(x.min() * 0.85, x.max() * 1.25)
    ax.set_ylim(25, 106)
    ax.set_xlabel("Fault amplitude / sensor noise floor  (x)", color=INK_2, fontsize=10)
    ax.set_ylabel("F1 score (%)", color=INK_2, fontsize=10)

    # Derive the crossings from the data so the caption cannot go stale.
    def _crossing(cname: str) -> float | None:
        ok = df[df[f"F1[{cname}]"] >= target]
        return float(ok["fault_to_noise"].iloc[0]) if len(ok) else None

    bias, noise = _crossing("bias_fault"), _crossing("noise_fault")
    detail = []
    if bias:
        detail.append(f"drift needs {bias:.1f}x the noise floor")
    if noise:
        detail.append(f"spikes {noise:.1f}x")
    _titles(ax, "Detection limit by fault mode",
            "Flatline is structural and detected at every amplitude swept; "
            + ", ".join(detail) + ".")
    # Lower right is the only quadrant the curves leave empty.
    ax.legend(frameon=False, fontsize=9.5, loc="lower right",
              bbox_to_anchor=(0.995, 0.02), labelcolor=INK_2, handlelength=2.4)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out


def protocol_comparison(csv: Path, out: Path) -> Path:
    """Accuracy per model under each evaluation protocol - the null result.

    A dot plot, not bars: the whole point is that the spread is under half a
    percentage point, and bars on the non-zero axis needed to see that spread
    would exaggerate it. Dots carry no area, so a clipped axis stays honest.
    """
    df = pd.read_csv(csv, index_col=0)
    df = df.drop(columns=[c for c in df.columns if "INFLATION" in c], errors="ignore")
    df = df.drop(index=[i for i in df.index if "Majority" in str(i)], errors="ignore")

    protocols = list(df.columns)
    models = list(df.index)[::-1]
    markers = ["o", "s", "^", "D"]
    fig, ax = _fig(9.6, 5.2)

    ypos = np.arange(len(models))
    for j, m in enumerate(models):
        vals = [df.loc[m, p] for p in protocols]
        # Connector shows the whole spread for this model at a glance.
        ax.plot([min(vals), max(vals)], [ypos[j]] * 2, color=GRID, lw=3.0,
                solid_capstyle="round", zorder=2)
    for i, p in enumerate(protocols):
        ax.scatter(df.loc[models, p].to_numpy(), ypos, s=62,
                   color=SERIES[i % len(SERIES)], marker=markers[i % len(markers)],
                   edgecolor=SURFACE, linewidth=1.3, zorder=3,
                   label=p)

    ax.set_yticks(ypos)
    ax.set_yticklabels(models, fontsize=9.5, color=INK)
    ax.set_ylim(-0.9, len(models) - 0.15)
    ax.grid(True, axis="x", color=GRID, linewidth=0.8)
    ax.grid(False, axis="y")
    ax.set_xlabel("Accuracy (%)", color=INK_2, fontsize=10)
    _titles(ax, "Accuracy is flat across evaluation protocols",
            "The leaky random split does not inflate results - the features carry no "
            "drive-specific information.")
    ax.legend(frameon=False, fontsize=9, loc="lower left", ncol=2,
              labelcolor=INK_2, columnspacing=1.6, handletextpad=0.4)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out


def confusion(cm: np.ndarray, class_names: list[str], out: Path, title: str) -> Path:
    """Row-normalised confusion matrix, single-hue sequential ramp."""
    pct = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100
    fig, ax = plt.subplots(figsize=(6.4, 5.4), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels([c.replace("_", "\n") for c in class_names], fontsize=9, color=INK_2)
    ax.set_yticklabels([c.replace("_", "\n") for c in class_names], fontsize=9, color=INK_2)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            v = pct[i, j]
            if v < 0.05:
                continue
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=9.5,
                    fontweight="700" if i == j else "400",
                    color="#ffffff" if v > 55 else INK)

    ax.set_xlabel("Predicted", color=INK_2, fontsize=10)
    ax.set_ylabel("Actual", color=INK_2, fontsize=10)
    ax.set_title(title, color=INK, fontsize=12, fontweight="700", loc="left", pad=12)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=INK_2, labelsize=8, length=0)
    cb.set_label("% of actual class", color=INK_2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out


def real_glitch(signal: np.ndarray, idx: int, out: Path) -> Path:
    """The acquisition fault found in the raw PVS data, with its IIR recovery."""
    lo, hi = idx - 12, idx + 16
    seg = signal[lo:hi]
    t = np.arange(lo, hi) - idx

    fig, ax = _fig(9.0, 4.4)
    ax.plot(t, seg, color=SERIES[0], lw=2.0, marker="o", markersize=4.5,
            markeredgecolor=SURFACE, markeredgewidth=1.3, zorder=3, label="measured")

    # Model only the recovery (t >= 0) and draw it *over* the measurement:
    # the two coincide almost exactly, so underneath it would be invisible.
    steady = float(np.median(signal[lo:idx]))
    fit = np.where(t >= 0, steady * (1 - 0.5 ** np.clip(t, 0, None)), np.nan)
    ax.plot(t, fit, color=SERIES[1], lw=2.2, ls=(0, (4, 3)), zorder=4,
            label="first-order IIR, alpha = 0.5")

    ax.axvline(0, color=MUTED, lw=1.0, ls=":", zorder=1)
    ax.annotate("dropped read\nall 28 channels = 0", xy=(0, seg.min()),
                xytext=(1.2, steady * 0.35), color=INK_2, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.0))

    ax.set_xlabel("Samples relative to the glitch (100 Hz)", color=INK_2, fontsize=10)
    ax.set_ylabel("Temperature (degC)", color=INK_2, fontsize=10)
    _titles(ax, "A real acquisition fault in the PVS data",
            "Recovery follows 1 - 0.5^n to four decimals: the logger smooths with a "
            "first-order IIR filter.")
    ax.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out
