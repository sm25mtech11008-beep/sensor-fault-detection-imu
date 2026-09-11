"""Experiment 5 - Reproduce the v1 injection artifact, then show it is fixed.

v1 added continuous Gaussian noise to every class except flatline. Real
MPU-9250 readings are quantised (~16 distinct values per 100-sample window),
so this made ``unique_ratio`` a detector for *"was synthetic noise added"*
rather than for a fault - and v1's perfect 100 % F1 on flatline was an
artefact of its own preprocessing.

The test is simple and damning: train each pipeline, then ask it to score
**raw, unmodified, healthy** sensor data. A correct model should call almost
all of it ``normal``.

    python experiments/05_v1_artifact.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from sfd import Config
from sfd.data import build_windows, signal_statistics
from sfd.evaluate import evaluate
from sfd.faults import FaultConfig, inject_dataset
from sfd.features import FEATURE_NAMES, extract
from sfd.models import build_models
from sfd.splits import leave_one_vehicle_out

# v1: 15 % of window sigma of Gaussian noise on every class but flatline,
# and no quantisation. v2: no synthetic noise, snapped to the sensor grid.
RECIPES = {
    "v1 (Gaussian noise, no quantisation)": dict(adc_noise=0.15, lsb=0.0),
    "v2 (no added noise, grid-preserving)": dict(adc_noise=0.0, lsb=None),
}

HELD_OUT_DRIVE = ("PVS3", "left")


def main() -> None:
    cfg = Config()
    cfg.validate()
    out = cfg.results_root / "experiments"
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("EXPERIMENT 5 - THE v1 INJECTION ARTIFACT")
    print("=" * 78)

    stats = signal_statistics(cfg, cfg.channel)
    ws = build_windows(cfg, cfg.channel)
    ui = FEATURE_NAMES.index("unique_ratio")

    # Raw, untouched, healthy windows from one drive - the reality check.
    raw_ws = build_windows(cfg, cfg.channel, datasets=[HELD_OUT_DRIVE[0]])
    raw_X = extract(raw_ws.windows)
    raw_uniq = float(np.median(raw_X[:, ui]))
    print(f"\nraw real data          : unique_ratio median {raw_uniq:.3f}")
    print(f"quantisation step (lsb): {stats['lsb']:.5f} degC")
    print(f"within-window sigma    : {stats['window_std_median']:.5f} degC\n")

    rows, health = [], []
    for name, kw in RECIPES.items():
        lsb = stats["lsb"] if kw["lsb"] is None else kw["lsb"]
        fc = FaultConfig(window_sigma=stats["window_std_median"],
                         adc_noise=kw["adc_noise"], lsb=lsb)
        faulted, y = inject_dataset(ws.windows, fc, n_classes=cfg.n_classes, seed=cfg.seed)
        X = extract(faulted)

        print("-" * 78)
        print(f"RECIPE: {name}")
        print("-" * 78)
        for c, cname in enumerate(cfg.class_names):
            u = float(np.median(X[y == c, ui]))
            rows.append({"recipe": name, "class": cname, "unique_ratio": u,
                         "raw_real_data": raw_uniq})
            flag = "  <-- matches real data" if abs(u - raw_uniq) < 0.06 else ""
            print(f"  {cname:16s} unique_ratio {u:.3f}{flag}")

        # Train leave-one-vehicle-out, then score raw healthy data.
        sp = next(iter(leave_one_vehicle_out(ws)))
        Xtr, Xte, ytr, yte = sp.apply(X, y)
        model = build_models(seed=cfg.seed, fast=True)["XGBoost"]
        r = evaluate(model, Xtr, ytr, Xte, yte, cfg.class_names, "XGBoost", sp.name)

        pred = model.predict(raw_X)
        counts = pd.Series([cfg.class_names[p] for p in pred]).value_counts()
        pct_normal = 100.0 * counts.get("normal", 0) / len(pred)
        pct_flat = 100.0 * counts.get("flatline_fault", 0) / len(pred)

        print(f"\n  held-out F1 macro        : {r.f1_macro * 100:.2f}%")
        print(f"  F1 on flatline           : {r.per_class_f1.get('flatline_fault', 0) * 100:.2f}%")
        print(f"  scoring RAW HEALTHY data ({len(pred):,} windows):")
        for cname, n in counts.items():
            print(f"      {cname:16s} {n:6,}  ({100 * n / len(pred):5.1f}%)")
        verdict = "OK" if pct_normal > 80 else "BROKEN - healthy data called faulty"
        print(f"  verdict: {verdict}\n")

        health.append({
            "recipe": name,
            "f1_macro": round(r.f1_macro * 100, 2),
            "f1_flatline": round(r.per_class_f1.get("flatline_fault", 0) * 100, 2),
            "raw_pct_normal": round(pct_normal, 1),
            "raw_pct_flatline": round(pct_flat, 1),
            "verdict": verdict,
        })

    pd.DataFrame(rows).to_csv(out / "v1_artifact_unique_ratio.csv", index=False)
    hdf = pd.DataFrame(health)
    hdf.to_csv(out / "v1_artifact_healthcheck.csv", index=False)

    print("=" * 78)
    print("SUMMARY - both recipes score well on their own test set;")
    print("          only one of them survives contact with real data.")
    print("=" * 78)
    print(hdf.to_string(index=False))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
