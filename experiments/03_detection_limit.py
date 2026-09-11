"""Experiment 3 - Minimum detectable fault amplitude.

A single accuracy number says almost nothing about a fault detector: it is a
function of how large the injected faults were, which is a free parameter.
The engineering-meaningful quantity is the *detection limit* - how small a
fault can be before the classifier stops seeing it.

Here the fault-to-noise ratio is swept while the hardware noise floor is held
fixed, and per-class F1 is measured under leave-one-vehicle-out. The output is
a characteristic curve plus, per fault mode, the smallest amplitude still
detected at F1 >= 90 %.
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
from sfd.features import extract
from sfd.models import build_models
from sfd.splits import leave_one_vehicle_out

SEVERITIES = [0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.25, 0.4, 0.6, 1.0, 1.5]
TARGET_F1 = 90.0


def main() -> None:
    cfg = Config()
    cfg.validate()
    out = cfg.results_root / "experiments"
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("EXPERIMENT 3 - DETECTION LIMIT / MINIMUM DETECTABLE FAULT")
    print("=" * 78)

    stats = signal_statistics(cfg, cfg.channel)
    sigma = stats["window_std_median"]
    print(f"channel          : {cfg.channel}")
    print(f"within-window std: {sigma:.5f} degC   (the natural noise the fault must beat)")
    print(f"global std       : {stats['global_std']:.5f} degC   (dominated by 20-min warm-up drift)")
    print(f"quantisation step: {stats['lsb']:.5f} degC   (injected windows are snapped to this)")

    ws = build_windows(cfg, cfg.channel)
    print(f"windows          : {len(ws):,}\n")

    rows = []
    for sev in SEVERITIES:
        fc = FaultConfig(window_sigma=sigma, severity=sev, lsb=stats["lsb"])
        faulted, y = inject_dataset(ws.windows, fc, n_classes=cfg.n_classes, seed=cfg.seed)
        X = extract(faulted)

        accs, f1s, per_class = [], [], []
        for sp in leave_one_vehicle_out(ws):
            Xtr, Xte, ytr, yte = sp.apply(X, y)
            model = build_models(seed=cfg.seed, fast=True)["XGBoost"]
            r = evaluate(model, Xtr, ytr, Xte, yte, cfg.class_names, "XGBoost", sp.name)
            accs.append(r.accuracy * 100)
            f1s.append(r.f1_macro * 100)
            per_class.append(r.per_class_f1)

        row = {
            "severity": sev,
            "fault_to_noise": fc.fault_to_noise,
            "spike_degC_mean": fc.s(0.5 * (fc.spike_min + fc.spike_max)),
            "noise_floor_degC": fc.floor(),
            "Accuracy": np.mean(accs),
            "F1 Macro": np.mean(f1s),
            "F1 std": np.std(f1s),
        }
        for cname in cfg.class_names:
            row[f"F1[{cname}]"] = np.mean([p.get(cname, 0.0) * 100 for p in per_class])
        rows.append(row)
        print(f"  sev={sev:5.2f}  F/N={fc.fault_to_noise:6.2f}x  "
              f"spike={fc.s(0.5*(fc.spike_min+fc.spike_max))*1000:6.1f} mK  "
              f"acc={row['Accuracy']:6.2f}%  F1={row['F1 Macro']:6.2f}%")

    df = pd.DataFrame(rows)
    df.to_csv(out / "detection_limit.csv", index=False)

    print("\n" + "=" * 78)
    print("DETECTION LIMIT CURVE")
    print("=" * 78)
    show = ["severity", "fault_to_noise", "spike_degC_mean", "Accuracy", "F1 Macro"] + \
           [f"F1[{c}]" for c in cfg.class_names]
    print(df[show].round(3).to_string(index=False))

    print("\n" + "-" * 78)
    print(f"MINIMUM DETECTABLE AMPLITUDE (first severity reaching F1 >= {TARGET_F1}%)")
    print("-" * 78)
    limits = []
    for cname in cfg.class_names:
        col = f"F1[{cname}]"
        ok = df[df[col] >= TARGET_F1]
        if len(ok):
            r0 = ok.iloc[0]
            amp = r0["spike_degC_mean"]
            limits.append({"fault": cname, "min_severity": r0["severity"],
                           "fault_to_noise": round(r0["fault_to_noise"], 2),
                           "amplitude_mK": round(amp * 1000, 1)})
            print(f"  {cname:16s} severity>={r0['severity']:5.2f}  "
                  f"F/N>={r0['fault_to_noise']:6.2f}x  ~{amp*1000:7.1f} mK")
        else:
            limits.append({"fault": cname, "min_severity": None,
                           "fault_to_noise": None, "amplitude_mK": None})
            print(f"  {cname:16s} never reaches {TARGET_F1}% in the swept range")
    pd.DataFrame(limits).to_csv(out / "detection_limits.csv", index=False)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
