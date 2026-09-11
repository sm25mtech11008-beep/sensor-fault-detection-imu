"""Experiment 2 - What did v1's correlation pruning actually cost?

v1 computed 13 features, then dropped any with |r| > 0.90 against another,
leaving 8: mean, std, min, max, skewness, kurtosis, trend, unique_ratio.
That rule removed ``max_diff`` and ``mean_abs_diff`` - the two statistics
that isolate an impulsive spike - and v1's worst confusion was precisely
normal vs noise_fault.

This experiment holds everything else fixed and varies only the feature set.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from sfd import Config, build_dataset
from sfd.evaluate import evaluate
from sfd.features import FEATURE_NAMES
from sfd.models import build_models
from sfd.splits import leave_one_vehicle_out

V1_FEATURES = ["mean", "std", "min", "max", "skewness", "kurtosis", "trend", "unique_ratio"]
SPIKE_FEATURES = ["max_diff", "mean_abs_diff"]

SETS = {
    "v1 (8 feats, pruned)": V1_FEATURES,
    "v1 + spike detectors": V1_FEATURES + SPIKE_FEATURES,
    "full (16 feats)": list(FEATURE_NAMES),
}


def cols(names: list[str]) -> np.ndarray:
    return np.array([FEATURE_NAMES.index(n) for n in names])


def main() -> None:
    cfg = Config()
    cfg.validate()
    out = cfg.results_root / "experiments"
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("EXPERIMENT 2 - FEATURE SET ABLATION")
    print("=" * 78)
    ds = build_dataset(cfg)
    print()

    results = []
    for set_name, feats in SETS.items():
        idx = cols(feats)
        print("-" * 78)
        print(f"FEATURE SET: {set_name}  ({len(feats)} features)")
        print("-" * 78)
        for sp in leave_one_vehicle_out(ds.ws):
            Xtr, Xte, ytr, yte = sp.apply(ds.X[:, idx], ds.y)
            for name, model in build_models(seed=cfg.seed, fast=True).items():
                if name in ("Majority Baseline", "Decision Stump"):
                    continue
                r = evaluate(model, Xtr, ytr, Xte, yte, cfg.class_names, name, sp.name)
                r.model = f"{set_name} | {name}"
                results.append({
                    "Feature set": set_name, "n_feats": len(feats),
                    "Model": name, "Fold": sp.name,
                    "Accuracy": r.accuracy * 100, "F1 Macro": r.f1_macro * 100,
                    **{f"F1[{k}]": v * 100 for k, v in r.per_class_f1.items()},
                })
                print(f"  {sp.name:24s} {name:22s} acc={r.accuracy*100:6.2f}%  "
                      f"F1={r.f1_macro*100:6.2f}%")

    df = pd.DataFrame(results)
    df.to_csv(out / "feature_ablation_raw.csv", index=False)

    print("\n" + "=" * 78)
    print("ABLATION SUMMARY - mean over 3 leave-one-vehicle-out folds")
    print("=" * 78)
    summ = df.groupby(["Feature set", "Model"])[
        ["Accuracy", "F1 Macro", "F1[normal]", "F1[noise_fault]",
         "F1[bias_fault]", "F1[flatline_fault]"]
    ].mean().round(2)
    print(summ.to_string())
    summ.to_csv(out / "feature_ablation_summary.csv")

    print("\n" + "-" * 78)
    print("COST OF PRUNING - per model, full minus v1")
    print("-" * 78)
    piv = df.pivot_table(index="Model", columns="Feature set", values="F1 Macro")
    piv["gain (full - v1)"] = piv["full (16 feats)"] - piv["v1 (8 feats, pruned)"]
    print(piv.round(2).to_string())
    piv.to_csv(out / "feature_ablation_pivot.csv")


if __name__ == "__main__":
    main()
