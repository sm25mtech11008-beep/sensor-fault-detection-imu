"""Experiment 1 - How much does the v1 evaluation protocol inflate results?

Trains every model under four protocols of increasing strictness and reports
the gap. This is the core claim of the project: the original 90.8 % figure is
an artefact of evaluating on windows that overlap the training set.

    python experiments/01_leakage_study.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from sfd import Config, build_dataset
from sfd.evaluate import evaluate, results_frame
from sfd.models import build_models
from sfd.splits import leakage_report, leave_one_vehicle_out, make_split

PROTOCOLS = ["random", "disjoint", "grouped", "vehicle"]


def main() -> None:
    cfg = Config()
    cfg.validate()
    out = cfg.results_root / "experiments"
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("EXPERIMENT 1 - EVALUATION PROTOCOL / DATA LEAKAGE STUDY")
    print("=" * 78)
    print(f"channel = {cfg.channel}   window = {cfg.window_size} @ step {cfg.window_step}")
    print()

    ds = build_dataset(cfg)
    print(f"\ndataset: {len(ds):,} windows x {ds.X.shape[1]} features, "
          f"{cfg.n_classes} classes\n")

    # ── Leakage audit ────────────────────────────────────────────────────
    print("-" * 78)
    print("LEAKAGE AUDIT")
    print("-" * 78)
    audit = []
    for p in PROTOCOLS:
        sp = make_split(p, ds.ws, ds.y, cfg)
        rep = leakage_report(ds.ws, sp)
        audit.append({"Protocol": sp.name, **rep})
        print(f"  {sp.name:26s} test={rep['test_windows']:6,}  "
              f"shared drives={rep['shared_drives']:2d}  "
              f"overlapping train={rep['pct_test_overlapping']:5.1f}%")
    pd.DataFrame(audit).to_csv(out / "leakage_audit.csv", index=False)

    # ── Protocol x model sweep ───────────────────────────────────────────
    results = []
    for p in PROTOCOLS:
        sp = make_split(p, ds.ws, ds.y, cfg)
        Xtr, Xte, ytr, yte = sp.apply(ds.X, ds.y)
        print("\n" + "-" * 78)
        print(f"PROTOCOL: {sp.name}   train={len(ytr):,}  test={len(yte):,}")
        print("-" * 78)
        for name, model in build_models(seed=cfg.seed).items():
            r = evaluate(model, Xtr, ytr, Xte, yte, cfg.class_names, name, sp.name)
            results.append(r)
            print(f"  {name:22s} acc={r.accuracy*100:6.2f}%  "
                  f"F1={r.f1_macro*100:6.2f}%  MCC={r.mcc:6.3f}  "
                  f"fit={r.fit_s:5.1f}s")

    df = results_frame(results)
    df.to_csv(out / "protocol_sweep.csv", index=False)

    # ── Headline comparison ──────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("HEADLINE - accuracy (%) by protocol")
    print("=" * 78)
    pivot = df.pivot_table(index="Model", columns="Protocol", values="Accuracy")
    order = [c for c in
             ["random (leaky)", "disjoint windows", "grouped drives (fold 0)"]
             if c in pivot.columns] + \
            [c for c in pivot.columns if c.startswith("leave-out")]
    pivot = pivot[order]
    pivot["INFLATION"] = pivot[order[0]] - pivot[order[-1]]
    print(pivot.sort_values("INFLATION", ascending=False).to_string())
    pivot.to_csv(out / "protocol_pivot.csv")

    # ── Full leave-one-vehicle-out ───────────────────────────────────────
    print("\n" + "=" * 78)
    print("LEAVE-ONE-VEHICLE-OUT (all 3 folds) - the number to quote")
    print("=" * 78)
    logo = []
    for sp in leave_one_vehicle_out(ds.ws):
        Xtr, Xte, ytr, yte = sp.apply(ds.X, ds.y)
        for name, model in build_models(seed=cfg.seed, fast=True).items():
            r = evaluate(model, Xtr, ytr, Xte, yte, cfg.class_names, name, sp.name)
            logo.append(r)
            print(f"  {sp.name:24s} {name:22s} acc={r.accuracy*100:6.2f}%  "
                  f"F1={r.f1_macro*100:6.2f}%")
    ldf = results_frame(logo)
    ldf.to_csv(out / "leave_one_vehicle_out.csv", index=False)

    print("\n" + "-" * 78)
    print("LOVO summary (mean +/- std across the 3 held-out vehicles)")
    print("-" * 78)
    summ = ldf.groupby("Model")[["Accuracy", "F1 Macro", "Balanced Acc"]].agg(["mean", "std"])
    print(summ.round(2).to_string())
    summ.to_csv(out / "lovo_summary.csv")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
