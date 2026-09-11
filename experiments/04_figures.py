"""Generate every figure used in the README and the report.

    python experiments/04_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from sfd import Config, build_dataset
from sfd.data import load_channel
from sfd.evaluate import evaluate
from sfd.models import build_models
from sfd.plots import confusion, detection_limit, protocol_comparison, real_glitch
from sfd.splits import leave_one_vehicle_out


def main() -> None:
    cfg = Config()
    cfg.validate()
    exp = cfg.results_root / "experiments"
    figs = cfg.results_root / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    made = []

    # 1. Detection limit curve
    src = exp / "detection_limit.csv"
    if src.exists():
        made.append(detection_limit(src, figs / "detection_limit.png"))
    else:
        print(f"  skip detection_limit - run experiments/03 first ({src} missing)")

    # 2. Protocol comparison
    src = exp / "protocol_pivot.csv"
    if src.exists():
        made.append(protocol_comparison(src, figs / "protocol_comparison.png"))
    else:
        print("  skip protocol_comparison - run experiments/01 first")

    # 3. Confusion matrix at the operating point, held-out vehicle
    ds = build_dataset(cfg, verbose=False)
    sp = next(iter(leave_one_vehicle_out(ds.ws)))
    Xtr, Xte, ytr, yte = sp.apply(ds.X, ds.y)
    model = build_models(seed=cfg.seed, fast=True)["XGBoost"]
    r = evaluate(model, Xtr, ytr, Xte, yte, cfg.class_names, "XGBoost", sp.name)
    made.append(confusion(r.confusion, cfg.class_names, figs / "confusion_matrix.png",
                          f"XGBoost, {sp.name}  (F1 macro {r.f1_macro * 100:.2f}%)"))

    # 4. The real acquisition fault found in the raw data
    sig = load_channel(cfg, "PVS2", "left", "temp_dashboard")
    zeros = np.flatnonzero(sig == 0)
    if len(zeros):
        made.append(real_glitch(sig, int(zeros[0]), figs / "real_glitch.png"))

    for p in made:
        print(f"  wrote {p.relative_to(cfg.results_root)}")


if __name__ == "__main__":
    main()
