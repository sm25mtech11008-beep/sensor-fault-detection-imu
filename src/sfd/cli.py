"""Command line interface.

    sfd stats                      signal statistics per temperature channel
    sfd audit                      quantify leakage in each split protocol
    sfd train --protocol vehicle   train and persist the best model
    sfd predict --input x.csv      score a CSV of raw sensor readings
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .config import TEMP_CHANNELS, Config
from .data import signal_statistics, sliding_windows
from .evaluate import evaluate, results_frame
from .features import FEATURE_NAMES, extract
from .models import build_models
from .pipeline import build_dataset
from .splits import PROTOCOLS, leakage_report, leave_one_vehicle_out, make_split


def cmd_stats(args, cfg: Config) -> int:
    cfg.validate()
    rows = []
    for ch in TEMP_CHANNELS:
        st = signal_statistics(cfg, ch)
        rows.append({"channel": ch, **st})
        print(f"  {ch:24s} global_std={st['global_std']:8.4f}  "
              f"window_std={st['window_std_median']:8.5f}  "
              f"ratio={st['global_std'] / st['window_std_median']:6.1f}x")
    df = pd.DataFrame(rows)
    out = cfg.results_root / "channel_stats.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.round(5).to_csv(out, index=False)
    ratio = df["global_std"].iloc[0] / df["window_std_median"].iloc[0]
    print("\nThe ratio matters: scaling injected faults by the global std makes them")
    print(f"about {ratio:.0f}x larger than natural in-window variation.")
    print(f"Saved {out}")
    return 0


def cmd_audit(args, cfg: Config) -> int:
    cfg.validate()
    ds = build_dataset(cfg)
    print(f"\n{'protocol':28s} {'test':>8} {'shared drives':>14} {'overlap %':>10}")
    print("-" * 64)
    for p in PROTOCOLS:
        sp = make_split(p, ds.ws, ds.y, cfg)
        r = leakage_report(ds.ws, sp)
        print(f"{sp.name:28s} {r['test_windows']:>8,} {r['shared_drives']:>14} "
              f"{r['pct_test_overlapping']:>9.1f}%")
    return 0


def cmd_train(args, cfg: Config) -> int:
    cfg.validate()
    ds = build_dataset(cfg)
    models = build_models(seed=cfg.seed, fast=args.fast)
    if args.model and args.model not in models:
        print(f"unknown model {args.model!r}; choose from {list(models)}", file=sys.stderr)
        return 2
    chosen = {args.model: models[args.model]} if args.model else models

    results = []
    if args.protocol == "vehicle":
        for sp in leave_one_vehicle_out(ds.ws):
            Xtr, Xte, ytr, yte = sp.apply(ds.X, ds.y)
            for name, m in chosen.items():
                results.append(evaluate(m, Xtr, ytr, Xte, yte, cfg.class_names, name, sp.name))
    else:
        sp = make_split(args.protocol, ds.ws, ds.y, cfg)
        Xtr, Xte, ytr, yte = sp.apply(ds.X, ds.y)
        for name, m in chosen.items():
            results.append(evaluate(m, Xtr, ytr, Xte, yte, cfg.class_names, name, sp.name))

    df = results_frame(results)
    print()
    print(df.groupby("Model")[["Accuracy", "Balanced Acc", "F1 Macro", "MCC"]]
          .mean().round(3).to_string())

    best_name = df.groupby("Model")["F1 Macro"].mean().idxmax()
    print(f"\nbest: {best_name}")

    # Refit the winner on everything and persist it with its metadata.
    final = build_models(seed=cfg.seed, fast=args.fast)[best_name]
    final.fit(ds.X, ds.y)
    out_dir = cfg.results_root / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": final,
        "feature_names": list(FEATURE_NAMES),
        "class_names": cfg.class_names,
        "channel": cfg.channel,
        "window_size": cfg.window_size,
        "window_step": cfg.window_step,
        "protocol": args.protocol,
        "cv_f1_macro": float(df.groupby("Model")["F1 Macro"].mean()[best_name]),
        "signal_stats": ds.stats,
        "fault_config": ds.fault_config,
    }
    path = out_dir / "model.pkl"
    with open(path, "wb") as fh:
        pickle.dump(bundle, fh)
    df.to_csv(out_dir / "train_results.csv", index=False)
    print(f"saved {path}")
    return 0


def cmd_predict(args, cfg: Config) -> int:
    path = cfg.results_root / "models" / "model.pkl"
    if not path.exists():
        print(f"no trained model at {path}; run 'sfd train' first", file=sys.stderr)
        return 2
    with open(path, "rb") as fh:
        bundle = pickle.load(fh)

    src = Path(args.input)
    if not src.exists():
        print(f"input not found: {src}", file=sys.stderr)
        return 2
    col = cfg.channel or bundle["channel"]
    df = pd.read_csv(src, usecols=[col])
    sig = df[col].dropna().to_numpy(dtype=np.float32)

    w = sliding_windows(sig, bundle["window_size"], bundle["window_step"])
    if len(w) == 0:
        print(f"signal too short: need at least {bundle['window_size']} samples", file=sys.stderr)
        return 2

    X = extract(w)
    pred = bundle["model"].predict(X)
    names = bundle["class_names"]

    counts = pd.Series([names[p] for p in pred]).value_counts()
    print(f"scored {len(pred):,} windows from {src.name} ({col})")
    print()
    for name, n in counts.items():
        print(f"  {name:16s} {n:7,}  ({100 * n / len(pred):5.1f}%)")

    if args.out:
        pd.DataFrame({
            "window": np.arange(len(pred)),
            "start_sample": np.arange(len(pred)) * bundle["window_step"],
            "prediction": [names[p] for p in pred],
        }).to_csv(args.out, index=False)
        print(f"\nsaved {args.out}")
    return 0


def cmd_download(args, cfg: Config) -> int:
    print("The PVS dataset is mirrored by the authors at UFSC (no login required):")
    print("  https://arquivos.ufsc.br/d/55ee83c2c9d647eaa252/")
    print()
    print("Fetch just the CSVs (about 1.1 GB; the videos are ~48 GB and unused):")
    print("  bash scripts/download_pvs.sh")
    print()
    print(f"Expected layout under {cfg.data_root}:")
    print("  PVS1/dataset_mpu_left.csv, PVS1/dataset_mpu_right.csv, ... through PVS9/")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="sfd", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=None, help="override dataset location")
    p.add_argument("--channel", default=None, help="sensor column to model")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("download", help="how to obtain the dataset").set_defaults(fn=cmd_download)
    sub.add_parser("stats", help="per-channel signal statistics").set_defaults(fn=cmd_stats)
    sub.add_parser("audit", help="quantify leakage per split protocol").set_defaults(fn=cmd_audit)

    t = sub.add_parser("train", help="train and persist a model")
    t.add_argument("--protocol", default="vehicle", choices=PROTOCOLS)
    t.add_argument("--model", default=None)
    t.add_argument("--fast", action="store_true", help="skip SVM and KNN")
    t.set_defaults(fn=cmd_train)

    q = sub.add_parser("predict", help="score a raw sensor CSV")
    q.add_argument("--input", required=True)
    q.add_argument("--out", default=None, help="write per-window predictions here")
    q.set_defaults(fn=cmd_predict)

    args = p.parse_args(argv)
    kw = {}
    if args.data_root:
        kw["data_root"] = args.data_root
    if args.channel:
        kw["channel"] = args.channel
    cfg = Config(**kw)
    return args.fn(args, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
