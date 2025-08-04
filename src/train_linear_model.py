#!/usr/bin/env python3
"""
Fit linear y = a·t + b to workload traces.

Supports three modes:
  • single  → exactly one day; error if more.
  • per-day → separate (a,b) for every calendar day.
  • global  → one line over entire file (multiple days allowed).

Usage examples
--------------
# Fit one model for each day:
python train_linear_model.py --trace data/long_trace.csv --mode per-day

# Fit global line across the whole span:
python train_linear_model.py --trace data/long_trace.csv --mode global
"""
from __future__ import annotations
import argparse, json, pathlib
import pandas as pd
import numpy as np


def fit_linear(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float]:
    x_m, y_m = xs.mean(), ys.mean()
    a = ((xs - x_m) @ (ys - y_m)) / ((xs - x_m) ** 2).sum()
    b = y_m - a * x_m
    return float(a), float(b)


def process_single_day(df: pd.DataFrame):
    secs = (df["timestamp"] - df["timestamp"].dt.normalize()).dt.total_seconds()
    return fit_linear(secs.values, df["requests_per_sec"].values)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trace", required=True,
                   help="CSV with timestamp,requests_per_sec")
    p.add_argument("--mode", choices=["single", "per-day", "global"],
                   default="single", help="Fitting strategy (default=single)")
    p.add_argument("--outfile", default="linear_model.json")
    p.add_argument("--cpu-per-vm", type=int, default=4)
    args = p.parse_args()

    df = pd.read_csv(args.trace, parse_dates=["timestamp"])
    if "requests_per_sec" not in df.columns:
        raise ValueError("trace missing 'requests_per_sec' column")

    df.sort_values("timestamp", inplace=True)
    df["date"] = df["timestamp"].dt.date
    unique_days = df["date"].unique()

    if args.mode == "single":
        if len(unique_days) != 1:
            raise ValueError("--mode single requires exactly one day "
                             f"(trace has {len(unique_days)})")
        a, b = process_single_day(df)
        out = {"a": a, "b": b, "cpu_per_vm": args.cpu_per_vm}

    elif args.mode == "per-day":
        day_models = []
        for day in unique_days:
            sub = df[df["date"] == day]
            a, b = process_single_day(sub)
            day_models.append({"date": str(day), "a": a, "b": b})
        out = {"days": day_models, "cpu_per_vm": args.cpu_per_vm}

    elif args.mode == "global":
        secs = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()
        a, b = fit_linear(secs.values, df["requests_per_sec"].values)
        out = {"a": a, "b": b, "cpu_per_vm": args.cpu_per_vm}

    # write JSON
    pathlib.Path(args.outfile).write_text(json.dumps(out, indent=2))
    print(f"[✓] coefficients → {args.outfile}")


if __name__ == "__main__":
    main()
