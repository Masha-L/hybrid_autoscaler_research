#!/usr/bin/env python3
"""
run_simulation.py  (v2: with optional visualisation)
----------------------------------------------------
Replay a workload trace through multiple CPU-scaling controllers,
export per-run metrics to CSV, and—optionally—plot aggregated results.

Example
-------
python run_simulation.py \
    --workload data/trace.csv \
    --controllers reactive predictive_hourly predictive_arima predictive_ml hybrid \
    --model-coeffs linear_model.json \
    --runs 10 \
    --outfile data/results.csv \
    --plot-file data/results.png
"""
from __future__ import annotations

import argparse, json, pathlib, sys
import pandas as pd
import matplotlib.pyplot as plt         # only for plotting
from cluster import Cluster


# ──────────────────────────────────────────────────────────────────────────────
# Controller factory
# ──────────────────────────────────────────────────────────────────────────────
def load_controller(name: str, trace: pd.Series, coeffs_file: str | None):
    if name == "reactive":
        from controllers.reactive import Reactive
        return Reactive()

    elif name == "predictive_hourly":
        from controllers.predictive_hourly import PredictiveHourly
        hourly = trace.resample("1H").mean().values
        return PredictiveHourly(hourly)

    elif name == "predictive_arima":
        from controllers.predictive_arima import PredictiveARIMA
        return PredictiveARIMA()

    elif name == "predictive_ml":
        if coeffs_file is None:
            raise ValueError("--model-coeffs required for predictive_ml")
        from controllers.predictive_ml import load_from_json
        return load_from_json(coeffs_file)

    elif name == "hybrid":
        from controllers.hybrid import Hybrid
        hourly = trace.resample("1H").mean().values
        return Hybrid(hourly)

    else:
        raise ValueError(f"Unknown controller '{name}'")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Simulate workload and plot results")
    p.add_argument("--workload", required=True, help="CSV with timestamp,requests_per_sec")
    p.add_argument("--controllers", nargs="+",
                   default=["reactive", "predictive_hourly",
                            "predictive_arima", "predictive_ml", "hybrid"],
                   help="Controller IDs to evaluate")
    p.add_argument("--runs", type=int, default=10, help="Independent repetitions")
    p.add_argument("--model-coeffs", help="JSON with {a,b[,cpu_per_vm]} for predictive_ml")
    p.add_argument("--outfile", default="results.csv", help="Result CSV path")
    p.add_argument("--plot-file", default="results.png",
                   help="If given, write bar chart to this PNG (requires matplotlib)")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
def plot_aggregates(df: pd.DataFrame, png: str):
    """Create side-by-side bar charts for latency, SLA %, and wasted CPU-hours."""
    means = (df.groupby("controller")
               .agg(latency_ms=("latency_ms", "mean"),
                    sla_violation=("sla_violation", "mean"),
                    wasted_cpu_h=("wasted_cpu_h", "mean"))
               .sort_index())

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), sharex=True)
    means["latency_ms"].plot(kind="bar", ax=axes[0], color="#4C72B0")
    axes[0].set_ylabel("Mean latency (ms)")
    axes[0].set_title("Latency ↓")

    means["sla_violation"].plot(kind="bar", ax=axes[1], color="#55A868")
    axes[1].set_ylabel("SLA violations (%)")
    axes[1].set_title("SLA-violation ↓")

    means["wasted_cpu_h"].plot(kind="bar", ax=axes[2], color="#C44E52")
    axes[2].set_ylabel("Wasted CPU-hours")
    axes[2].set_title("Wasted CPU ↓")

    for ax in axes:
        ax.set_xlabel("")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(png, dpi=150)
    print(f"[✓] plot saved → {png}")


# ──────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # 1. Load workload
    trace = (pd.read_csv(args.workload, parse_dates=["timestamp"], index_col="timestamp")
               ["requests_per_sec"])

    # 2. Simulate each controller
    rows = []
    for ctrl_name in args.controllers:
        print(f"[+] controller: {ctrl_name}")
        for run in range(args.runs):
            # instantiate a fresh controller for each run so that internal state
            # from previous simulations does not leak across repetitions
            ctrl = load_controller(ctrl_name, trace, args.model_coeffs)
            sim  = Cluster(trace, ctrl)
            summ = sim.run()
            summ.update(controller=ctrl_name, run=run)
            rows.append(summ)

    df = pd.DataFrame(rows)
    pathlib.Path(args.outfile).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.outfile, index=False)
    print(f"[✓] wrote {len(df)} rows → {args.outfile}")

    # 3. Optional visualisation
    if args.plot_file:
        plot_aggregates(df, args.plot_file)


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        print(f"Fatal: {ex}", file=sys.stderr)
        sys.exit(1)
