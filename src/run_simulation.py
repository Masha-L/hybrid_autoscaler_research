#!/usr/bin/env python3
"""
run_simulation.py
-----------------
Replay a synthetic (or real) RPS trace through multiple CPU-scaling
controllers and export summary metrics.

Example
-------
python run_simulation.py \
        --workload data/trace.csv \
        --controllers reactive predictive_hourly predictive_arima \
                       predictive_ml hybrid \
        --model-coeffs linear_model.json \
        --runs 10 \
        --outfile data/results.csv
"""
from __future__ import annotations
import argparse, importlib, json, pathlib, sys
import pandas as pd
from cluster import Cluster


# ---------------------------------------------------------------------------#
# Controller factory                                                          #
# ---------------------------------------------------------------------------#
def load_controller(name: str,
                    trace: pd.Series,
                    coeffs_file: str | None = None):
    """Return an instantiated controller object."""
    if name == "reactive":
        from controllers.reactive import Reactive
        return Reactive()

    elif name == "predictive_hourly":
        from controllers.predictive_hourly import PredictiveHourly
        hourly = trace.resample("1H").mean().values       # 24-element
        return PredictiveHourly(hourly)

    elif name == "predictive_arima":
        from controllers.predictive_arima import PredictiveARIMA
        return PredictiveARIMA()

    elif name == "predictive_ml":
        if coeffs_file is None:
            raise ValueError("predictive_ml requires --model-coeffs JSON file")
        from controllers.predictive_ml import load_from_json
        return load_from_json(coeffs_file)

    elif name == "hybrid":
        from controllers.hybrid import Hybrid
        hourly = trace.resample("1H").mean().values
        return Hybrid(hourly)

    else:
        raise ValueError(f"Unknown controller: {name}")


# ---------------------------------------------------------------------------#
# CLI                                                                         #
# ---------------------------------------------------------------------------#
def parse_args():
    p = argparse.ArgumentParser(description="Simulate workload under multiple CPU-scaling controllers")
    p.add_argument("--workload", required=True, help="CSV with timestamp,requests_per_sec")
    p.add_argument("--controllers", nargs="+", default=["reactive","predictive_hourly","predictive_arima","predictive_ml","hybrid"],
                   help="List of controller IDs to test")
    p.add_argument("--runs", type=int, default=10, help="Independent seeds / noise realisations")
    p.add_argument("--model-coeffs", help="JSON file for predictive_ml {a,b[,cpu_per_vm]}")
    p.add_argument("--outfile", default="results.csv", help="Result CSV path")
    return p.parse_args()


# ---------------------------------------------------------------------------#
# Main                                                                        #
# ---------------------------------------------------------------------------#
def main():
    args = parse_args()

    # 1. Load workload trace
    df_trace = pd.read_csv(args.workload, parse_dates=["timestamp"], index_col="timestamp")
    series   = df_trace["requests_per_sec"]

    # 2. Run each controller N times
    rows = []
    for ctrl_name in args.controllers:
        print(f"[+] Running controller: {ctrl_name}")
        ctrl_obj = load_controller(ctrl_name, series, args.model_coeffs)

        for run in range(args.runs):
            cluster = Cluster(series, ctrl_obj)
            summary = cluster.run()
            summary.update({"controller": ctrl_name, "run": run})
            rows.append(summary)

    # 3. Save results
    out = pathlib.Path(args.outfile)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[✓] wrote {len(rows)} rows → {out}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal: {e}", file=sys.stderr)
        sys.exit(1)
