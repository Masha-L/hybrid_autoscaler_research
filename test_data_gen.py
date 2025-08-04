#!/usr/bin/env python3
"""generate_workload.py
Generate a synthetic, bursty request‑per‑second trace for CPU‑scaling experiments.

Features
--------
* 24‑hour timeline at 1‑second resolution.
* Two configurable Gaussian spikes (e.g., breakfast & lunch).
* Base traffic + pink‑noise (1/f) variability + optional white Gaussian noise.
* Deterministic output with --seed for reproducibility.
* CSV output: `timestamp,requests_per_sec` in UTC.

Usage
-----
>>> python generate_workload.py \
        --date 2025-09-01 \
        --peak-hours 8 12 \
        --base-rate 400 \
        --peak-multiplier 3.0  \
        --noise-std 0.15 \
        --outfile nyt_trace.csv

Tweaking Times / Spikes
-----------------------
* Use `--peak-hours 7 13 --peak-width 1800` to shift and widen surges.
* Set `--noise-std 0` to disable white noise and retain only pink variability.
"""
import argparse
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Pink-noise generator (1/f) via FFT weighting
# ---------------------------------------------------------------------------

def pink_noise(n: int) -> np.ndarray:
    """Return *n* pink-noise samples (mean≈0, std≈1).

    Method: generate white noise in the frequency domain and scale by
    1/sqrt(f) to obtain 1/f magnitude. Uses rFFT; *n* may be any
    positive integer. Output is real-valued and normalised.
    """
    rng = np.random.default_rng()
    # Real FFT frequency bins (length n//2+1)
    freqs = np.fft.rfftfreq(n)
    freqs[0] = 1.0  # avoid division by zero at DC
    spectrum = (rng.standard_normal(len(freqs)) + 1j * rng.standard_normal(len(freqs))) / np.sqrt(freqs)
    y = np.fft.irfft(spectrum, n)
    return (y - y.mean()) / y.std()



# Core generation
# ---------------------------------------------------------------------------

def generate_trace(date: str,
                   base_rate: int,
                   peak_hours: tuple,
                   peak_multiplier: float,
                   peak_width: int,
                   noise_std: float,
                   seed: int):
    """Return a pandas DataFrame with columns [timestamp, requests_per_sec]."""
    np.random.seed(seed)

    # Timeline: 86_400 seconds
    dt = datetime.fromisoformat(date).replace(tzinfo=timezone.utc)
    seconds = np.arange(86_400)
    timestamps = [dt + timedelta(seconds=int(s)) for s in seconds]

    # Base + pink noise
    traffic = base_rate + pink_noise(len(seconds)) * base_rate * 0.1

    # Two Gaussian spikes
    for ph in peak_hours:
        center = ph * 3600  # seconds
        spike = peak_multiplier * base_rate * np.exp(-0.5 * ((seconds - center) / peak_width) ** 2)
        traffic += spike

    # White noise
    if noise_std > 0:
        traffic += np.random.randn(len(seconds)) * base_rate * noise_std

    traffic = np.clip(traffic, 0, None).round().astype(int)
    return pd.DataFrame({"timestamp": timestamps, "requests_per_sec": traffic})

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic workload trace")
    parser.add_argument("--date", default=datetime.now(timezone.utc).date().isoformat(),
                        help="UTC date for trace (YYYY-MM-DD)")
    parser.add_argument("--base-rate", type=int, default=300,
                        help="Baseline requests per second")
    parser.add_argument("--peak-hours", type=int, nargs="*", default=[8, 12],
                        help="Hours (0‑23) for demand spikes")
    parser.add_argument("--peak-multiplier", type=float, default=3.0,
                        help="Multiplier applied at spike center relative to base rate")
    parser.add_argument("--peak-width", type=int, default=1800,
                        help="Spike width (sigma) in seconds; larger → wider spike")
    parser.add_argument("--noise-std", type=float, default=0.1,
                        help="Std‑dev of white noise as fraction of base rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--outfile", default="synthetic_trace.csv", help="CSV output filename")
    args = parser.parse_args()

    df = generate_trace(args.date,
                        args.base_rate,
                        tuple(args.peak_hours),
                        args.peak_multiplier,
                        args.peak_width,
                        args.noise_std,
                        args.seed)

    df.to_csv(args.outfile, index=False)
    print(df)
    print(f"[+] Written {len(df):,} rows to {args.outfile}")

if __name__ == "__main__":
    main()
