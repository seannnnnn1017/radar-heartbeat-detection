#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mmWave vital waveform reconstruction from JSON logs.

Usage:
  python radar_vitals.py --inputs replay_1.json replay_2.json replay_3.json \
                         --outdir outputs

Requires: numpy, pandas, scipy, matplotlib
"""
import os, json, math, argparse
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt


def load_frames(paths):
    rows = []
    for p in paths:
        if not os.path.exists(p):
            print(f"[warn] file not found: {p}")
            continue
        with open(p, "r", encoding="utf-8") as f:
            root = json.load(f)

        for entry in root.get("data", []):
            fd = entry.get("frameData", {})
            ts = entry.get("timestamp", None)  # ms
            frame_num = fd.get("frameNum", None)

            track = None
            td = fd.get("trackData", None)
            # Accept common shapes: list-of-lists or list-of-dicts
            if isinstance(td, list) and len(td) > 0:
                if isinstance(td[0], list):
                    track = td[0]
                elif isinstance(td[0], dict):
                    # try common keys
                    cand = td[0]
                    if all(k in cand for k in ("x", "y", "z")):
                        track = [cand["x"], cand["y"], cand["z"]]
                    elif "pos" in cand and isinstance(cand["pos"], list) and len(cand["pos"]) >= 3:
                        track = cand["pos"][:3]

            rows.append({
                "source": os.path.basename(p),
                "frame": frame_num,
                "timestamp_ms": ts,
                "track": track,
                "numTracks": fd.get("numDetectedTracks", None),
                "numPoints": fd.get("numDetectedPoints", None)
            })
    df = pd.DataFrame(rows).dropna(subset=["timestamp_ms"])
    df = df.sort_values("timestamp_ms").reset_index(drop=True)
    return df


def extract_xyz(track):
    if track is None:
        return (np.nan, np.nan, np.nan)
    if isinstance(track, (list, tuple)) and len(track) >= 3:
        return (float(track[0]), float(track[1]), float(track[2]))
    return (np.nan, np.nan, np.nan)


def bandpass(sig, fs, f1, f2, order=4):
    ny = 0.5 * fs
    low = max(1e-4, f1 / ny)
    high = min(0.99, f2 / ny)
    b, a = signal.butter(order, [low, high], btype="band")
    return signal.filtfilt(b, a, sig)


def estimate_rate(sig, fs, min_bpm, max_bpm):
    if len(sig) < 10:
        return np.nan
    min_dist = max(1, int(fs * 60.0 / max_bpm))
    prom = np.std(sig) * 0.2
    peaks, _ = signal.find_peaks(sig, distance=min_dist, prominence=prom)
    if len(peaks) < 2:
        return np.nan
    intervals = np.diff(peaks) / fs
    med = np.median(intervals)
    if not np.isfinite(med) or med <= 0:
        return np.nan
    return 60.0 / med


def resample_uniform(t_sec, y, fs_hint=None):
    dt = np.median(np.diff(t_sec))
    if not np.isfinite(dt) or dt <= 0:
        fs = fs_hint if fs_hint else 20.0
    else:
        fs = 1.0 / dt
    fs = float(np.clip(fs, 5.0, 200.0))
    t_uniform = np.arange(t_sec[0], t_sec[-1], 1.0 / fs)
    y_uniform = np.interp(t_uniform, t_sec, y)
    return fs, t_uniform, y_uniform


def plot_series(t, y, title, ylabel, out_png):
    plt.figure(figsize=(10, 3))
    plt.plot(t, y)
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    if out_png:
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        plt.savefig(out_png, dpi=200)
    plt.close()


def main(args):
    df = load_frames(args.inputs)
    if df.empty:
        print("[error] no frames with timestamps found.")
        return

    # Extract xyz and range
    xyz = np.array([extract_xyz(tr) for tr in df["track"]], dtype=float)
    df["x"], df["y"], df["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    df["range_m"] = np.sqrt(df["x"]**2 + df["y"]**2 + df["z"]**2)
    df2 = df.dropna(subset=["range_m", "timestamp_ms"])
    if df2.empty:
        print("[error] no valid trackData with xyz found.")
        return

    # Time axis in seconds
    t = (df2["timestamp_ms"].to_numpy() - df2["timestamp_ms"].iloc[0]) / 1000.0
    r = df2["range_m"].to_numpy()

    # Uniform resample for stable filtering
    fs, t_uni, r_uni = resample_uniform(t, r, fs_hint=args.fs_hint)
    # Detrend
    r_detr = signal.detrend(r_uni, type="linear")

    # Filters
    resp = bandpass(r_detr, fs, 0.1, 0.5)     # 6–30 bpm
    heart = bandpass(r_detr, fs, 0.8, 3.0)    # 48–180 bpm

    # Rate estimates
    rr_bpm = estimate_rate(resp, fs, 6, 30)
    hr_bpm = estimate_rate(heart, fs, 40, 180)

    # Save CSV
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, "vital_series.csv")
    pd.DataFrame({
        "t_sec": t_uni,
        "range_detrended": r_detr,
        "respiration": resp,
        "heartbeat": heart
    }).to_csv(csv_path, index=False)

    # Plots
    plot_series(t_uni, resp,
                f"Respiration-band displacement (est ≈ {rr_bpm:.1f} bpm)" if np.isfinite(rr_bpm) else "Respiration-band displacement",
                "Respiration (a.u.)",
                os.path.join(outdir, "respiration.png"))
    plot_series(t_uni, heart,
                f"Heartbeat-band displacement (est HR ≈ {hr_bpm:.1f} bpm)" if np.isfinite(hr_bpm) else "Heartbeat-band displacement",
                "Heartbeat (a.u.)",
                os.path.join(outdir, "heartbeat.png"))

    # Summary
    summary = {
        "samples": int(len(t_uni)),
        "duration_s": float(t_uni[-1] - t_uni[0]),
        "fs_Hz": float(fs),
        "est_resp_bpm": float(rr_bpm) if np.isfinite(rr_bpm) else None,
        "est_hr_bpm": float(hr_bpm) if np.isfinite(hr_bpm) else None,
        "csv": csv_path,
        "respiration_png": os.path.join(outdir, "respiration.png"),
        "heartbeat_png": os.path.join(outdir, "heartbeat.png")
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="Paths to JSON logs")
    ap.add_argument("--outdir", default="outputs", help="Directory for CSV and PNGs")
    ap.add_argument("--fs_hint", type=float, default=20.0, help="Fallback sampling rate if timestamps are irregular")
    args = ap.parse_args()
    main(args)
