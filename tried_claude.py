#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified mmWave vitals merger using load_frames(paths).
- Keeps the file-reading style from the second snippet (load_frames).
- Extracts per-frame vitals.heartWaveform and merges across files.
- Produces stats and a plot.

Usage:
    python mmwave_heart_merge.py /path/to/folder [-o outputs]

In notebooks:
    from mmwave_heart_merge import process_vitals_folder
    summary = process_vitals_folder("data", outdir="outputs")
"""
import os
import re
import glob
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, HTML

# -----------------------------
# I/O helpers
# -----------------------------

def natural_sort_key(filename):
    """自然排序：replay_1, replay_2, ..., replay_10"""
    numbers = re.findall(r'\d+', filename)
    return int(numbers[0]) if numbers else 0

def find_replay_jsons(folder):
    """Find replay_*.json under folder, natural-sorted."""
    pattern = os.path.join(folder, "replay_*.json")
    files = glob.glob(pattern)
    files.sort(key=lambda x: natural_sort_key(os.path.basename(x)))
    return files

# -----------------------------
# Core loader following the "load_frames(paths)" pattern
# -----------------------------

def load_frames(paths):
    """
    Load and parse JSON frame data from multiple files.
    Returns a pandas DataFrame sorted by timestamp.
    Columns:
        source, frame, timestamp_ms,
        heartWaveform, heartRate, breathRate, rangeBin, breathDeviation,
        numTracks, numPoints
    """
    rows = []
    for p in paths:
        if not os.path.exists(p):
            print(f"[warn] file not found: {p}")
            continue
        with open(p, "r", encoding="utf-8") as f:
            root = json.load(f)

        for entry in root.get("data", []):
            fd = entry.get("frameData", {}) or {}
            ts = entry.get("timestamp", None)  # ms
            frame_num = fd.get("frameNum", None)

            vit = fd.get("vitals", None) or {}
            hw = vit.get("heartWaveform", None)
            hr = vit.get("heartRate", None)
            br = vit.get("breathRate", None)
            rb = vit.get("rangeBin", None)
            bd = vit.get("breathDeviation", None)

            rows.append({
                "source": os.path.basename(p),
                "frame": frame_num,
                "timestamp_ms": ts,
                "numTracks": fd.get("numDetectedTracks", None),
                "numPoints": fd.get("numDetectedPoints", None),
                "heartWaveform": hw if isinstance(hw, list) else None,
                "heartRate": hr,
                "breathRate": br,
                "rangeBin": rb,
                "breathDeviation": bd,
            })

    df = pd.DataFrame(rows).dropna(subset=["timestamp_ms"])
    df = df.sort_values(["timestamp_ms", "source", "frame"], kind="mergesort").reset_index(drop=True)
    return df

# -----------------------------
# Processing and plotting
# -----------------------------

def merge_heart_waveforms(df):
    """
    Merge all available heartWaveform arrays in time order.
    Returns:
        wave (np.ndarray), segments (list of dict per source file)
    """
    all_wave = []
    segments = []  # per-file segment boundaries for plotting and stats

    # record cumulative offset when each file starts
    cur_offset = 0
    for src, sub in df.groupby("source", sort=False):
        file_points = 0
        frames_info = []
        for _, row in sub.iterrows():
            wf = row.get("heartWaveform", None)
            if isinstance(wf, list) and len(wf) > 0:
                start = cur_offset + file_points
                all_wave.extend(wf)
                end = cur_offset + file_points + len(wf)
                frames_info.append({
                    "frameNum": int(row["frame"]) if pd.notna(row["frame"]) else None,
                    "timestamp": float(row["timestamp_ms"]),
                    "heartRate": float(row["heartRate"]) if pd.notna(row["heartRate"]) else None,
                    "waveformLength": int(len(wf)),
                    "startIndex": int(start),
                    "endIndex": int(end)
                })
                file_points += len(wf)

        segments.append({
            "filename": src,
            "totalPoints": int(file_points),
            "frameCount": int(len(frames_info)),
            "startIndex": int(cur_offset),
            "endIndex": int(cur_offset + file_points),
            "frame_info": frames_info
        })
        cur_offset += file_points

    return np.asarray(all_wave, dtype=float), segments

def print_statistics(wave, segments, df_meta=None):
    """Print summary statistics and per-file info."""
    if wave.size == 0:
        print("[warn] No heartWaveform data found.")
        return

    print("\n" + "="*60)
    print("統計資訊")
    print("="*60)
    print(f"總資料點數: {len(wave)}")
    print(f"最大振幅: {np.max(wave):.4f}")
    print(f"最小振幅: {np.min(wave):.4f}")
    print(f"平均振幅: {np.mean(wave):.4f}")
    print(f"標準差: {np.std(wave):.4f}\n")

    print("檔案詳情:")
    print("-"*60)
    for seg in segments:
        print(f"\n檔案: {seg['filename']}")
        print(f"  資料點: {seg['totalPoints']}")
        print(f"  Frame數: {seg['frameCount']}")
        print(f"  索引範圍: [{seg['startIndex']}, {seg['endIndex']})")

        hrs = [fi["heartRate"] for fi in seg["frame_info"] if fi["heartRate"] is not None and fi["heartRate"] > 0]
        if hrs:
            print(f"  平均心率: {np.mean(hrs):.2f} bpm")
            print(f"  心率範圍: {np.min(hrs):.2f} - {np.max(hrs):.2f} bpm")

def plot_heart_waveform(wave, segments, fs=None, t=None, save_path=None):
    """
    Plot merged heart waveform with time axis in seconds if fs or t is provided.
    """
    if wave.size == 0:
        return

    # --- time axis (seconds) ---
    if t is None:
        if fs is not None and fs > 0:
            t = np.arange(len(wave), dtype=float) / float(fs)
        else:
            t = np.arange(len(wave), dtype=float)  # falls back to index

    # Primary waveform
    plt.figure(figsize=(16, 5))
    plt.plot(t, wave, linewidth=0.8, alpha=0.9)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Merged heart waveform")
    plt.grid(True, alpha=0.3)

    # Mark file boundaries using time axis
    ymax = float(np.max(wave)) if wave.size else 1.0
    for seg in segments:
        idx = seg["startIndex"]
        bx = t[idx] if 0 <= idx < len(t) else (idx/float(fs) if fs else idx)
        plt.axvline(x=bx, linestyle="--", alpha=0.4, linewidth=1.0)
        plt.text(bx, ymax, seg["filename"], rotation=90, va="top", fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()

    # Per-file counts chart（此圖仍為點數統計，維持不變）
    plt.figure(figsize=(12, 4))
    counts = [s["totalPoints"] for s in segments]
    labels = [s["filename"] for s in segments]
    x = np.arange(len(counts))
    plt.bar(x, counts, alpha=0.8)
    plt.xticks(x, labels, rotation=45, ha="right", fontsize=9)
    plt.ylabel("Points")
    plt.title("Per-file heart waveform points")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    if save_path:
        base, ext = os.path.splitext(save_path)
        plt.savefig(base + "_counts" + ext, dpi=200, bbox_inches="tight")
    plt.show()

# -----------------------------
# High-level API
# -----------------------------

def process_vitals_folder(folder_path, outdir="outputs", fs_hint=10.0):
    """
    Find and load replay_*.json, merge vitals.heartWaveform, print stats, and plot.
    Returns summary with keys:
      samples, duration_s, fs_Hz, est_hr_bpm, csv, heartbeat_png, dataframe
    """
    files = find_replay_jsons(folder_path)
    if not files:
        print(f"[error] No replay_*.json files found in: {folder_path}")
        return None

    print(f"Found {len(files)} files:")
    for f in files:
        print("  -", os.path.basename(f))
    print()

    df = load_frames(files)
    if df.empty:
        print("[error] No frames with timestamps found.")
        return None

    wave, segments = merge_heart_waveforms(df)
    if wave.size == 0:
        print("[error] No heartWaveform data found in frames.")
        return None

    # 輸出資料夾
    os.makedirs(outdir, exist_ok=True)

    # 以 fs_hint 產生時間軸（秒）
    fs = float(fs_hint)
    t_uni = np.arange(len(wave), dtype=float) / fs if len(wave) > 0 else np.array([], dtype=float)

    # 輸出 CSV：t_sec 與 heartbeat
    csv_path = os.path.join(outdir, "heart_rate_series.csv")
    result_df = pd.DataFrame({
        "t_sec": t_uni,
        "heartbeat": wave.astype(float)
    })
    result_df.to_csv(csv_path, index=False)

    # 圖：x 軸以秒顯示
    heart_png = os.path.join(outdir, "heartbeat.png")
    print_statistics(wave, segments, df_meta=df)
    plot_heart_waveform(wave, segments, fs=fs, t=t_uni, save_path=heart_png)

    # 固定回傳結構
    summary = {
        "samples": int(len(t_uni)),
        "duration_s": float(t_uni[-1] - t_uni[0]) if len(t_uni) > 1 else 0.0,
        "fs_Hz": float(fs),
        "est_hr_bpm": None,
        "csv": csv_path,
        "heartbeat_png": heart_png,
        "dataframe": result_df,
    }
    return summary

# -----------------------------
# CLI
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="Folder containing replay_*.json")
    ap.add_argument("-o", "--outdir", default="outputs", help="Output directory")
    args = ap.parse_args()

    summary = process_vitals_folder(args.folder, outdir=args.outdir)
    if summary is None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
