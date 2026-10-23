"""Recreate the stage-256 waveform figure from the preserved native samples.

Run in `nix develop .#benchmarking -c python3 <this file>`.
"""
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
records = json.loads((root / "chain256-crossings.json").read_text())
colors = ["#215ca2", "#6b9ace", "#d68d22", "#6651a3", "#111111", "#269076"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
for record, color in zip(records, colors):
    label = record["label"]
    filename = label.lower().replace(" ", "-") + "-s256.csv.gz"
    with gzip.open(root / filename, "rt") as file:
        data = np.loadtxt(file, delimiter=",", skiprows=1)
    for ax in axes:
        ax.plot(data[:, 0] * 1e9, data[:, 1], label=label, color=color,
                lw=1.4, ls="--" if "refined" in label else "-")
for ax in axes:
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Stage 256 voltage (V)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.18)
    ax.axhline(.9, color="#aaaaaa", lw=.7)
axes[0].set_xlim(0, 50)
axes[0].set_title("256-stage inverter chain, original circuit")
axes[1].set_xlim(24, 26)
axes[1].set_title("First rising edge; no time alignment")
baseline = records[0]["rising_crossings_seconds"][0] * 1e9
refined = records[-2]["rising_crossings_seconds"][0] * 1e9
axes[1].annotate("", xy=(baseline, .9), xytext=(refined, .9),
                 arrowprops={"arrowstyle": "<->", "color": "#333333"})
axes[1].text((baseline + refined) / 2, 1.02, f"{(baseline - refined) * 1000:.0f} ps",
             ha="center", fontsize=10)
axes[0].legend(fontsize=8)
fig.savefig(root / "chain256-accuracy.png", dpi=180)
fig.savefig(root / "chain256-accuracy.svg")
