"""Animation of the FEniCSx thermal cold-plate optimisation (reads results/thermal_frames.pkl from run_thermal.py)."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import imageio_ffmpeg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import animation  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

d = pickle.loads((ROOT / "results" / "thermal_frames.pkl").read_bytes())
frames, best, hist = d["frames"], d["best"], d["hist"]
tris = frames[0][2][0]
Tmax_hist = np.array([f[4] for f in frames])
its = np.array([f[0] for f in frames])
vmax = float(best["T_max"])                 # same colour scale as the best straight-channel design

fig = plt.figure(figsize=(10, 5.6), facecolor="white")
gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 0.8], hspace=0.35, wspace=0.06)
ax_d, ax_t, ax_c = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
pcs = []
for ax, cmap, lim, ttl in [(ax_d, "Blues", 1.0, "Channel layout (blue = coolant)"), (ax_t, "inferno", vmax, "Temperature")]:
    pc = PolyCollection(tris, array=np.zeros(len(tris)), cmap=cmap, edgecolors="face", linewidths=0.05)
    pc.set_clim(0, lim); ax.add_collection(pc); pcs.append(pc)
    ax.set_xlim(tris[:, :, 0].min(), tris[:, :, 0].max()); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(ttl, fontsize=10)
ax_d.annotate("coolant in (fixed pressure) →", (0.0, -0.1), xycoords="axes fraction", fontsize=8)
ax_c.axhline(best["T_max"], color="0.4", ls="--", lw=1, label=f"Best straight channels (N={best['channels']}): {best['T_max']:.3f}")
line, = ax_c.plot([], [], color="tab:orange", lw=1.8, label="Topology-optimised design")
ax_c.set_xlim(0, its.max()); ax_c.set_ylim(0, float(Tmax_hist.max()) * 1.08)
ax_c.axvspan(54, 60, color="0.85", zorder=0)
ax_c.annotate("material model hardened\n(Brinkman q: 0.01 → 0.1)", (60, float(Tmax_hist.max()) * 0.8), fontsize=7)
ax_c.set_xlabel("Optimisation iteration"); ax_c.set_ylabel("Hotspot T_max"); ax_c.legend(fontsize=8, loc="upper right")
title = fig.suptitle("", fontsize=11)


def update(k):
    it, beta, (_, rho), (_, T), tmax = frames[k]
    pcs[0].set_array(rho); pcs[1].set_array(T)
    line.set_data(its[: k + 1], Tmax_hist[: k + 1])
    title.set_text(f"FEniCSx Brinkman-flow + heat topology optimisation (MPI)   iteration {it}   beta = {beta:.0f}")
    return pcs + [line, title]


seq = list(range(len(frames))) + [len(frames) - 1] * 15
animation.FuncAnimation(fig, update, frames=seq, interval=80).save(
    ROOT / "figures" / "thermal_optimisation_movie.mp4", writer=animation.FFMpegWriter(fps=10, bitrate=2400), dpi=120)
animation.FuncAnimation(fig, update, frames=seq[::2], interval=140).save(
    ROOT / "figures" / "thermal_optimisation_movie.gif", writer=animation.PillowWriter(fps=7), dpi=70)
print("frames", len(frames))
