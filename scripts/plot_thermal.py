"""Fig. 2: best straight channels vs. the optimised design (rows: design / temperature), from results/thermal_fields.npz."""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
L, H = 2.0, 1.0
X0, X1, Y0, Y1 = 0.86, 1.34, 0.15, 0.55          # 8x heated region


def hot(a):
    for c, lw, ls in (("w", 4.0, "-"), ("#d7191c", 2.6, (0, (4, 2.5)))):
        a.plot([X0, X1, X1, X0, X0], [Y0, Y0, Y1, Y1, Y0], color=c, lw=lw, ls=ls)


def peak(a, v):
    a.text(L - 0.03, H - 0.04, f"peak {v:.4f}", ha="right", va="top", fontsize=13, fontweight="bold",
           bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="k", lw=1.2))


def field(a, tris, vals, cmap, vmin, vmax):
    pc = PolyCollection(tris, array=vals, cmap=cmap, edgecolors="face", linewidths=0.05)
    pc.set_clim(vmin, vmax); a.add_collection(pc)
    a.set_xlim(0, L); a.set_ylim(0, H); a.set_aspect("equal")
    return pc


def main():
    d = np.load(ROOT / "results" / "thermal_fields.npz")
    vmax = float(max(d["s_T"].max(), d["o_T"].max()))
    water = ListedColormap(["white", "#0b3c8c"])
    fig = plt.figure(figsize=(11.0, 5.2))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.04], hspace=0.08, wspace=0.06)
    cols = [("s", f"Best straight channels (N = {int(d['n_ch'])})"), ("o", "Optimised channels")]
    for c, (k, name) in enumerate(cols):
        a = fig.add_subplot(gs[0, c])
        field(a, d[f"{k}_tris"], (d[f"{k}_rho"] > 0.5).astype(float), water, 0, 1); hot(a)
        a.set_title(name, fontsize=16, fontweight="bold")
        if c == 0: a.set_ylabel("Design", fontsize=14, fontweight="bold")
        a = fig.add_subplot(gs[1, c])
        im = field(a, d[f"{k}_tris"], d[f"{k}_T"], "jet", 0, vmax); hot(a); peak(a, float(d[f"{k}_Tmax"]))
        if c == 0: a.set_ylabel("Temperature", fontsize=14, fontweight="bold")
    for a in fig.axes:
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values(): sp.set_linewidth(1.5)
    fig.add_subplot(gs[0, 2]).axis("off")
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[1, 2]))
    cb.set_label("temperature", fontsize=12, fontweight="bold"); cb.ax.tick_params(labelsize=10, width=1.5)
    for t in cb.ax.get_yticklabels(): t.set_fontweight("bold")
    fig.savefig(ROOT / "figures" / "fig2_thermal_design.png", dpi=200, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
