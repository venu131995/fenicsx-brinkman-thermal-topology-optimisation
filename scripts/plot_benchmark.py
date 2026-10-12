"""Solver testing on benchmarks (README figure).

Top: Borrvall-Petersson double-pipe set-up (sketch) and the computed designs. Bottom left: two straight pipes vs
the exact Poiseuille value for three solid strengths a_max and three meshes. Bottom right: heated-plates Nusselt
number vs the exact 140/17. Reads results/double_pipe.json, results/straight_pipe_accuracy.json, results/nusselt.json
and figures/fig1_double_pipe.png; writes figures/fig0_benchmark.png.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
dp = json.loads((ROOT / "results" / "double_pipe.json").read_text())
acc = json.loads((ROOT / "results" / "straight_pipe_accuracy.json").read_text())["rows"]
nu = json.loads((ROOT / "results" / "nusselt.json").read_text())["rows"]
img = plt.imread(ROOT / "figures" / "fig1_double_pipe.png")


def frame(ax):
    for s in ax.spines.values():
        s.set_linewidth(1.5)
    ax.tick_params(width=1.5, labelsize=11)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontweight("bold")


fig = plt.figure(figsize=(14, 12))
gs = fig.add_gridspec(3, 3, height_ratios=[1, 1.15, 1.0], width_ratios=[1, 1.5, 1.1])
for k, L in enumerate((1.0, 1.5)):
    a = fig.add_subplot(gs[0, k])
    a.add_patch(Rectangle((0, 0), L, 1, fc="#f4f6f8", ec="k", lw=2.5))
    for c in (0.25, 0.75):
        y = np.linspace(c - 1 / 12, c + 1 / 12, 40)
        u = 1 - ((y - c) * 12) ** 2
        a.fill_betweenx(y, -0.2 * u, 0, color="#3a7bd5", alpha=.8)
        a.fill_betweenx(y, L, L + 0.2 * u, color="#3a7bd5", alpha=.8)
    a.text(-0.13, 0.5, "in", ha="center", va="center", fontsize=13, fontweight="bold")
    a.text(L + 0.13, 0.5, "out", ha="center", va="center", fontsize=13, fontweight="bold")
    a.text(L / 2, 0.5, f"{L:g} x 1 box", ha="center", va="center", fontsize=13, fontweight="bold")
    a.set_xlim(-0.3, L + 0.3); a.set_ylim(-0.05, 1.05); a.set_aspect("equal"); a.axis("off")
    a.set_title(f"Set-up, length {L:g}", fontsize=14, fontweight="bold")

r1, r2 = dp["L=1.0"], dp["L=1.5"]
t = fig.add_subplot(gs[0, 2]); t.axis("off")
t.text(0, 1.02, "Task", fontsize=12, fontweight="bold", va="top")
t.text(0, 0.92, "fixed flow rate, water <= 1/3 of the box,\nminimise pumping power", fontsize=11.5, va="top")
t.text(0, 0.74, "Known answer (Borrvall & Petersson 2003)", fontsize=12, fontweight="bold", va="top")
t.text(0, 0.64, "length 1: two separate pipes\nlength 1.5: pipes merge into one", fontsize=11.5, va="top")
t.text(0, 0.44, "Computed (pumping power)", fontsize=12, fontweight="bold", va="top")
t.text(0, 0.34,f"length 1: {r1['J_optimised']:.2f}  (straight {r1['J_two_straight_pipes']:.2f})\n"
       f"length 1.5: {r2['J_optimised']:.2f}  (straight {r2['J_two_straight_pipes']:.2f},"
       f" {-r2['improvement_pct']:+.0f} %)", fontsize=11.5, va="top")
t.text(0, 0.10, f"Run time, 4 MPI processes: {r1['runtime_s'] / 60:.1f} / {r2['runtime_s'] / 60:.1f} min",
       fontsize=11.5, va="top")

b = fig.add_subplot(gs[1, :]); b.imshow(img); b.axis("off")

c = fig.add_subplot(gs[2, :2])
for a_max, m in ((2.5e4, "o"), (1e6, "s"), (1e8, "^")):
    rows = [r for r in acc if r["a_max"] == a_max]
    c.plot([r["ny"] for r in rows], [r["error_pct"] for r in rows], m + "-", lw=2, ms=8,
           label=f"solid strength a_max = {a_max:.1e}")
c.axhline(0, color="k", lw=1)
c.set_xscale("log"); c.set_xticks([60, 120, 240]); c.set_xticklabels(["60", "120", "240"])
c.minorticks_off()
c.set_xlabel("cells across the box", fontsize=12, fontweight="bold")
c.set_ylabel("error vs exact (%)", fontsize=12, fontweight="bold")
c.set_title("Flow: two straight pipes vs exact Poiseuille value", fontsize=13, fontweight="bold")
c.legend(frameon=True, edgecolor="k", fontsize=10.5, loc="center right")
frame(c)

d = fig.add_subplot(gs[2, 2])
d.plot([r["ny"] for r in nu], [r["Nu"] for r in nu], "o-", lw=2, ms=8, color="#c0392b", label="computed")
d.axhline(140 / 17, color="k", ls="--", lw=1.5, label="exact 8.235")
d.set_xscale("log"); d.set_xticks([10, 20, 40]); d.set_xticklabels(["10", "20", "40"]); d.minorticks_off()
d.set_xlabel("cells across the channel", fontsize=12, fontweight="bold")
d.set_ylabel("Nusselt number", fontsize=12, fontweight="bold")
d.set_title("Heat with flow: heated plates", fontsize=13, fontweight="bold")
d.legend(frameon=True, edgecolor="k", fontsize=10.5)
frame(d)

fig.suptitle("Solver testing on benchmarks", fontsize=17, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(ROOT / "figures" / "fig0_benchmark.png", dpi=110)
