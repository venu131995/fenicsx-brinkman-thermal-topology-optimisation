"""Thermal cold-plate topology optimisation (Brinkman + SUPG convection-diffusion) vs straight channels, under MPI."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from mpi4py import MPI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))

from common import cell_midpoints, draw, gather_cells, projected_dg0  # noqa: E402
from fxtopopt import Settings, ThermalPlate, conservation, optimise  # noqa: E402

comm = MPI.COMM_WORLD
VOL = 0.35
# a_max = 1e6: solid must be (nearly) impermeable at this pressure drop -- with the textbook 2.5e4 a third of the
# coolant seeped through "metal" and the optimiser exploited it (see README). Leakage is reported for every design.
S = Settings(length=2.0, nx=120, ny=60, thermal=True, pe=1e3, p_in=3e3, r_filter=0.025, a_max=1e6)
QUICK = "--quick" in sys.argv                      # smoke test: tiny mesh, few iterations, no outputs kept
ITERS = 200
if QUICK:
    S.nx, S.ny, ITERS = 40, 20, 8


def straight(prob, n, vol):
    """n straight channels of width vol/n; edge cells get fractional (anti-aliased) coverage so the coolant
    fraction varies continuously with width."""
    mid = cell_midpoints(prob)
    h = 1.0 / prob.s.ny
    d = np.zeros(len(mid))
    for c in range(n):
        d = np.maximum(d, np.clip((vol / n / 2 - np.abs(mid[:, 1] - (c + 0.5) / n)) / h + 0.5, 0.0, 1.0))
    return d


def T_field_local(prob):
    from dolfinx import fem
    f = fem.Function(prob.V0)
    f.interpolate(fem.Expression(prob.T, prob.V0.element.interpolation_points))
    return f.x.array[: prob.n_local].copy()


prob = ThermalPlate(S, comm)
t0 = time.time()
x, J, hist, snaps = optimise(prob, VOL, iters=ITERS, betas=(1, 1, 2, 4, 8, 16, 32), every=28, snapshot_every=2,
                             q_schedule=[(0, 0.01), (56, 0.1)], log=lambda m: print(m, flush=True))
runtime = time.time() - t0
rho = projected_dg0(prob)
crisp = (rho > 0.5).astype(float)
prob.beta.value = 64.0
prob.qc.value = 0.1
prob.set_design(crisp)
Jc, _ = prob.evaluate(gradient=False)
diag = prob.diagnostics()
cons = conservation(prob)
opt_plot = (gather_cells(prob, crisp), gather_cells(prob, T_field_local(prob)))
vol_c = prob.volume()

# straight-channel baselines at EXACTLY the optimised design's final coolant fraction (widths by bisection),
# pushed through the same filter + projection pipeline
base = []
for n in range(2, 13):
    lo, hi = 0.0, 1.0 / n
    for _ in range(22):
        w = 0.5 * (lo + hi)
        prob.set_design(straight(prob, n, w * n))
        lo, hi = (w, hi) if prob.volume() < vol_c else (lo, w)
    prob.set_design(straight(prob, n, w * n))
    J, _ = prob.evaluate(gradient=False)
    base.append({"channels": n, "width": w, "width_cells": w * S.ny, "fin_cells": (1 / n - w) * S.ny,
                 "J": J, **prob.diagnostics(), "vol": prob.volume()})
    if comm.rank == 0:
        print("baseline", n, base[-1]["T_max"], base[-1]["vol"], flush=True)
best = min(base, key=lambda r: r["T_max"])
feas = [r for r in base if min(r["width_cells"], r["fin_cells"]) >= 2 * S.r_filter * S.ny]
best_f = min(feas, key=lambda r: r["T_max"]) if feas else best
prob.set_design(straight(prob, best["channels"], best["width"] * best["channels"])); prob.evaluate(gradient=False)
best_plot = (gather_cells(prob, straight(prob, best["channels"], best["width"] * best["channels"])),
             gather_cells(prob, T_field_local(prob)))

# frames for the animation (rank 0 collects designs + temperatures)
frames = []
for it, beta, xs in snaps[::2]:
    prob.beta.value = beta
    prob.qc.value = 0.01 if it < 56 else 0.1
    prob.set_design(xs); prob.evaluate(gradient=False)
    frames.append((it, beta, gather_cells(prob, projected_dg0(prob)), gather_cells(prob, T_field_local(prob)),
                   prob.diagnostics()["T_max"]))

if comm.rank == 0 and QUICK:
    print("QUICK OK", diag, best["channels"], best["T_max"])
if comm.rank == 0 and not QUICK:
    res = {"mpi_ranks": comm.size, "mesh": [S.nx, S.ny], "cells": int(len(best_plot[0][1])),
           "settings": {"pe": S.pe, "p_in": S.p_in, "k_solid": S.k_solid, "k_fluid": S.k_fluid, "vol": VOL,
                        "r_filter": S.r_filter, "a_max": S.a_max},
           "straight_channels_equal_volume": base, "best_straight": best, "best_straight_feature_ge_filter": best_f,
           "optimised_crisp": {"T_max": diag["T_max"], "J": Jc, "flow_rate": diag["flow_rate"], "vol": vol_c,
                               "flow_through_solid_fraction": diag["flow_through_solid_fraction"]},
           "conservation_optimised": cons,
           "T_max_reduction_vs_best_straight_pct": 100 * (1 - diag["T_max"] / best["T_max"]),
           "T_max_reduction_vs_best_feasible_straight_pct": 100 * (1 - diag["T_max"] / best_f["T_max"]),
           "optimisation_runtime_s": runtime, "history": hist}
    (ROOT / "results" / "thermal.json").write_text(json.dumps(res, indent=1, default=float))
    import pickle
    (ROOT / "results" / "thermal_frames.pkl").write_bytes(pickle.dumps(
        {"frames": frames, "best": best, "opt": opt_plot, "straight": best_plot, "hist": hist}))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    vmax = max(best["T_max"], diag["T_max"])
    fig, axs = plt.subplots(2, 2, figsize=(10, 5.6))
    draw(axs[0, 0], *best_plot[0], title=f"Best straight channels (N = {best['channels']})")
    draw(axs[0, 1], *opt_plot[0], title="Topology-optimised (FEniCSx, Brinkman)")
    draw(axs[1, 0], *best_plot[1], cmap="inferno", vmax=vmax, title=f"T_max = {best['T_max']:.4f}")
    pc = draw(axs[1, 1], *opt_plot[1], cmap="inferno", vmax=vmax, title=f"T_max = {diag['T_max']:.4f}")
    fig.colorbar(pc, ax=axs[1, :], shrink=0.8, label="temperature")
    fig.suptitle("Pressure-driven cold plate, 8x hotspot, 35% coolant (coolant enters left)", fontsize=10)
    fig.savefig(ROOT / "figures" / "fig2_thermal_design.png", dpi=140, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    ns = [b["channels"] for b in base]
    ax[0].plot(ns, [b["T_max"] for b in base], "o-", label="straight channels")
    ax[0].axhline(diag["T_max"], color="tab:red", ls="--", label="topology-optimised")
    ax[0].set_xlabel("number of straight channels"); ax[0].set_ylabel("T_max"); ax[0].legend(fontsize=8)
    ax[0].set_title("Viscous friction gives straight channels an optimum")
    ax[1].semilogy(ns, [b["flow_rate"] for b in base], "s-", color="tab:green")
    ax[1].set_xlabel("number of straight channels"); ax[1].set_ylabel("flow rate at fixed pressure drop")
    ax[1].set_title("Narrow channels choke the flow (Brinkman wall friction)")
    fig.tight_layout(); fig.savefig(ROOT / "figures" / "fig3_straight_channel_tradeoff.png", dpi=140); plt.close(fig)
    print(json.dumps({k: v for k, v in res.items() if k not in ("history", "straight_channels_equal_volume")}, indent=1, default=float))
