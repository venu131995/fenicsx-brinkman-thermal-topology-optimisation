"""Borrvall-Petersson double-pipe benchmark: domain length 1.0 vs 1.5, fluid fraction 1/3, run under MPI."""
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
from fxtopopt import DoublePipe, Settings, optimise  # noqa: E402

comm = MPI.COMM_WORLD
out, plots = {"mpi_ranks": comm.size}, {}
CASES = [float(a) for a in sys.argv[1:]] or [1.0, 1.5]
for L in CASES:
    s = Settings(length=L, nx=int(60 * L), ny=60, r_filter=0.015)
    prob = DoublePipe(s, comm)
    t0 = time.time()
    # soft interpolation first (q = 0.01) so channels can explore merging, then hard (q = 0.1) + sharpening
    x, J, hist, _ = optimise(prob, 1 / 3, iters=180, betas=(1, 1, 1, 2, 4, 8, 16), every=25,
                             q_schedule=[(0, 0.01), (75, 0.1)], log=lambda m: print(f"[L={L}] {m}", flush=True))
    rho = projected_dg0(prob)
    # reference: two straight pipes joining each inlet to its outlet, same fluid fraction
    mid = cell_midpoints(prob)
    straight = ((np.abs(mid[:, 1] - 0.25) < 1 / 12) | (np.abs(mid[:, 1] - 0.75) < 1 / 12)).astype(float)
    prob.beta.value = 64.0
    prob.qc.value = 0.1
    prob.set_design(straight)
    J_straight, _ = prob.evaluate(gradient=False)
    vol_straight = prob.volume()
    out[f"L={L}"] = {"mesh": [s.nx, s.ny], "cells": comm.allreduce(prob.n_local), "J_optimised": J,
                     "J_two_straight_pipes": J_straight, "vol_straight": vol_straight,
                     "improvement_pct": 100 * (1 - J / J_straight), "runtime_s": time.time() - t0,
                     "history": [h["J"] for h in hist]}
    plots[L] = gather_cells(prob, rho)

if comm.rank == 0 and len(CASES) == 2:
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "double_pipe.json").write_text(json.dumps(out, indent=1))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), gridspec_kw={"width_ratios": [1, 1.5]})
    for ax, L in zip(axs, (1.0, 1.5)):
        tris, vals = plots[L]
        r = out[f"L={L}"]
        draw(ax, tris, vals, title=f"domain length {L}:  power {r['J_optimised']:.2f}\n"
                                    f"(two straight pipes: {r['J_two_straight_pipes']:.2f})")
    fig.suptitle("Borrvall-Petersson double pipe (FEniCSx, Taylor-Hood P2/P1, MPI)", fontsize=10)
    fig.tight_layout(); fig.savefig(ROOT / "figures" / "fig1_double_pipe.png", dpi=140); plt.close(fig)
if comm.rank == 0:
    tag = "_".join(str(c) for c in CASES)
    (ROOT / "results" / f"double_pipe_{tag}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "history"} if isinstance(v, dict) else v
                      for k, v in out.items()}, indent=1))
