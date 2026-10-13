"""MPI strong scaling of one optimisation step (flow + heat solves, two adjoints, filter and its adjoint).

Design: six straight channels at 35 % coolant (sharp solid/fluid contrast, alpha_max = 1e6), as in the thermal study.
Compares the MUMPS direct solver with the iterative solvers (AMG, block-preconditioned MINRES). Run with one
thread per process, e.g.
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  mpirun -n N python scripts/scaling.py NX {lu,iterative}
Each run appends one record to results/scaling.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from mpi4py import MPI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from common import cell_midpoints  # noqa: E402
from fxtopopt import Settings, ThermalPlate  # noqa: E402
from fxtopopt.model import STATS  # noqa: E402

comm = MPI.COMM_WORLD
nx = int(sys.argv[1]) if len(sys.argv) > 1 else 200
solver = sys.argv[2] if len(sys.argv) > 2 else "lu"
prob = ThermalPlate(Settings(length=2.0, nx=nx, ny=nx // 2, thermal=True, pe=1e3, p_in=3e3, r_filter=0.025,
                             beta=8.0, a_max=1e6, solver=solver), comm)
mid, h, n_ch, vol = cell_midpoints(prob), 1.0 / prob.s.ny, 6, 0.35
x = np.zeros(len(mid))
for c in range(n_ch):
    x = np.maximum(x, np.clip((vol / n_ch / 2 - np.abs(mid[:, 1] - (c + 0.5) / n_ch)) / h + 0.5, 0.0, 1.0))

prob.set_design(x)
prob.evaluate()                                              # warm-up (form compilation)
times, steps = [], []
for _ in range(2):
    STATS.clear()
    comm.Barrier(); t0 = time.perf_counter()
    prob.set_design(x)
    J, dJ = prob.evaluate()
    comm.Barrier()
    times.append(time.perf_counter() - t0)
    steps.append({k: dict(v) for k, v in STATS.items()})
best = int(np.argmin(times))
gnorm = np.sqrt(comm.allreduce(float(dJ @ dJ), op=MPI.SUM))
dofs = prob.W.dofmap.index_map.size_global + prob.V1.dofmap.index_map.size_global
if comm.rank == 0:
    f = ROOT / "results" / "scaling.json"
    data = json.loads(f.read_text()) if f.exists() else {}
    if not isinstance(data.get("runs"), list):
        data = {"runs": []}
    rec = {"solver": solver, "ranks": comm.size, "mesh": [nx, nx // 2], "cells": 4 * nx * (nx // 2),
           "dofs_flow_plus_heat": int(dofs), "seconds_per_step": float(times[best]),
           "per_solve": steps[best], "J": float(J), "grad_norm": float(gnorm)}
    data["runs"] = [r for r in data["runs"]
                    if not (r["solver"] == solver and r["ranks"] == comm.size and r["mesh"] == rec["mesh"])] + [rec]
    f.write_text(json.dumps(data, indent=1))
    its = {k: v["its"] for k, v in rec["per_solve"].items()}
    print(f"{solver:9s} {comm.size:3d} ranks  {rec['cells']} cells  {times[best]:.2f} s/step  J {J:.10g}  |dJ| {gnorm:.10g}  its {its}", flush=True)
