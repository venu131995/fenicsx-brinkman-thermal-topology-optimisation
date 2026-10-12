"""Accuracy of the Brinkman flow solver against an exact solution: two straight pipes (Poiseuille flow).

Each pipe is 1/6 wide with a parabolic in/outflow of peak speed 1, so with perfectly solid walls
J = 1/2 * integral(mu |grad u|^2) = 16 L / (3 h) * 1/2 * 2 pipes = 32 L  (h = 1/6, mu = 1).
The Brinkman "solid" is slightly permeable (penetration depth ~ 1/sqrt(a_max)), so J approaches 32 L only as a_max
grows and the mesh is refined. Writes results/straight_pipe_accuracy.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from mpi4py import MPI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from common import cell_midpoints  # noqa: E402
from fxtopopt import DoublePipe, Settings  # noqa: E402

comm = MPI.COMM_WORLD
L = 1.0
EXACT = 32.0 * L
rows = []
for ny in (60, 120, 240):
    for a_max in (2.5e4, 1e6, 1e8):
        t0 = time.time()
        prob = DoublePipe(Settings(length=L, nx=ny, ny=ny, r_filter=0.015, a_max=a_max), comm)
        mid = cell_midpoints(prob)
        straight = ((np.abs(mid[:, 1] - 0.25) < 1 / 12) | (np.abs(mid[:, 1] - 0.75) < 1 / 12)).astype(float)
        prob.beta.value = 64.0
        prob.qc.value = 0.1
        prob.set_design(straight)
        J, _ = prob.evaluate(gradient=False)
        r = {"ny": ny, "a_max": a_max, "J": J, "exact": EXACT, "error_pct": 100 * (J / EXACT - 1),
             "vol": prob.volume(), "time_s": time.time() - t0}
        rows.append(r)
        if comm.rank == 0:
            print(json.dumps(r), flush=True)
if comm.rank == 0:
    (ROOT / "results" / "straight_pipe_accuracy.json").write_text(json.dumps({"mpi_ranks": comm.size, "rows": rows}, indent=1))
