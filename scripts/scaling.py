"""MPI strong scaling of one optimisation step (flow + heat solves, two adjoints, filter) on a fixed mesh.

Fair comparison: one thread per process and MUMPS parallel analysis/ordering, e.g.
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
  export PETSC_OPTIONS="-flow_mat_mumps_icntl_28 2 -flow_mat_mumps_icntl_29 2 ..."   (same for heat_, adjT_, adjF_, filt_)
  mpirun -x OMP_NUM_THREADS -x PETSC_OPTIONS -n N python scripts/scaling.py 200
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
from fxtopopt import Settings, ThermalPlate  # noqa: E402

comm = MPI.COMM_WORLD
nx = int(sys.argv[1]) if len(sys.argv) > 1 else 200
prob = ThermalPlate(Settings(length=2.0, nx=nx, ny=nx // 2, thermal=True, pe=1e3, p_in=3e3, beta=4.0, a_max=1e6), comm)
prob.set_design(np.full(prob.n_local, 0.35))
prob.evaluate()                                              # warm-up (form compilation)
times = []
for _ in range(2):
    comm.Barrier(); t0 = time.perf_counter(); prob.evaluate(); comm.Barrier()
    times.append(time.perf_counter() - t0)
if comm.rank == 0:
    f = ROOT / "results" / "scaling.json"
    data = json.loads(f.read_text()) if f.exists() else {}
    data[str(comm.size)] = {"ranks": comm.size, "mesh": [nx, nx // 2], "cells": 4 * nx * (nx // 2),
                            "seconds_per_step": float(min(times))}
    f.write_text(json.dumps(data, indent=1))
    print(comm.size, "ranks:", min(times), "s/step")
