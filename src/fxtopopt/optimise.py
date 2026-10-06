"""Generic MMA loop with beta continuation for any Problem with .evaluate()."""
from __future__ import annotations

import time

import numpy as np

from .mma import MMA


def optimise(prob, vol_frac: float, iters: int = 150, betas=(1, 2, 4, 8, 16), every: int = 30,
             x0: float | None = None, snapshot_every: int = 0, log=None, q_schedule=None):
    """q_schedule: optional list of (start_iteration, q) for Brinkman-interpolation continuation."""
    n = prob.n_local
    x = np.full(n, vol_frac if x0 is None else x0)
    prob.set_design(x)
    mma = MMA(prob.comm, move=0.2)
    hist, snaps, J0, t0 = [], [], None, time.time()
    for it in range(iters):
        prob.beta.value = betas[min(it // every, len(betas) - 1)]
        if q_schedule:
            prob.qc.value = [q for start, q in q_schedule if it >= start][-1]
        J, dJ = prob.evaluate()
        J0 = J0 or J
        vol = prob.volume()
        dv = prob.volume_gradient()
        b = vol_frac - vol + prob.comm.allreduce(float(dv @ x), op=__import__("mpi4py").MPI.SUM)
        if snapshot_every and it % snapshot_every == 0:
            snaps.append((it, float(prob.beta.value), x.copy()))
        x = mma.step(x, dJ / J0, dv, b)
        prob.set_design(x)
        hist.append({"it": it, "J": J, "vol": vol, "beta": float(prob.beta.value), "t": time.time() - t0})
        if log and prob.comm.rank == 0 and it % 10 == 0:
            log(f"it {it:3d}  beta {prob.beta.value:4.0f}  J {J:.5g}  vol {vol:.3f}")
    J, _ = prob.evaluate(gradient=False)
    return x, J, hist, snaps
