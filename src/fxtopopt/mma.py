"""MPI-parallel Method of Moving Asymptotes (Svanberg 1987) with one linear constraint.

Every operation is element-wise on the locally owned design variables except the dual problem's
constraint value a.x, which is a global sum (MPI allreduce). Each rank therefore holds only its own
part of the design vector, and the result is identical for any number of processes.
"""
from __future__ import annotations

import numpy as np
from mpi4py import MPI


class MMA:
    def __init__(self, comm: MPI.Comm, move: float = 0.2):
        self.comm, self.move = comm, move
        self.it, self.x1, self.x2, self.low, self.upp = 0, None, None, None, None

    def _dot(self, a, b) -> float:
        return self.comm.allreduce(float(a @ b), op=MPI.SUM)

    def step(self, x: np.ndarray, df: np.ndarray, a: np.ndarray, b: float) -> np.ndarray:
        """Minimise f s.t. a.x <= b, 0 <= x <= 1, given df = grad f at x (local arrays)."""
        self.it += 1
        if self.it <= 2:
            self.low, self.upp = x - 0.5, x + 0.5
        else:
            sgn = (x - self.x1) * (self.x1 - self.x2)
            gam = np.where(sgn > 0, 1.2, np.where(sgn < 0, 0.7, 1.0))
            self.low = np.clip(x - gam * (self.x1 - self.low), x - 10, x - 0.01)
            self.upp = np.clip(x + gam * (self.upp - self.x1), x + 0.01, x + 10)
        lo = np.maximum.reduce([np.zeros_like(x), self.low + 0.1 * (x - self.low), x - self.move])
        hi = np.minimum.reduce([np.ones_like(x), self.upp - 0.1 * (self.upp - x), x + self.move])
        dfp, dfm = np.maximum(df, 0), np.maximum(-df, 0)
        reg = 1e-3 * np.abs(df) + 1e-6 / (self.upp - self.low)
        p = (self.upp - x) ** 2 * (1.001 * dfp + 0.001 * dfm + reg)
        q = (x - self.low) ** 2 * (0.001 * dfp + 1.001 * dfm + reg)

        def x_of(lam):
            l, h = lo.copy(), hi.copy()
            for _ in range(60):
                m = 0.5 * (l + h)
                pos = p / (self.upp - m) ** 2 - q / (m - self.low) ** 2 + lam * a > 0
                h, l = np.where(pos, m, h), np.where(pos, l, m)
            return 0.5 * (l + h)

        if self._dot(a, x_of(0.0)) <= b:
            xn = x_of(0.0)
        else:
            l, h, k = 0.0, 1.0, 0
            while self._dot(a, x_of(h)) > b and k < 60:
                h *= 2; k += 1
            for _ in range(60):
                m = 0.5 * (l + h)
                l, h = (m, h) if self._dot(a, x_of(m)) > b else (l, m)
            xn = x_of(h)
        self.x2 = self.x1 if self.x1 is not None else x.copy()
        self.x1 = x.copy()
        return xn
