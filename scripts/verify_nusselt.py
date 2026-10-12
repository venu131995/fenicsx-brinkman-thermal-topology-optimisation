"""Coupled flow + heat benchmark: laminar flow between two parallel plates, both walls heated with a uniform flux.

Uses the production ThermalPlate solver (Stokes-Brinkman, Taylor-Hood P2/P1, SUPG heat) on an all-fluid design:
the flow must be plane Poiseuille (mean speed p_in H^2 / (12 mu L)) and, once thermally developed, the Nusselt
number based on the hydraulic diameter D_h = 2H must equal 140/17 = 8.235 (Shah & London 1978).
Also checks global energy (wall heat in = heat advected out + conducted into the inlet) and mass balance.
Writes results/nusselt.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import dolfinx.fem.petsc as fpetsc
import ufl
from dolfinx import fem
from mpi4py import MPI
from petsc4py import PETSc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fxtopopt import Settings, ThermalPlate  # noqa: E402
from fxtopopt.model import dxq, solve_linear  # noqa: E402

NU_EXACT = 140.0 / 17.0
L, QW, K_F = 12.0, 1.0, 0.05
P_IN = 12.0 * L                       # mean speed 1 (H = 1, mu = 1)
X0, X1 = 8.0, 10.0                    # thermally developed slab (entry length ~ 0.05 Pe D_h << 8)


class HeatedPlates(ThermalPlate):
    def __init__(self, s, comm):
        super().__init__(s, comm)
        self.Q = fem.Constant(self.msh, PETSc.ScalarType(0.0))
        self.dsb = ufl.Measure("ds", domain=self.msh)
        X = ufl.SpatialCoordinate(self.msh)
        self.on_wall = ufl.conditional(ufl.Or(ufl.lt(X[1], 1e-8), ufl.gt(X[1], 1 - 1e-8)), 1.0, 0.0)
        self.in_slab = ufl.conditional(ufl.And(ufl.gt(X[0], X0), ufl.lt(X[0], X1)), 1.0, 0.0)

    def heat_residual(self, w, T, s_):
        return super().heat_residual(w, T, s_) - QW * self.on_wall * s_ * self.dsb


def tot(prob, form):
    return prob.comm.allreduce(fem.assemble_scalar(fem.form(form)), op=MPI.SUM)


comm = MPI.COMM_WORLD
rows = []
for ny in (10, 20, 40):
    t0 = time.time()
    s = Settings(length=L, nx=int(L * ny), ny=ny, thermal=True, pe=1.0, p_in=P_IN, k_fluid=K_F, beta=8.0)
    prob = HeatedPlates(s, comm)
    prob.set_design(np.ones(prob.n_local))
    # same linear solves as ThermalPlate.evaluate
    vq, dw = ufl.TestFunction(prob.W), ufl.TrialFunction(prob.W)
    RF = prob.flow_residual(prob.w, vq)
    solve_linear(ufl.lhs(ufl.replace(RF, {prob.w: dw})), ufl.rhs(ufl.replace(RF, {prob.w: dw})), prob.bcs, prob.w, "flow_")
    sT, dT = ufl.TestFunction(prob.V1), ufl.TrialFunction(prob.V1)
    RT = ufl.replace(prob.heat_residual(prob.w, prob.T, sT), {prob.T: dT})
    solve_linear(ufl.lhs(RT), ufl.rhs(RT), prob.bcs_T, prob.T, "heat_")
    u, _ = ufl.split(prob.w)
    n = ufl.FacetNormal(prob.msh)
    q_out = tot(prob, ufl.inner(u, n) * prob.ds(2))
    q_in = -tot(prob, ufl.inner(u, n) * prob.ds(1))
    T, sl, wl = prob.T, prob.in_slab, prob.on_wall
    Tb = tot(prob, sl * u[0] * T * dxq) / tot(prob, sl * u[0] * dxq)           # mixing-cup temperature
    Tw = tot(prob, sl * wl * T * prob.dsb) / tot(prob, sl * wl * prob.dsb)       # wall temperature
    nu = QW * 2.0 / (K_F * (Tw - Tb))
    heat_in = QW * 2 * L
    adv_out = tot(prob, s.pe * T * ufl.inner(u, n) * prob.ds(2))
    # heat conducted back out through the inlet: consistent (residual-based) flux at the Dirichlet dofs,
    # the same construction as fxtopopt.conservation
    Rv = fpetsc.assemble_vector(fem.form(prob.heat_residual(prob.w, T, ufl.TestFunction(prob.V1))))
    Rv.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    nloc = prob.V1.dofmap.index_map.size_local
    arr = Rv.array_r[:nloc]
    inl = prob.inlet_T_dofs[prob.inlet_T_dofs < nloc]
    cond_in = -prob.comm.allreduce(float(arr[inl].sum()), op=MPI.SUM)
    free = np.setdiff1d(np.arange(nloc), inl)
    res_local = prob.comm.allreduce(float(np.abs(arr[free]).max()), op=MPI.MAX)
    Rv.destroy()
    r = {"ny": ny, "nx": int(L * ny), "mean_speed": q_in, "mean_speed_exact": 1.0,
         "flow_error_pct": 100 * (q_in - 1.0), "mass_imbalance_rel": (q_out - q_in) / q_in,
         "Nu": nu, "Nu_exact": NU_EXACT, "Nu_error_pct": 100 * (nu / NU_EXACT - 1),
         "energy_closure_rel": (adv_out + cond_in - heat_in) / heat_in,
         "heat_nodal_residual_max_rel": res_local / (heat_in / prob.comm.allreduce(nloc)), "time_s": time.time() - t0}
    rows.append(r)
    if comm.rank == 0:
        print(json.dumps(r), flush=True)
if comm.rank == 0:
    (ROOT / "results" / "nusselt.json").write_text(json.dumps({"mpi_ranks": comm.size, "Pe_Dh": 2.0 / K_F, "rows": rows}, indent=1))
