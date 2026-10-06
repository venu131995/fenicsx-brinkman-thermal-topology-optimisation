"""Method of manufactured solutions: convergence rates of the Brinkman (Taylor-Hood P2/P1) and the
SUPG convection-diffusion (P1) discretisations used in the optimiser.

Exact fields on the unit square:
    psi = sin^2(pi x) sin^2(pi y),  u = (d psi/dy, -d psi/dx)   (divergence-free, zero on the boundary)
    p   = cos(pi x) cos(pi y)
    T   = sin(pi x) sin(pi y) + x y   (non-trivial Dirichlet data)
Sources are derived symbolically with UFL, so the forms differentiated are exactly those solved.
Expected rates: |u|_L2 ~ h^3, |u|_H1 ~ h^2, |p|_L2 ~ h^2, |T|_L2 ~ h^2, |T|_H1 ~ h.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import basix.ufl
import numpy as np
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI
from petsc4py import PETSc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fxtopopt.model import dxq, solve_linear  # noqa: E402

comm = MPI.COMM_WORLD
ALPHA, MU, PE, K = 50.0, 1.0, 20.0, 1.0


def errnorm(e, kind="L2"):
    f = ufl.inner(e, e) if kind == "L2" else ufl.inner(ufl.grad(e), ufl.grad(e))
    return np.sqrt(comm.allreduce(fem.assemble_scalar(fem.form(f * dxq)), op=MPI.SUM))


def run(n):
    msh = mesh.create_unit_square(comm, n, n, mesh.CellType.triangle, diagonal=mesh.DiagonalType.crossed)
    X = ufl.SpatialCoordinate(msh)
    pi = ufl.pi
    psi = ufl.sin(pi * X[0]) ** 2 * ufl.sin(pi * X[1]) ** 2
    ue = ufl.as_vector((psi.dx(1), -psi.dx(0)))
    pe_ = ufl.cos(pi * X[0]) * ufl.cos(pi * X[1])
    f = ALPHA * ue - MU * ufl.div(ufl.grad(ue)) + ufl.grad(pe_)
    cell = msh.basix_cell()
    W = fem.functionspace(msh, basix.ufl.mixed_element([basix.ufl.element("Lagrange", cell, 2, shape=(2,)),
                                                         basix.ufl.element("Lagrange", cell, 1)]))
    (u, p), (v, q) = ufl.TrialFunctions(W), ufl.TestFunctions(W)
    a = (ALPHA * ufl.inner(u, v) + MU * ufl.inner(ufl.grad(u), ufl.grad(v)) - p * ufl.div(v) - q * ufl.div(u)) * dxq
    L = ufl.inner(f, v) * dxq
    msh.topology.create_connectivity(1, 2)
    bf = mesh.exterior_facet_indices(msh.topology)
    W0, _ = W.sub(0).collapse()
    zero = fem.Function(W0)                                   # exact velocity is zero on the boundary
    bcs = [fem.dirichletbc(zero, fem.locate_dofs_topological((W.sub(0), W0), 1, bf), W.sub(0))]
    w = fem.Function(W)
    solve_linear(a, L, bcs, w, f"mms{n}_")
    uh, ph = w.sub(0).collapse(), w.sub(1).collapse()
    area = 1.0
    pmean = comm.allreduce(fem.assemble_scalar(fem.form((ph - pe_) * dxq)), op=MPI.SUM) / area
    eu_L2, eu_H1 = errnorm(uh - ue), errnorm(uh - ue, "H1")
    ep_L2 = errnorm(ph - pe_ - pmean)

    # convection-diffusion with the same SUPG form as the optimiser, driven by the computed velocity uh
    V1 = fem.functionspace(msh, ("Lagrange", 1))
    Te = ufl.sin(pi * X[0]) * ufl.sin(pi * X[1]) + X[0] * X[1]
    Q = PE * ufl.inner(uh, ufl.grad(Te)) - K * ufl.div(ufl.grad(Te))
    T, s = ufl.TrialFunction(V1), ufl.TestFunction(V1)
    h = ufl.CellDiameter(msh)
    un = ufl.sqrt(ufl.inner(uh, uh) + 1e-12)
    tau = 1.0 / ufl.sqrt((2 * PE * un / h) ** 2 + (12 * K / h**2) ** 2)
    aT = (PE * ufl.inner(uh, ufl.grad(T)) * s + K * ufl.inner(ufl.grad(T), ufl.grad(s))) * dxq \
        + tau * PE * ufl.inner(uh, ufl.grad(T)) * PE * ufl.inner(uh, ufl.grad(s)) * dxq
    LT = Q * s * dxq + tau * (Q + K * ufl.div(ufl.grad(Te))) * PE * ufl.inner(uh, ufl.grad(s)) * dxq
    # (P1 strong residual omits the diffusion term on both sides consistently: element-wise lap(T_h) = 0)
    Tbc = fem.Function(V1)
    Tbc.interpolate(fem.Expression(Te, V1.element.interpolation_points))
    bcT = [fem.dirichletbc(Tbc, fem.locate_dofs_topological(V1, 1, bf))]
    Th = fem.Function(V1)
    solve_linear(aT, LT, bcT, Th, f"mmsT{n}_")
    return {"n": n, "h": 1.0 / n, "u_L2": eu_L2, "u_H1": eu_H1, "p_L2": ep_L2,
            "T_L2": errnorm(Th - Te), "T_H1": errnorm(Th - Te, "H1")}


rows = [run(n) for n in (8, 16, 32, 64)]
rates = {k: [float(np.log2(rows[i][k] / rows[i + 1][k])) for i in range(len(rows) - 1)]
         for k in ("u_L2", "u_H1", "p_L2", "T_L2", "T_H1")}
if comm.rank == 0:
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "mms_convergence.json").write_text(json.dumps({"rows": rows, "rates": rates}, indent=1))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 4))
    h = np.array([r["h"] for r in rows])
    for k, mk in [("u_L2", "o"), ("u_H1", "s"), ("p_L2", "^"), ("T_L2", "v"), ("T_H1", "d")]:
        ax.loglog(h, [r[k] for r in rows], mk + "-", label=f"{k}  (rate {np.mean(rates[k][-2:]):.2f})")
    ax.set_xlabel("mesh size h"); ax.set_ylabel("error vs exact solution"); ax.legend(fontsize=7)
    ax.set_title("Manufactured-solution convergence (FEniCSx)")
    fig.tight_layout(); fig.savefig(ROOT / "figures" / "fig4_mms_convergence.png", dpi=140)
    print(json.dumps(rates, indent=1))
