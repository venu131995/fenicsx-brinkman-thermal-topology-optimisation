"""FEniCSx (DOLFINx) Stokes-Brinkman flow and convection-diffusion heat transfer for topology optimisation.

Design field  x  (DG0, one value per cell, 1 = fluid, 0 = solid)
  -> Helmholtz PDE filter      -r^2 lap(xf) + xf = x                  (P1, parallel-friendly)
  -> smoothed Heaviside        rho = H_beta(xf)
  -> Brinkman inverse permeability  alpha(rho) = a_max * q (1 - rho) / (q + rho)   (Borrvall & Petersson 2003)

Flow (Taylor-Hood P2/P1):  alpha u - mu lap u + grad p = 0,  div u = 0
Heat (P1, SUPG):           Pe u.grad T - div(k(rho) grad T) = Q

All objectives and residuals are UFL forms. Gradients come from the discrete adjoint generated
symbolically by UFL (ufl.derivative / ufl.adjoint) and solved with PETSc, followed by the adjoint
of the Helmholtz filter. No hand-coded derivatives.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import basix.ufl
import dolfinx.fem.petsc as fpetsc
import numpy as np
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI
from petsc4py import PETSc

QDEG = 6                                    # fixed quadrature: identical integration in solves and residual checks
dxq = ufl.dx(metadata={"quadrature_degree": QDEG})
LU = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
# Iterative alternatives (Settings.solver = "iterative"): scalable in parallel, no factorisation.
AMG = {"pc_type": "hypre", "pc_hypre_type": "boomeramg"}
ITERATIVE = {
    "cg": {"ksp_type": "cg", "ksp_rtol": 1e-12, "ksp_atol": 1e-50, **AMG},                     # Helmholtz filter (SPD)
    "gmres": {"ksp_type": "gmres", "ksp_gmres_restart": 200, "ksp_rtol": 1e-12, "ksp_atol": 1e-50, **AMG},  # heat
    # Stokes-Brinkman: FGMRES with an upper block-triangular Schur preconditioner. The Schur complement is
    # approximated by -B diag(A)^-1 B^T ("selfp"), which contains alpha and so adapts to solid (Darcy-like) and
    # fluid (Stokes-like) regions; AMG on both blocks. A plain weighted pressure mass matrix needed > 1,000 iterations.
    "stokes": {"ksp_type": "fgmres", "ksp_gmres_restart": 300, "ksp_rtol": 1e-12, "ksp_atol": 1e-50,
               "pc_type": "fieldsplit", "pc_fieldsplit_type": "schur", "pc_fieldsplit_schur_fact_type": "upper",
               "pc_fieldsplit_schur_precondition": "selfp",
               "fieldsplit_u_ksp_type": "preonly", "fieldsplit_u_pc_type": "hypre",
               "fieldsplit_u_pc_hypre_type": "boomeramg",
               "fieldsplit_p_ksp_type": "preonly", "fieldsplit_p_pc_type": "hypre",
               "fieldsplit_p_pc_hypre_type": "boomeramg"},
}
STATS: dict = {}                            # prefix -> {"seconds": accumulated time, "its": last iteration count}


def solve_linear(a, L, bcs, u, prefix, opts=None, a_pc=None, fields=None, nullspace=None):
    """Solve a(u, v) = L(v). Default: MUMPS LU with iterative refinement. With opts/a_pc/fields: a Krylov
    method, an optional separate preconditioner form and named index sets for a fieldsplit preconditioner."""
    t0 = time.perf_counter()
    opts = opts or LU
    A = fpetsc.assemble_matrix(fem.form(a), bcs=bcs)
    A.assemble()
    P = None
    if a_pc is not None:
        P = fpetsc.assemble_matrix(fem.form(a_pc), bcs=bcs)
        P.assemble()
    b = fpetsc.assemble_vector(fem.form(L))
    fpetsc.apply_lifting(b, [fem.form(a)], bcs=[bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    for bc in bcs:
        bc.set(b.array_w)
    if nullspace is not None:
        A.setNullSpace(nullspace)
        nullspace.remove(b)
    ksp = PETSc.KSP().create(u.function_space.mesh.comm)
    ksp.setOptionsPrefix(prefix)
    o = PETSc.Options()
    for k, v in opts.items():
        o[prefix + k] = v
    ksp.setOperators(A, P if P is not None else A)
    if fields:
        ksp.getPC().setType("fieldsplit")
        ksp.getPC().setFieldSplitIS(*fields)
    ksp.setFromOptions()
    x = u.x.petsc_vec
    ksp.solve(b, x)
    if ksp.getConvergedReason() <= 0:
        raise RuntimeError(f"{prefix}: Krylov solver did not converge (reason {ksp.getConvergedReason()})")
    its = ksp.getIterationNumber()
    if opts is LU:
        # iterative refinement with the same factorisation: recovers digits lost to ill-conditioning
        # (convection-dominated heat equation), so discrete residuals reach round-off
        r, dx = b.duplicate(), b.duplicate()
        for _ in range(3):
            A.mult(x, r); r.aypx(-1.0, b)
            if r.norm() <= 1e-14 * max(b.norm(), 1e-300):
                break
            ksp.solve(r, dx); x.axpy(1.0, dx)
        r.destroy(); dx.destroy()
    u.x.scatter_forward()
    ksp.destroy(); A.destroy(); b.destroy()
    if P is not None:
        P.destroy()
    s = STATS.setdefault(prefix, {"seconds": 0.0, "its": 0})
    s["seconds"] += time.perf_counter() - t0
    s["its"] = its


def assemble_gradient(form, V0) -> np.ndarray:
    """Assemble a linear form on V0 and return the locally owned values."""
    vec = fpetsc.assemble_vector(fem.form(form))
    vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    n = V0.dofmap.index_map.size_local * V0.dofmap.index_map_bs
    out = vec.array_r[:n].copy()
    vec.destroy()
    return out


@dataclass
class Settings:
    length: float = 1.5
    nx: int = 90
    ny: int = 60
    mu: float = 1.0
    a_max: float = 2.5e4
    q: float = 0.1
    r_filter: float = 0.03
    beta: float = 1.0
    eta: float = 0.5
    # thermal
    thermal: bool = False
    pe: float = 1.0e3
    k_solid: float = 1.0
    k_fluid: float = 0.05
    p_in: float = 1.0e3
    pnorm: float = 8.0
    solver: str = "lu"                      # "lu" (MUMPS) or "iterative" (AMG / block-preconditioned Krylov)
    extra: dict = field(default_factory=dict)


class Problem:
    """Shared machinery: mesh, spaces, filter, projection, Brinkman interpolation."""

    def __init__(self, s: Settings, comm=MPI.COMM_WORLD):
        self.s, self.comm = s, comm
        self.msh = mesh.create_rectangle(comm, [np.array([0.0, 0.0]), np.array([s.length, 1.0])], [s.nx, s.ny],
                                         mesh.CellType.triangle, diagonal=mesh.DiagonalType.crossed)
        self.tdim = self.msh.topology.dim
        self.msh.topology.create_connectivity(self.tdim - 1, self.tdim)
        cell = self.msh.basix_cell()
        self.V0 = fem.functionspace(self.msh, ("DG", 0))
        self.V1 = fem.functionspace(self.msh, ("Lagrange", 1))
        P2 = basix.ufl.element("Lagrange", cell, 2, shape=(2,))
        P1 = basix.ufl.element("Lagrange", cell, 1)
        self.W = fem.functionspace(self.msh, basix.ufl.mixed_element([P2, P1]))
        self.x = fem.Function(self.V0, name="design")
        self.xf = fem.Function(self.V1, name="filtered")
        self.beta = fem.Constant(self.msh, PETSc.ScalarType(s.beta))
        self.qc = fem.Constant(self.msh, PETSc.ScalarType(s.q))      # Brinkman interpolation parameter (continued)
        self.amax = fem.Constant(self.msh, PETSc.ScalarType(s.a_max))  # solid inverse permeability (continued)
        self.w = fem.Function(self.W, name="flow")
        self.area = comm.allreduce(fem.assemble_scalar(fem.form(fem.Constant(self.msh, 1.0) * ufl.dx(domain=self.msh))), op=MPI.SUM)
        self.n_local = self.V0.dofmap.index_map.size_local

    # -------------------------------------------------------------- linear solves
    def solve(self, a, L, bcs, u, prefix, kind):
        """kind: "cg" (filter), "gmres" (heat) or "stokes" (flow); ignored with the default LU solver."""
        if self.s.solver == "lu":
            return solve_linear(a, L, bcs, u, prefix)
        if kind != "stokes":
            return solve_linear(a, L, bcs, u, prefix, ITERATIVE[kind])
        return solve_linear(a, L, bcs, u, prefix, ITERATIVE["stokes"], fields=self.flow_fields(),
                            nullspace=self.pressure_nullspace())

    def flow_fields(self):
        if not hasattr(self, "_fields"):
            im = self.W.dofmap.index_map
            self._fields = []
            for i, name in enumerate("up"):
                dofs = np.asarray(self.W.sub(i).collapse()[1], dtype=np.int64)
                owned = np.sort(dofs[dofs < im.size_local]) + im.local_range[0]
                self._fields.append((name, PETSc.IS().createGeneral(owned.astype(PETSc.IntType), comm=self.comm)))
        return self._fields

    def pressure_nullspace(self):
        return None

    # -------------------------------------------------------------- design pipeline
    def rho(self):
        b, e = self.beta, self.s.eta
        return (ufl.tanh(b * e) + ufl.tanh(b * (self.xf - e))) / (ufl.tanh(b * e) + ufl.tanh(b * (1 - e)))

    def alpha(self):
        r = self.rho()
        return self.amax * self.qc * (1 - r) / (self.qc + r)

    def filter_forms(self, rhs):
        u, v = ufl.TrialFunction(self.V1), ufl.TestFunction(self.V1)
        a = self.s.r_filter ** 2 * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + u * v * ufl.dx
        return a, rhs * v * ufl.dx

    def apply_filter(self):
        a, L = self.filter_forms(self.x)
        self.solve(a, L, [], self.xf, "filt_", "cg")

    def filter_adjoint(self, dJ_dxf_form) -> np.ndarray:
        """Chain rule through the (symmetric) Helmholtz filter: returns dJ/dx on owned DG0 dofs."""
        a, _ = self.filter_forms(self.xf)
        lam = fem.Function(self.V1)
        self.solve(a, dJ_dxf_form, [], lam, "filtadj_", "cg")
        return assemble_gradient(lam * ufl.TestFunction(self.V0) * ufl.dx, self.V0)

    def volume(self):
        return self.comm.allreduce(fem.assemble_scalar(fem.form(self.rho() * ufl.dx)), op=MPI.SUM) / self.area

    def volume_gradient(self):
        dxf = ufl.derivative(self.rho() * ufl.dx, self.xf, ufl.TestFunction(self.V1))
        return self.filter_adjoint(dxf) / self.area

    def set_design(self, local_values: np.ndarray):
        self.x.x.array[: self.n_local] = local_values
        self.x.x.scatter_forward()
        self.apply_filter()

    # -------------------------------------------------------------- boundaries
    def facets(self, marker):
        return mesh.locate_entities_boundary(self.msh, self.tdim - 1, marker)


class DoublePipe(Problem):
    """Borrvall & Petersson (2003) double pipe: minimise dissipated power with a fluid-volume limit.

    Two parabolic inflows on the left (y in [1/6,1/3] and [2/3,5/6]) and matching outflows on the
    right; no-slip elsewhere. The flow problem is self-adjoint for this objective, which the tests use.
    """

    def __init__(self, s: Settings, comm=MPI.COMM_WORLD):
        super().__init__(s, comm)
        W0, _ = self.W.sub(0).collapse()
        g = fem.Function(W0)
        L, l = s.length, 1.0 / 6.0

        def profile(xx):
            vals = np.zeros((2, xx.shape[1]))
            for c in (0.25, 0.75):
                t = (xx[1] - c) / (l / 2)
                inside = np.abs(t) < 1
                vals[0] += np.where(inside, 1 - t**2, 0.0)
            return vals

        g.interpolate(profile)
        wall = self.facets(lambda xx: np.full(xx.shape[1], True))
        dofs = fem.locate_dofs_topological((self.W.sub(0), W0), self.tdim - 1, wall)
        self.bcs = [fem.dirichletbc(g, dofs, self.W.sub(0))]
        self.u_bc = g

    def pressure_nullspace(self):
        """Velocity is prescribed on the whole boundary, so the pressure is fixed only up to a constant."""
        if not hasattr(self, "_ns"):
            _, pdofs = self.W.sub(1).collapse()
            z = fem.Function(self.W)
            z.x.array[pdofs] = 1.0
            vec = z.x.petsc_vec.copy()
            vec.normalize()
            self._ns = PETSc.NullSpace().create(vectors=[vec], comm=self.comm)
        return self._ns

    def flow_forms(self, w, vq):
        u, p = ufl.split(w)
        v, q = ufl.split(vq)
        mu = self.s.mu
        return (self.alpha() * ufl.inner(u, v) + mu * ufl.inner(ufl.grad(u), ufl.grad(v))
                - p * ufl.div(v) - q * ufl.div(u)) * ufl.dx

    def solve_flow(self):
        (u, p), (v, q) = ufl.TrialFunctions(self.W), ufl.TestFunctions(self.W)
        mu = self.s.mu
        a = (self.alpha() * ufl.inner(u, v) + mu * ufl.inner(ufl.grad(u), ufl.grad(v))
             - p * ufl.div(v) - q * ufl.div(u)) * ufl.dx
        zero = fem.Constant(self.msh, PETSc.ScalarType((0.0, 0.0)))
        self.solve(a, ufl.inner(zero, v) * ufl.dx, self.bcs, self.w, "flow_", "stokes")
        # pressure is defined up to a constant (velocity BCs everywhere) -- MUMPS handles the null space;
        # fix the mean to zero for clean output
        _, pdofs = self.W.sub(1).collapse()
        pm = self.comm.allreduce(fem.assemble_scalar(fem.form(self.w.sub(1) * ufl.dx)), op=MPI.SUM) / self.area
        self.w.x.array[pdofs] -= pm

    def objective_form(self, w):
        u, _ = ufl.split(w)
        return 0.5 * (self.alpha() * ufl.inner(u, u) + self.s.mu * ufl.inner(ufl.grad(u), ufl.grad(u))) * ufl.dx

    def evaluate(self, gradient: bool = True):
        self.solve_flow()
        J_form = self.objective_form(self.w)
        J = self.comm.allreduce(fem.assemble_scalar(fem.form(J_form)), op=MPI.SUM)
        if not gradient:
            return J, None
        # adjoint:  (dR/dw)^T lam = -(dJ/dw)   with homogeneous Dirichlet BCs
        vq, dw = ufl.TestFunction(self.W), ufl.TrialFunction(self.W)
        R = self.flow_forms(self.w, vq)
        a_adj = ufl.adjoint(ufl.derivative(R, self.w, dw))
        L_adj = -ufl.derivative(J_form, self.w, vq)
        lam = fem.Function(self.W)
        self.solve(a_adj, L_adj, self._homogeneous_bcs(), lam, "adj_", "stokes")
        # total derivative w.r.t. the filtered field:  dJ/dxf + lam^T dR/dxf
        dxf = ufl.TestFunction(self.V1)
        dJ = ufl.derivative(J_form, self.xf, dxf) + ufl.derivative(ufl.replace(R, {vq: lam}), self.xf, dxf)
        return J, self.filter_adjoint(dJ)

    def _homogeneous_bcs(self):
        W0, _ = self.W.sub(0).collapse()
        z = fem.Function(W0)
        wall = self.facets(lambda xx: np.full(xx.shape[1], True))
        dofs = fem.locate_dofs_topological((self.W.sub(0), W0), self.tdim - 1, wall)
        return [fem.dirichletbc(z, dofs, self.W.sub(0))]


class ThermalPlate(Problem):
    """Pressure-driven Brinkman flow + SUPG-stabilised convection-diffusion in a cold plate.

    Coolant enters on the left at pressure p_in and leaves on the right at p = 0 (traction BCs), with
    no-slip on top and bottom. The chip injects heat Q everywhere and a hotspot injects more; the inlet
    is held at T = 0. Unlike a Darcy model, the -mu lap(u) term charges viscous wall friction, so very
    narrow channels are penalised physically rather than by a length-scale rule alone.

    Objective: p-norm of temperature (smooth hotspot surrogate), subject to a fluid-volume limit.
    """

    def __init__(self, s: Settings, comm=MPI.COMM_WORLD, hotspot=(0.55, 0.35, 0.12, 0.2, 8.0)):
        super().__init__(s, comm)
        L = s.length
        fdim = self.tdim - 1
        left = self.facets(lambda xx: np.isclose(xx[0], 0.0))
        right = self.facets(lambda xx: np.isclose(xx[0], L))
        walls = self.facets(lambda xx: np.isclose(xx[1], 0.0) | np.isclose(xx[1], 1.0))
        idx = np.concatenate([left, right]); vals = np.concatenate([np.full(len(left), 1), np.full(len(right), 2)])
        order = np.argsort(idx)
        self.ft = mesh.meshtags(self.msh, fdim, idx[order], vals[order].astype(np.int32))
        self.ds = ufl.Measure("ds", domain=self.msh, subdomain_data=self.ft)
        W0, _ = self.W.sub(0).collapse()
        z = fem.Function(W0)
        wdofs = fem.locate_dofs_topological((self.W.sub(0), W0), fdim, walls)
        self.bcs = [fem.dirichletbc(z, wdofs, self.W.sub(0))]
        self.wall_dofs, self.W0 = wdofs, W0
        self.T = fem.Function(self.V1, name="temperature")
        tdofs = fem.locate_dofs_topological(self.V1, fdim, left)
        self.bcs_T = [fem.dirichletbc(PETSc.ScalarType(0.0), tdofs, self.V1)]
        self.inlet_T_dofs = tdofs
        cx, cy, hx, hy, mult = hotspot
        X = ufl.SpatialCoordinate(self.msh)
        smooth = lambda d, h: 0.5 * (1 - ufl.tanh((abs(d) - h) / 0.01))  # noqa: E731
        self.Q = 1.0 + (mult - 1.0) * smooth(X[0] - cx * L, hx * L) * smooth(X[1] - cy, hy)
        self.Q_total = self.comm.allreduce(fem.assemble_scalar(fem.form(self.Q * ufl.dx)), op=MPI.SUM)
        self.hotspot = hotspot

    def k(self):
        r = self.rho()
        s = 1 - r
        return self.s.k_fluid + (self.s.k_solid - self.s.k_fluid) * s / (1 + 8.0 * (1 - s))

    def flow_residual(self, w, vq):
        u, p = ufl.split(w); v, q = ufl.split(vq)
        n = ufl.FacetNormal(self.msh)
        return ((self.alpha() * ufl.inner(u, v) + self.s.mu * ufl.inner(ufl.grad(u), ufl.grad(v))
                 - p * ufl.div(v) - q * ufl.div(u)) * dxq + self.s.p_in * ufl.inner(n, v) * self.ds(1))

    def heat_residual(self, w, T, s_):
        u, _ = ufl.split(w)
        h = ufl.CellDiameter(self.msh)
        unorm = ufl.sqrt(ufl.inner(u, u) + 1e-12)
        pe, k = self.s.pe, self.k()
        tau = 1.0 / ufl.sqrt((2 * pe * unorm / h) ** 2 + (12 * k / h**2) ** 2)
        res = pe * ufl.inner(u, ufl.grad(T)) - self.Q          # strong residual (P1: no 2nd derivatives)
        return ((pe * ufl.inner(u, ufl.grad(T)) * s_ + k * ufl.inner(ufl.grad(T), ufl.grad(s_)) - self.Q * s_) * dxq
                + tau * res * pe * ufl.inner(u, ufl.grad(s_)) * dxq)

    def objective_form(self, T):
        return (ufl.max_value(T, 0.0) + 1e-12) ** self.s.pnorm * ufl.dx

    def evaluate(self, gradient: bool = True):
        vq, dw = ufl.TestFunction(self.W), ufl.TrialFunction(self.W)
        RF = self.flow_residual(self.w, vq)
        self.solve(ufl.lhs(ufl.replace(RF, {self.w: dw})), ufl.rhs(ufl.replace(RF, {self.w: dw})),
                   self.bcs, self.w, "flow_", "stokes")
        sT, dT = ufl.TestFunction(self.V1), ufl.TrialFunction(self.V1)
        RT = self.heat_residual(self.w, self.T, sT)
        RT_lin = ufl.replace(RT, {self.T: dT})
        self.solve(ufl.lhs(RT_lin), ufl.rhs(RT_lin), self.bcs_T, self.T, "heat_", "gmres")
        Jp = self.comm.allreduce(fem.assemble_scalar(fem.form(self.objective_form(self.T))), op=MPI.SUM)
        J = (Jp / self.area) ** (1.0 / self.s.pnorm)
        if not gradient:
            return J, None
        dJ_dJp = J / (self.s.pnorm * Jp)
        Jform = dJ_dJp * self.objective_form(self.T)                         # linearisation of the p-norm
        # heat adjoint
        lamT = fem.Function(self.V1)
        a_T = ufl.adjoint(ufl.derivative(RT, self.T, dT))
        self.solve(a_T, -ufl.derivative(Jform, self.T, sT), self.bcs_T, lamT, "adjT_", "gmres")
        # flow adjoint: forced by the heat equation's dependence on velocity
        lamF = fem.Function(self.W)
        a_F = ufl.adjoint(ufl.derivative(RF, self.w, dw))
        L_F = -ufl.derivative(ufl.replace(RT, {sT: lamT}), self.w, vq)
        self.solve(a_F, L_F, self.bcs, lamF, "adjF_", "stokes")
        dxf = ufl.TestFunction(self.V1)
        dJ = (ufl.derivative(Jform, self.xf, dxf) + ufl.derivative(ufl.replace(RT, {sT: lamT}), self.xf, dxf)
              + ufl.derivative(ufl.replace(RF, {vq: lamF}), self.xf, dxf))
        return J, self.filter_adjoint(dJ)

    def diagnostics(self):
        u, _ = ufl.split(self.w)
        n = ufl.FacetNormal(self.msh)
        flow = self.comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(u, n) * self.ds(2))), op=MPI.SUM)
        adv = self.comm.allreduce(fem.assemble_scalar(fem.form(self.s.pe * self.T * ufl.inner(u, n) * self.ds(2))), op=MPI.SUM)
        Tmax = self.comm.allreduce(float(self.T.x.array.max()), op=MPI.MAX)
        # coolant leaking through solid: x-flow weighted by solid indicator, averaged over the plate length
        leak = self.comm.allreduce(fem.assemble_scalar(fem.form((1 - self.rho()) * u[0] * dxq)), op=MPI.SUM) / self.s.length
        return {"T_max": Tmax, "flow_rate": flow, "pumping_power": flow * self.s.p_in,
                "flow_through_solid_fraction": leak / flow if flow > 0 else float("nan"),
                "heat_in": self.Q_total, "heat_out_advected": adv}


def _sum(prob, form):
    return prob.comm.allreduce(fem.assemble_scalar(fem.form(form)), op=MPI.SUM)


def conservation(prob) -> dict:
    """Global and local mass, momentum and energy balances for the current ThermalPlate solution.

    Global checks are independent integral statements of each conservation law; local checks are
    cell-wise mass imbalance and the nodal residuals of the discrete momentum and heat equations
    (the finite-element statement of local conservation at every degree of freedom).
    """
    u, p = ufl.split(prob.w)
    n = ufl.FacetNormal(prob.msh)
    ds, s = prob.ds, prob.s
    walls = ufl.Measure("ds", domain=prob.msh)  # whole boundary; walls = total - inlet - outlet
    out = {}

    # ---------------- mass
    q_in = -_sum(prob, ufl.inner(u, n) * ds(1))
    q_out = _sum(prob, ufl.inner(u, n) * ds(2))
    q_net = _sum(prob, ufl.inner(u, n) * walls)
    div_cells = assemble_gradient(ufl.div(u) * ufl.TestFunction(prob.V0) * ufl.dx, prob.V0)
    max_cell = prob.comm.allreduce(float(np.abs(div_cells).max()), op=MPI.MAX)
    div_l2 = np.sqrt(_sum(prob, ufl.div(u) ** 2 * ufl.dx))
    grad_l2 = np.sqrt(_sum(prob, ufl.inner(ufl.grad(u), ufl.grad(u)) * ufl.dx))
    out["mass"] = {"inflow": q_in, "outflow": q_out, "global_imbalance_rel": q_net / q_in,
                   "max_cell_imbalance_rel_to_throughflow": max_cell / q_in,
                   "div_u_L2_over_grad_u_L2": div_l2 / grad_l2}

    # ---------------- momentum (x-direction): inlet pressure force = Brinkman drag + wall friction
    push = s.p_in * 1.0                                            # p_in times inlet height
    drag = _sum(prob, prob.alpha() * u[0] * dxq)
    wall_shear = -(_sum(prob, s.mu * ufl.inner(ufl.grad(u[0]), n) * walls)
                   - _sum(prob, s.mu * ufl.inner(ufl.grad(u[0]), n) * ds(1))
                   - _sum(prob, s.mu * ufl.inner(ufl.grad(u[0]), n) * ds(2)))
    out["momentum_x"] = {"inlet_pressure_force": push, "brinkman_drag": drag, "wall_friction": wall_shear,
                         "closure_rel": (drag + wall_shear - push) / push}

    # ---------------- mechanical energy: pumping power = dissipation (exact for this weak form)
    diss = _sum(prob, (prob.alpha() * ufl.inner(u, u) + s.mu * ufl.inner(ufl.grad(u), ufl.grad(u))) * dxq)
    out["mechanical_energy"] = {"pumping_power": s.p_in * q_in, "dissipation": diss,
                                "closure_rel": (diss - s.p_in * q_in) / (s.p_in * q_in)}

    # ---------------- thermal energy with consistent (residual-based) inlet conduction flux
    sT = ufl.TestFunction(prob.V1)
    RT = fpetsc.assemble_vector(fem.form(prob.heat_residual(prob.w, prob.T, sT)))
    RT.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    nloc = prob.V1.dofmap.index_map.size_local
    arr = RT.array_r[:nloc]
    inlet = prob.inlet_T_dofs[prob.inlet_T_dofs < nloc]
    free = np.setdiff1d(np.arange(nloc), inlet)
    cond_in = -prob.comm.allreduce(float(arr[inlet].sum()), op=MPI.SUM)          # heat conducted out at inlet
    res_free = prob.comm.allreduce(float(np.abs(arr[free]).max()) if len(free) else 0.0, op=MPI.MAX)
    RT.destroy()
    adv_out = _sum(prob, s.pe * prob.T * ufl.inner(u, n) * ds(2))
    div_src = _sum(prob, s.pe * prob.T * ufl.div(u) * dxq)
    q_heat = prob.Q_total
    out["thermal_energy"] = {"heat_in": q_heat, "advected_out": adv_out, "conducted_into_inlet": cond_in,
                             "spurious_from_div_u": div_src,
                             "closure_rel": (adv_out + cond_in - div_src - q_heat) / q_heat,
                             "closure_rel_ignoring_div_u": (adv_out + cond_in - q_heat) / q_heat}

    # ---------------- local residuals of the discrete equations (free dofs)
    vq = ufl.TestFunction(prob.W)
    RF = fpetsc.assemble_vector(fem.form(prob.flow_residual(prob.w, vq)))
    RF.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    nW = prob.W.dofmap.index_map.size_local * prob.W.dofmap.index_map_bs
    rf = RF.array_r[:nW].copy(); RF.destroy()
    rf[prob.wall_dofs[0][prob.wall_dofs[0] < nW]] = 0.0
    rf_max = prob.comm.allreduce(float(np.abs(rf).max()), op=MPI.MAX)
    out["local_residuals"] = {"momentum_mass_nodal_max_rel": rf_max / push,
                              "heat_nodal_max_rel": res_free / (q_heat / prob.comm.allreduce(nloc))}
    return out
