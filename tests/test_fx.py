import sys
from pathlib import Path

import numpy as np
import pytest
from mpi4py import MPI

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fxtopopt import MMA, DoublePipe, Settings, ThermalPlate  # noqa: E402


def taylor_test(prob, seed=0, eps=(1e-2, 5e-3, 2.5e-3)):
    """Second-order Taylor remainder |J(x+e h) - J(x) - e dJ.h| must shrink like e^2."""
    rng = np.random.default_rng(seed + prob.comm.rank)
    x = 0.3 + 0.4 * rng.random(prob.n_local)
    h = rng.standard_normal(prob.n_local)
    prob.set_design(x)
    J, dJ = prob.evaluate()
    dJh = prob.comm.allreduce(float(dJ @ h), op=MPI.SUM)
    rem = []
    for e in eps:
        prob.set_design(x + e * h)
        rem.append(abs(prob.evaluate(gradient=False)[0] - J - e * dJh))
    rates = [np.log2(rem[i] / rem[i + 1]) for i in range(len(rem) - 1)]
    return rates, rem


def test_double_pipe_gradient_taylor():
    prob = DoublePipe(Settings(nx=24, ny=16, beta=2.0))
    rates, _ = taylor_test(prob)
    assert min(rates) > 1.8


def test_thermal_gradient_taylor():
    prob = ThermalPlate(Settings(length=2.0, nx=30, ny=15, beta=2.0, thermal=True, pe=200.0, p_in=200.0))
    rates, _ = taylor_test(prob, seed=1)
    assert min(rates) > 1.8


def test_brinkman_self_adjoint_energy_identity():
    # For the dissipation objective the adjoint equals -w, so dJ/drho = -0.5 dalpha/drho |u|^2 <= 0 everywhere:
    # adding fluid anywhere can only lower dissipated power.
    prob = DoublePipe(Settings(nx=24, ny=16))
    prob.set_design(np.full(prob.n_local, 0.5))
    _, dJ = prob.evaluate()
    assert np.all(dJ <= 1e-10)


def test_thermal_energy_balance_and_pressure_drives_flow():
    prob = ThermalPlate(Settings(length=2.0, nx=40, ny=20, thermal=True, pe=200.0, p_in=200.0))
    prob.set_design(np.full(prob.n_local, 0.6))
    prob.evaluate(gradient=False)
    d = prob.diagnostics()
    assert d["flow_rate"] > 0
    # heat leaves by advection at the outlet and by conduction into the T=0 inlet; advection should dominate
    assert 0.5 * d["heat_in"] < d["heat_out_advected"] < 1.05 * d["heat_in"]


def test_parallel_mma_kkt():
    comm = MPI.COMM_WORLD
    rng = np.random.default_rng(3 + comm.rank)
    n = 20
    c, t = rng.uniform(0.5, 2, n), rng.uniform(0, 1, n)
    N = comm.allreduce(n)
    a, b = np.full(n, 1 / N), 0.3
    x, mma = np.full(n, 0.3), MMA(comm)
    for _ in range(150):
        x = mma.step(x, 2 * c * (x - t), a, b)
    lo, hi = 0.0, 1e4
    for _ in range(100):
        lam = 0.5 * (lo + hi)
        xs = np.clip(t - lam / (2 * c * N), 0, 1)
        lo, hi = (lam, hi) if comm.allreduce(xs.sum()) / N > b else (lo, lam)
    assert np.allclose(x, xs, atol=2e-3)


def test_solid_is_effectively_impermeable_at_working_settings():
    # Guard against the "leaky solid" exploit: with the production a_max, an all-solid plate must carry
    # only a few percent of the flow of an open-channel plate at the same pressure drop.
    s = Settings(length=2.0, nx=40, ny=20, thermal=True, pe=1e3, p_in=3e3, beta=64.0, a_max=1e6)
    prob = ThermalPlate(s)
    prob.set_design(np.zeros(prob.n_local)); prob.evaluate(gradient=False)
    solid_flow = prob.diagnostics()["flow_rate"]
    prob.set_design(np.full(prob.n_local, 1.0)); prob.evaluate(gradient=False)
    open_flow = prob.diagnostics()["flow_rate"]
    assert solid_flow < 0.01 * open_flow


def test_conservation_report_closes():
    from fxtopopt import conservation
    prob = ThermalPlate(Settings(length=2.0, nx=40, ny=20, thermal=True, pe=1e3, p_in=3e3, beta=8.0, a_max=1e6))
    prob.set_design(np.random.default_rng(prob.comm.rank).uniform(0.2, 0.8, prob.n_local))
    prob.evaluate(gradient=False)
    c = conservation(prob)
    assert abs(c["mass"]["global_imbalance_rel"]) < 1e-10
    assert abs(c["mechanical_energy"]["closure_rel"]) < 1e-9
    assert abs(c["thermal_energy"]["closure_rel"]) < 1e-9
    assert c["local_residuals"]["momentum_mass_nodal_max_rel"] < 1e-9
    assert c["local_residuals"]["heat_nodal_max_rel"] < 1e-8


@pytest.mark.parametrize("cls,kw", [
    (DoublePipe, dict(nx=24, ny=16, beta=2.0)),
    (ThermalPlate, dict(length=2.0, nx=40, ny=20, thermal=True, pe=1e3, p_in=3e3, beta=8.0, a_max=1e6)),
])
def test_iterative_solvers_match_direct(cls, kw):
    # AMG / block-preconditioned MINRES must reproduce the MUMPS objective and gradient
    rng = np.random.default_rng(5 + MPI.COMM_WORLD.rank)
    out = []
    for solver in ("lu", "iterative"):
        prob = cls(Settings(solver=solver, **kw))
        if not out:
            x = rng.uniform(0.2, 0.8, prob.n_local)
        prob.set_design(x)
        out.append(prob.evaluate())
    (J0, g0), (J1, g1) = out
    comm = MPI.COMM_WORLD
    gerr = np.sqrt(comm.allreduce(float((g1 - g0) @ (g1 - g0)))) / np.sqrt(comm.allreduce(float(g0 @ g0)))
    assert abs(J1 - J0) / abs(J0) < 1e-8
    assert gerr < 1e-6
