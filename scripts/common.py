"""Helpers: gather a DG0 field on rank 0 for plotting, and draw it."""
from __future__ import annotations

import numpy as np
from dolfinx import mesh as dmesh


def gather_cells(prob, values_local):
    """Return (triangles (N,3,2), values (N,)) on rank 0, None elsewhere."""
    msh = prob.msh
    n = prob.n_local
    conn = msh.geometry.dofmap[:n]
    tris = msh.geometry.x[conn][:, :, :2]
    all_t = prob.comm.gather(tris, root=0)
    all_v = prob.comm.gather(np.asarray(values_local[:n]), root=0)
    if prob.comm.rank == 0:
        return np.concatenate(all_t), np.concatenate(all_v)
    return None


def cell_midpoints(prob):
    n = prob.n_local
    return dmesh.compute_midpoints(prob.msh, prob.tdim, np.arange(n, dtype=np.int32))[:, :2]


def draw(ax, tris, vals, cmap="Blues", vmin=0, vmax=1, title=""):
    from matplotlib.collections import PolyCollection
    pc = PolyCollection(tris, array=vals, cmap=cmap, edgecolors="face", linewidths=0.05)
    pc.set_clim(vmin, vmax)
    ax.add_collection(pc)
    ax.set_xlim(tris[:, :, 0].min(), tris[:, :, 0].max()); ax.set_ylim(tris[:, :, 1].min(), tris[:, :, 1].max())
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)
    return pc


def projected_dg0(prob):
    """Physical density rho (projection of the filtered field) evaluated at cell midpoints, local owned cells."""
    import ufl
    from dolfinx import fem
    expr = fem.Expression(prob.rho(), prob.V0.element.interpolation_points)
    f = fem.Function(prob.V0)
    f.interpolate(expr)
    return f.x.array[: prob.n_local].copy()
