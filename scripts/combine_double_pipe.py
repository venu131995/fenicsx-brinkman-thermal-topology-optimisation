"""Merge the per-length double-pipe runs (each run in its own MPI job) into results/double_pipe.json and
figures/fig1_double_pipe.png."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts"))
from common import draw  # noqa: E402

out = {"mpi_ranks": 4}
fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), gridspec_kw={"width_ratios": [1, 1.5]})
for ax, L in zip(axs, (1.0, 1.5)):
    r = json.loads((ROOT / "results" / f"double_pipe_L{L}.json").read_text())
    d = np.load(ROOT / "results" / f"double_pipe_L{L}.npz")
    out[f"L={L}"] = r
    draw(ax, d["tris"], d["vals"], title=f"domain length {L}:  power {r['J_optimised']:.2f}\n"
                                          f"(two straight pipes: {r['J_two_straight_pipes']:.2f})")
fig.suptitle("Borrvall-Petersson double pipe (FEniCSx, Taylor-Hood P2/P1, MPI)", fontsize=10)
fig.tight_layout(); fig.savefig(ROOT / "figures" / "fig1_double_pipe.png", dpi=140)
(ROOT / "results" / "double_pipe.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "history"} if isinstance(v, dict) else v for k, v in out.items()}, indent=1))
