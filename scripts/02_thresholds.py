"""Phase 2 -- memory thresholds, to check the noise model against the literature.

If the rotated surface code under this circuit-level model does not threshold
somewhere near ~0.5-1% in p_2q, the noise model is wrong and every number
downstream is wrong with it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.circuits import memory  # noqa: E402
from src.config import shots as _shots  # noqa: E402
from src.noise import NoiseModel  # noqa: E402
from src.sampling import SweepPoint, run_sweep  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA, FIGS = ROOT / "data", ROOT / "figs"
DATA.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)

DISTANCES = (3, 5, 7)
PS = np.array([1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 8e-3, 1e-2, 1.3e-2])
SHOTS = _shots(400_000)


def crossing(rows: list[dict]) -> float:
    """Where the d=3 and d=7 curves cross -- a serviceable threshold estimate."""
    def curve(d):
        r = sorted([x for x in rows if x["d"] == d], key=lambda x: x["p_2q"])
        return np.array([x["p_2q"] for x in r]), np.array([x["ler"] for x in r])

    p3, l3 = curve(3)
    p7, l7 = curve(7)
    diff = np.log(np.maximum(l3, 1e-12)) - np.log(np.maximum(l7, 1e-12))
    sign = np.sign(diff)
    idx = np.flatnonzero(np.diff(sign) != 0)
    if not len(idx):
        return float("nan")
    i = idx[0]
    x0, x1, y0, y1 = p3[i], p3[i + 1], diff[i], diff[i + 1]
    return float(x0 - y0 * (x1 - x0) / (y1 - y0))


if __name__ == "__main__":
    points = [
        SweepPoint(memory(d, "Z"), d, NoiseModel.uniform(p, loss_ratio=0.0), "pauli", SHOTS, 1000 + i)
        for d in DISTANCES
        for i, p in enumerate(PS)
    ]
    print(f"running {len(points)} sweep points x {SHOTS:,} shots ...")
    rows = run_sweep(points)
    p_th = crossing(rows)

    print(f"\n{'d':>2} {'p_2q':>8} {'LER':>10} {'±':>9}")
    for r in sorted(rows, key=lambda x: (x["d"], x["p_2q"])):
        print(f"{r['d']:>2} {r['p_2q']:>8.1e} {r['ler']:>10.3e} {r['ler_err']:>9.1e}")
    print(f"\nthreshold (d=3 / d=7 crossing): p_th ~ {p_th:.2%}")
    print("Literature for circuit-level depolarising noise on the rotated surface code")
    print("sits around 0.5-1%. If we are in that band, the noise model is sane.")

    (DATA / "02_thresholds.json").write_text(
        json.dumps({"rows": rows, "threshold": p_th}, indent=2)
    )

    fig, ax = plt.subplots(figsize=(6, 4.2))
    for d, c in zip(DISTANCES, ["#4C72B0", "#DD8452", "#55A868"]):
        r = sorted([x for x in rows if x["d"] == d], key=lambda x: x["p_2q"])
        x = np.array([v["p_2q"] for v in r])
        y = np.array([v["ler"] for v in r])
        e = np.array([v["ler_err"] for v in r])
        ax.errorbar(x, y, yerr=e, marker="o", ms=4, lw=1.4, capsize=2, color=c, label=f"d = {d}")
    if np.isfinite(p_th):
        ax.axvline(p_th, ls="--", lw=1, color="0.4")
        ax.text(p_th * 1.03, 3e-4, f"$p_{{th}} \\approx {100 * p_th:.2f}\\%$",
                fontsize=9, color="0.3")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("two-qubit gate error $p_{2q}$")
    ax.set_ylabel("logical error per shot")
    ax.set_title("Memory threshold (MWPM), $d$ rounds", fontsize=11)
    ax.legend(frameon=False)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(FIGS / "00_memory_threshold.png", dpi=170)
    print(f"wrote {FIGS / '00_memory_threshold.png'}")
