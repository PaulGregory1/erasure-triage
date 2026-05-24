"""Phase 3 -- heralded atom loss and dynamic reweighting. Figure 1.

Three decoders, same shots, same graph topology:

  (a) blind    -- the decoder's DEM is built with p_loss = 0. It does not know
                  atom loss exists. Erasures show up as unexplained syndrome.
  (b) pauli    -- the DEM includes the erasure channel, but the herald bit is
                  thrown away. An erased atom is just a noisier atom.
  (c) reweight -- the herald is read per shot, and every matching edge the erased
                  atom can flip is set to weight 0 (probability 1/2).

(c) - (a) is the value of the herald. (c) - (b) is the value of using it *per shot*
rather than on average. Both are reported, because they answer different questions.
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

from src.circuits import memory, transversal_cnot  # noqa: E402
from src.config import shots as _shots  # noqa: E402
from src.noise import NoiseModel  # noqa: E402
from src.sampling import SweepPoint, run_sweep  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA, FIGS = ROOT / "data", ROOT / "figs"
DATA.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)

DISTANCES = (3, 5, 7)
PS = np.array([1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 8e-3, 1e-2, 1.3e-2])
LOSS_RATIOS = (0.0, 0.5, 1.0, 2.0)  # p_loss / p_2q
VARIANTS = ("blind", "pauli", "reweight")
SHOTS = _shots(300_000)

COLORS = {"blind": "#C44E52", "pauli": "#DD8452", "reweight": "#55A868"}
LABELS = {
    "blind": "(a) erasure ignored",
    "pauli": "(b) erasure as Pauli noise",
    "reweight": "(c) dynamic reweighting",
}


def threshold_of(rows: list[dict]) -> float:
    def curve(d):
        r = sorted([x for x in rows if x["d"] == d], key=lambda x: x["p_2q"])
        return np.array([x["p_2q"] for x in r]), np.array([x["ler"] for x in r])

    p3, l3 = curve(3)
    _, l7 = curve(7)
    diff = np.log(np.maximum(l3, 1e-12)) - np.log(np.maximum(l7, 1e-12))
    idx = np.flatnonzero(np.diff(np.sign(diff)) != 0)
    if not len(idx):
        return float("nan")
    i = idx[0]
    return float(p3[i] - diff[i] * (p3[i + 1] - p3[i]) / (diff[i + 1] - diff[i]))


if __name__ == "__main__":
    points = []
    seed = 0
    for ratio in LOSS_RATIOS:
        for variant in VARIANTS:
            for d in DISTANCES:
                for p in PS:
                    seed += 1
                    points.append(
                        SweepPoint(
                            memory(d, "Z"), d,
                            NoiseModel.uniform(p, loss_ratio=ratio),
                            variant, SHOTS, seed,
                        )
                    )
    print(f"running {len(points)} points x {SHOTS:,} shots ...")
    rows = run_sweep(points)
    for r, pt in zip(rows, points):
        r["loss_ratio"] = pt.noise.p_loss / pt.noise.p_2q if pt.noise.p_2q else 0.0

    thresholds: dict[str, list[float]] = {v: [] for v in VARIANTS}
    for ratio in LOSS_RATIOS:
        for v in VARIANTS:
            sel = [r for r in rows if abs(r["loss_ratio"] - ratio) < 1e-9 and r["variant"] == v]
            thresholds[v].append(threshold_of(sel))

    print(f"\n{'p_loss/p_2q':>12} " + "".join(f"{LABELS[v]:>28}" for v in VARIANTS))
    for i, ratio in enumerate(LOSS_RATIOS):
        cells = "".join(f"{thresholds[v][i]:>27.2%} " for v in VARIANTS)
        print(f"{ratio:>12.1f} {cells}")

    gain = [
        thresholds["reweight"][i] / thresholds["blind"][i] if thresholds["blind"][i] else np.nan
        for i in range(len(LOSS_RATIOS))
    ]
    print("\nthreshold gain (c)/(a):", " ".join(f"{g:.2f}x" for g in gain))

    (DATA / "03_erasure.json").write_text(
        json.dumps(
            {"rows": rows, "loss_ratios": list(LOSS_RATIOS), "thresholds": thresholds},
            indent=2,
        )
    )

    # ---- Figure 1 -------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))

    ax = axes[0]
    for v in VARIANTS:
        ax.plot(LOSS_RATIOS, [t * 100 for t in thresholds[v]], marker="o", ms=5,
                lw=1.8, color=COLORS[v], label=LABELS[v])
    ax.set_xlabel("atom-loss rate / gate-error rate  ($p_{loss} / p_{2q}$)")
    ax.set_ylabel("threshold $p_{th}$  (%)")
    ax.set_title("Threshold vs atom-loss rate", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, lw=0.5)

    ax = axes[1]
    ratio = 1.0
    for v in VARIANTS:
        for d, ls in zip(DISTANCES, ["-", "--", ":"]):
            r = sorted(
                [x for x in rows if abs(x["loss_ratio"] - ratio) < 1e-9
                 and x["variant"] == v and x["d"] == d],
                key=lambda x: x["p_2q"],
            )
            x = [v2["p_2q"] for v2 in r]
            y = [v2["ler"] for v2 in r]
            e = [v2["ler_err"] for v2 in r]
            ax.errorbar(x, y, yerr=e, ls=ls, marker="o", ms=3, lw=1.3, capsize=1.5,
                        color=COLORS[v],
                        label=LABELS[v] if d == DISTANCES[0] else None)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("two-qubit gate error $p_{2q}$")
    ax.set_ylabel("logical error per shot")
    ax.set_title("Memory, $p_{loss} = p_{2q}$  (solid/dashed/dotted = $d$=3/5/7)", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, lw=0.5)

    fig.tight_layout()
    fig.savefig(FIGS / "01_erasure_threshold.png", dpi=170)
    print(f"wrote {FIGS / '01_erasure_threshold.png'}")
