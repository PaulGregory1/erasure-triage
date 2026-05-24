"""Phase 8 -- the figures. Four of them. Not five.

  fig 1  threshold vs atom-loss rate, with and without dynamic reweighting  (from 03)
  fig 2  the Pareto frontier: logical error per transversal gate vs latency (mean + p99)
  fig 3  feature importance -- does the erasure information actually carry the gain?
  fig 4  generalisation: train at d = 3,5, evaluate at d = 7 without retraining
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate import per_gate_error  # noqa: E402
from src.features import ALL_FEATURES, EDGE, ERASURE, GAP, MATCH  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA, FIGS = ROOT / "data", ROOT / "figs"
FIGS.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 110,
})

STYLE = {
    "oracle":            dict(color="#111111", ls="--", lw=1.5, marker=None, zorder=5),
    "LightGBM":          dict(color="#4C72B0", ls="-", lw=2.3, marker="o", ms=3.2, zorder=4),
    "LightGBM (no gap)": dict(color="#8FA8CD", ls="-", lw=1.5, marker="s", ms=2.8, zorder=3),
    "gap only":          dict(color="#DD8452", ls="-", lw=2.1, marker="^", ms=3.2, zorder=4),
    "matching weight":   dict(color="#55A868", ls="-.", lw=1.4, marker=None, zorder=2),
    "n detectors":       dict(color="#937860", ls=":", lw=1.4, marker=None, zorder=2),
    "random":            dict(color="#C44E52", ls=":", lw=1.7, marker=None, zorder=2),
    "LightGBM (no erasure)": dict(color="#55A868", ls="--", lw=1.6, marker="D", ms=2.8, zorder=3),
}
ERASURE_FEATURES = set(ERASURE)


def load(name: str) -> dict:
    return json.loads((DATA / name).read_text())


# --------------------------------------------------------------------------------------
def fig2_pareto(pareto: dict) -> None:
    """The figure. Logical error per transversal gate vs decode latency."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    key = "cnot_d5"
    blk = pareto["pareto"][key]

    for ax, xkey, xlabel in [
        (axes[0], "mean_us", "mean decode latency per shot (µs)"),
        (axes[1], "p99_us", "p99 decode latency per shot (µs)"),
    ]:
        for name, rows in blk["curves"].items():
            if name not in STYLE:
                continue
            x = np.array([r[xkey] for r in rows])
            y = np.array([r["ler_per_gate"] for r in rows])
            o = np.argsort(x)
            ax.plot(x[o], np.maximum(y[o], 1e-7), label=name, **STYLE[name])

        # The two anchors, converted to the same per-gate units as the curves.
        n_cx = blk["n_cx"]
        m = per_gate_error(blk["matching_ler"], n_cx)
        b = per_gate_error(blk["bposd_ler"], n_cx)
        ax.axhline(m, color="#999999", lw=1, ls="-")
        ax.text(ax.get_xlim()[0], m * 1.15, "matching alone (never escalate)",
                fontsize=7.5, color="#666666")
        ax.axhline(b, color="#8172B3", lw=1, ls="-")
        ax.text(ax.get_xlim()[0], b * 1.15, "BP+OSD always (escalate everything)",
                fontsize=7.5, color="#8172B3")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("logical error per transversal CNOT")

    axes[0].set_title("Pareto frontier — mean latency", fontsize=10.5)
    axes[1].set_title("… and the tail, which is what builds the backlog", fontsize=10.5)
    axes[0].legend(frameon=False, fontsize=7.6, loc="lower left", ncol=2)
    fig.suptitle(
        "Transversal CNOT, d = 5, p = 1e-3, heralded loss = 1e-3 — escalation restores "
        "the code distance",
        fontsize=10.5, y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGS / "02_pareto.png", dpi=170)
    print(f"wrote {FIGS / '02_pareto.png'}")


# --------------------------------------------------------------------------------------
def fig3_importance(pareto: dict) -> None:
    """Do the erasure features actually carry the gain? Answer it two ways, either way.

    Left: mean |SHAP| per feature, from LightGBM's own ``pred_contrib``. That says what the
    model *used*.

    Right: the ablation. Retrain with the heralded-loss features deleted and re-draw the
    escalation curve. That says what the model *needed* -- which is the question, and an
    importance bar chart cannot answer it.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))

    # ---- left: SHAP ------------------------------------------------------------------
    ax = axes[0]
    cname = "cnot"
    with open(DATA / f"model_{cname}.pkl", "rb") as f:
        mp = pickle.load(f)
    X = pd.read_parquet(DATA / f"eval_{cname}_d5.parquet")[ALL_FEATURES].to_numpy(np.float32)
    sample = X[np.random.default_rng(0).choice(len(X), size=min(20000, len(X)), replace=False)]
    contrib = mp["lgbm"].predict(sample, pred_contrib=True)  # (n, n_features + 1 bias)
    shap = np.abs(contrib[:, :-1]).mean(axis=0)
    shap = shap / max(shap.sum(), 1e-12) * 100

    order = np.argsort(shap)[-18:]
    names = [ALL_FEATURES[i] for i in order]
    vals = shap[order]

    def colour(n: str) -> str:
        if n in ERASURE_FEATURES:
            return "#55A868"
        if n in GAP:
            return "#DD8452"
        if n in EDGE:
            return "#8172B3"
        if n in MATCH:
            return "#4C72B0"
        return "#BBBBBB"

    ax.barh(range(len(vals)), vals, color=[colour(n) for n in names], height=0.75)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(names, fontsize=7.5)
    ax.set_xlabel("mean |SHAP|  (% of total attribution)")
    ax.set_title("What the model used  (cnot, trained on d = 3, 5)", fontsize=10.5)
    ax.grid(axis="y", alpha=0)

    erase_share = shap[[ALL_FEATURES.index(f) for f in ERASURE]].sum()
    gap_share = shap[[ALL_FEATURES.index(f) for f in GAP]].sum()
    ax.text(
        0.97, 0.05,
        f"heralded loss: {erase_share:.0f}%\ncomplementary gap: {gap_share:.0f}%",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.8", lw=0.6),
    )
    handles = [plt.Rectangle((0, 0), 1, 1, color=c)
               for c in ["#DD8452", "#55A868", "#8172B3", "#4C72B0", "#BBBBBB"]]
    ax.legend(handles,
              ["complementary gap", "heralded loss", "hyperedge structure",
               "matching weight", "syndrome shape"],
              loc="lower right", bbox_to_anchor=(1.0, 0.18), frameon=False, fontsize=7.2)

    # ---- right: the ablation ---------------------------------------------------------
    ax = axes[1]
    blk = pareto["pareto"]["cnot_d5"]
    for name in ("oracle", "LightGBM", "LightGBM (no erasure)", "LightGBM (no gap)",
                 "gap only", "random"):
        rows = blk["curves"].get(name)
        if not rows:
            continue
        x = np.array([r["rate"] for r in rows]) * 100
        y = np.array([r["ler_per_gate"] for r in rows])
        o = np.argsort(x)
        st = dict(STYLE[name])
        st["marker"] = None
        ax.plot(x[o], np.maximum(y[o], 1e-7), label=name, **st)
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_yscale("log")
    ax.set_xlabel("shots escalated to BP+OSD (%)")
    ax.set_ylabel("logical error per transversal CNOT")
    ax.set_title("What the model needed  (ablation, cnot d = 5)", fontsize=10.5)
    ax.legend(frameon=False, fontsize=7.6, loc="lower left")

    fig.tight_layout()
    fig.savefig(FIGS / "03_feature_importance.png", dpi=170)
    print(f"wrote {FIGS / '03_feature_importance.png'}")


# --------------------------------------------------------------------------------------
def fig4_generalisation(pareto: dict) -> None:
    """Train at d = 3, 5. Evaluate at d = 7 without retraining. Two outcomes, two stories."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    for ax, cname in zip(axes, ["cnot", "chain3"]):
        for d, alpha in [(3, 0.35), (5, 0.65), (7, 1.0)]:
            key = f"{cname}_d{d}"
            if key not in pareto["pareto"]:
                continue
            blk = pareto["pareto"][key]
            held_out = d == 7
            for name in ("LightGBM", "gap only"):
                rows = blk["curves"].get(name)
                if not rows:
                    continue
                x = np.array([r["rate"] for r in rows]) * 100
                y = np.array([r["ler_per_gate"] for r in rows])
                o = np.argsort(x)
                st = dict(STYLE[name])
                st["alpha"] = alpha
                st["lw"] = 2.4 if held_out else 1.5
                st["marker"] = None
                lab = f"{name}, d={d}" + (" (held out)" if held_out else "")
                ax.plot(x[o], np.maximum(y[o], 1e-8), label=lab, **st)
        ax.set_xscale("symlog", linthresh=0.1)
        ax.set_yscale("log")
        ax.set_xlabel("shots escalated to BP+OSD (%)")
        ax.set_ylabel("logical error per transversal CNOT")
        ax.set_title(f"{cname} — model trained on d = 3, 5 only", fontsize=10.5)
        ax.legend(frameon=False, fontsize=7.2, loc="lower left")

    fig.suptitle(
        "Generalisation across code distance: the classifier never saw d = 7",
        fontsize=10.5, y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGS / "04_generalization.png", dpi=170)
    print(f"wrote {FIGS / '04_generalization.png'}")


if __name__ == "__main__":
    pareto = load("07_pareto.json")
    fig2_pareto(pareto)
    fig3_importance(pareto)
    fig4_generalisation(pareto)
    print("\nfigures 1-4 in figs/")
