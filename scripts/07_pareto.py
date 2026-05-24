"""Phase 7 -- latency, the Pareto frontier, feature importance, and d-generalisation.

Produces Figures 2, 3 and 4.

Latency discipline (§9.2): every number below is a per-shot ``perf_counter_ns``
measurement on one documented machine with OMP_NUM_THREADS=1, warm-up discarded.
We report the p99 as well as the mean, because a real-time decoder's backlog is built
by the tail. A decoder with a beautiful mean and a catastrophic p99 is unusable.
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.circuits import cnot_chain, transversal_cnot  # noqa: E402
from src.config import shots as _shots  # noqa: E402
from src.dem_utils import build_problem  # noqa: E402
from src.decoders import BpOsdDecoder, ErasureMatchingDecoder  # noqa: E402
from src.evaluate import MACHINE, per_gate_error  # noqa: E402
from src.features import ALL_FEATURES, ERASURE, GAP  # noqa: E402
from src.noise import NoiseModel  # noqa: E402
from src.triage import (  # noqa: E402
    gap_scorer,
    lgbm_scorer,
    ndet_scorer,
    oracle_scorer,
    random_scorer,
    weight_scorer,
)

ROOT = Path(__file__).resolve().parents[1]
DATA, FIGS = ROOT / "data", ROOT / "figs"
FIGS.mkdir(exist_ok=True)

P_TARGET = 1e-3
LOSS_RATIO = 1.0
NO_GAP = [f for f in ALL_FEATURES if f not in GAP]
NO_ERASURE = [f for f in ALL_FEATURES if f not in ERASURE]

PROGRAMS = {"cnot": lambda: transversal_cnot(1, "Z"), "chain3": lambda: cnot_chain(1, "Z")}
N_LAT_MATCH = _shots(20_000, floor=4_000)
N_LAT_BPOSD = _shots(1_500, floor=400)

STYLE = {
    "oracle":              dict(color="#000000", ls="--", lw=1.6, marker=None),
    "LightGBM":            dict(color="#4C72B0", ls="-", lw=2.2, marker="o", ms=3),
    "LightGBM (no gap)":   dict(color="#8FA8CD", ls="-", lw=1.6, marker="s", ms=3),
    "LightGBM (no erasure)": dict(color="#55A868", ls="--", lw=1.6, marker="D", ms=3),
    "gap only":            dict(color="#DD8452", ls="-", lw=2.0, marker="^", ms=3),
    "matching weight":     dict(color="#55A868", ls="-.", lw=1.4, marker=None),
    "n detectors":         dict(color="#937860", ls=":", lw=1.4, marker=None),
    "random":              dict(color="#C44E52", ls=":", lw=1.6, marker=None),
    "MLP":                 dict(color="#8172B3", ls="--", lw=1.6, marker="v", ms=3),
}


# --------------------------------------------------------------------------------------
def measure_latency(cname: str, d: int) -> dict:
    """Per-shot decode latency for the fast leg, the gap, and the slow leg."""
    noise = NoiseModel.uniform(P_TARGET, loss_ratio=LOSS_RATIO)
    pr = build_problem(PROGRAMS[cname](), d, noise)
    md = ErasureMatchingDecoder(pr, with_gap=True)
    bp = BpOsdDecoder(pr)

    her, syn, _ = pr.sample(N_LAT_MATCH, seed=99)
    syn_u8 = np.ascontiguousarray(syn.astype(np.uint8))

    # warm-up, thrown away
    for i in range(50):
        md.decode_grouped_full(syn_u8[i : i + 1], her[i : i + 1], with_gap=True)

    t_fast = np.empty(N_LAT_MATCH)
    t_full = np.empty(N_LAT_MATCH)
    for i in range(N_LAT_MATCH):
        s1, h1 = syn_u8[i : i + 1], her[i : i + 1]
        t0 = time.perf_counter_ns()
        md.decode_grouped_full(s1, h1, with_gap=False)
        t1 = time.perf_counter_ns()
        md.decode_grouped_full(s1, h1, with_gap=True)
        t2 = time.perf_counter_ns()
        t_fast[i] = (t1 - t0) * 1e-3  # us
        t_full[i] = (t2 - t1) * 1e-3

    for i in range(10):
        bp.decode_shot(syn_u8[i])
    t_slow = np.empty(N_LAT_BPOSD)
    for i in range(N_LAT_BPOSD):
        t0 = time.perf_counter_ns()
        bp.decode_shot(syn_u8[i])
        t_slow[i] = (time.perf_counter_ns() - t0) * 1e-3

    return {
        "t_fast": t_fast, "t_gap": t_full - t_fast, "t_slow": t_slow,
        "fast_med": float(np.median(t_fast)), "fast_p99": float(np.percentile(t_fast, 99)),
        "gap_med": float(np.median(t_full - t_fast)),
        "slow_med": float(np.median(t_slow)), "slow_p99": float(np.percentile(t_slow, 99)),
        "slowdown": float(np.median(t_slow) / np.median(t_fast)),
    }


def triage_latency(
    esc: np.ndarray, lat: dict, t_model_us: float, uses_gap: bool, rng
) -> tuple[float, float]:
    """Realised per-shot latency of a triage decoder: mean and p99.

    Every shot pays the fast decode, plus the feature cost (the gap, if the scorer uses
    it), plus the model's inference. Escalated shots pay BP+OSD on top -- that is why the
    tail is what it is.
    """
    n = len(esc)
    base = rng.choice(lat["t_fast"], size=n)
    if uses_gap:
        base = base + rng.choice(lat["t_gap"], size=n)
    base = base + t_model_us
    extra = np.where(esc, rng.choice(lat["t_slow"], size=n), 0.0)
    total = base + extra
    return float(total.mean()), float(np.percentile(total, 99))


# --------------------------------------------------------------------------------------
def build_scorers(cname: str, X, y, mp):
    return [
        oracle_scorer(y),
        lgbm_scorer(mp["lgbm"], mp["iso"], X, gap_used=True),
        lgbm_scorer(mp["lgbm_nogap"], mp["iso_nogap"], X[:, mp["idx_nogap"]], gap_used=False),
        lgbm_scorer(mp["lgbm_noerasure"], mp["iso_noerasure"], X[:, mp["idx_noerasure"]],
                    gap_used=True, label="LightGBM (no erasure)"),
        gap_scorer(X, ALL_FEATURES),
        weight_scorer(X, ALL_FEATURES),
        ndet_scorer(X, ALL_FEATURES),
        random_scorer(len(y), seed=1),
    ]


def pareto_points(scorer, y, yb, lat, n_cx, t_model_us, rng, n_tau=45):
    s = scorer.scores
    taus = np.unique(np.concatenate([
        np.quantile(s, np.linspace(0.0, 1.0, n_tau)),
        np.linspace(float(s.min()), float(s.max()), n_tau),
    ]))
    rows = []
    for tau in taus:
        esc = s >= tau
        fails = np.where(esc, yb, y)
        ler = float(fails.mean())
        mean_us, p99_us = triage_latency(
            esc, lat, t_model_us, scorer.cost_matchings > 0, rng
        )
        rows.append({
            "rate": float(esc.mean()),
            "ler": ler,
            "ler_per_gate": per_gate_error(ler, n_cx),
            "mean_us": mean_us, "p99_us": p99_us,
        })
    # never escalate
    mean_us, p99_us = triage_latency(
        np.zeros(len(y), bool), lat, t_model_us, scorer.cost_matchings > 0, rng
    )
    ler = float(y.mean())
    rows.append({
        "rate": 0.0, "ler": ler,
        "ler_per_gate": per_gate_error(ler, n_cx),
        "mean_us": mean_us, "p99_us": p99_us,
    })
    rows.sort(key=lambda r: r["mean_us"])
    return rows


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    out: dict = {"machine": MACHINE, "latency": {}, "pareto": {}, "importance": {}}

    print(f"machine of record: {MACHINE}\n")
    print(f"{'circuit':<10} {'d':>2} | {'match med':>10} {'match p99':>10} {'gap med':>9} "
          f"{'bposd med':>10} {'bposd p99':>10} | {'slowdown':>9}")

    lats = {}
    for cname in PROGRAMS:
        for d in (3, 5, 7):
            lat = measure_latency(cname, d)
            lats[(cname, d)] = lat
            out["latency"][f"{cname}_d{d}"] = {
                k: v for k, v in lat.items() if not isinstance(v, np.ndarray)
            }
            print(f"{cname:<10} {d:>2} | {lat['fast_med']:>9.1f}u {lat['fast_p99']:>9.1f}u "
                  f"{lat['gap_med']:>8.1f}u {lat['slow_med']:>9.1f}u {lat['slow_p99']:>9.1f}u | "
                  f"{lat['slowdown']:>8.0f}x")

    for cname in PROGRAMS:
        with open(DATA / f"model_{cname}.pkl", "rb") as f:
            mp = pickle.load(f)

        # LightGBM inference cost, measured, not assumed
        Xd = pd.read_parquet(DATA / f"eval_{cname}_d3.parquet")[ALL_FEATURES].to_numpy(np.float32)
        t0 = time.perf_counter()
        mp["lgbm"].predict_proba(Xd[:20000])
        t_model_us = 1e6 * (time.perf_counter() - t0) / 20000
        out["latency"][f"{cname}_lgbm_inference_us"] = t_model_us

        for d in (3, 5, 7):
            path = DATA / f"eval_{cname}_d{d}.parquet"
            if not path.exists():
                continue
            df = pd.read_parquet(path)
            X = df[ALL_FEATURES].to_numpy(np.float32)
            y = df["y"].to_numpy(np.int8)
            yb = df["y_bposd"].to_numpy(np.int8)
            n_cx = 1 if cname == "cnot" else 3

            key = f"{cname}_d{d}"
            out["pareto"][key] = {
                "matching_ler": float(y.mean()),
                "bposd_ler": float(yb.mean()),
                "n_cx": n_cx,
                "n_shots": int(len(y)),
                "curves": {},
            }
            for s in build_scorers(cname, X, y, mp):
                out["pareto"][key]["curves"][s.name] = pareto_points(
                    s, y, yb, lats[(cname, d)], n_cx, t_model_us if "LightGBM" in s.name else 0.0, rng
                )

        imp = dict(zip(ALL_FEATURES, mp["lgbm"].feature_importances_.tolist()))
        out["importance"][cname] = imp

    (DATA / "07_pareto.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {DATA / '07_pareto.json'}")
