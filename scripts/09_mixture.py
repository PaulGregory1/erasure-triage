"""Why the classifier does not transfer across noise level -- and the fix.

Observed in Phase 6: LightGBM beats the gap-only heuristic *in distribution* (at p_train)
and then **loses to it** at p_target. That is a transfer failure, and §7.3 says to report it
rather than bury it. But it is also diagnosable.

The cause is units. A matching edge carries weight log((1-p)/p): about 5.8 at p = 3e-3 and
about 6.9 at p = 1e-3. Every weight-derived feature -- the matching weight, the complementary
gap -- therefore *shifts* when the noise level changes. A tree that learned "escalate when
gap < 5.0" at p_train is asking a different question at p_target. The gap-only baseline is
immune because it never fixes an absolute threshold: it ranks, and the threshold is swept.

Two fixes are tested here:

  mixed-p     train on a blend of noise levels so the trees see the shift
  ratio-only  keep only the scale-free features (gap / weight, and the counts), deleting the
              raw weights entirely

If either recovers the in-distribution advantage at p_target, the transfer failure was an
artefact of feature units, not evidence that the learned model is useless.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.circuits import cnot_chain, transversal_cnot  # noqa: E402
from src.config import shots as _shots  # noqa: E402
from src.features import ALL_FEATURES  # noqa: E402
from src.noise import NoiseModel  # noqa: E402
from src.sampling import generate_dataset  # noqa: E402
from src.triage import escalation_curve, gap_scorer, lgbm_scorer, train_lightgbm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

LOSS_RATIO = 1.0
MIX_PS = (1.0e-3, 1.7e-3, 3.0e-3)  # p_target ... p_train, log-spaced
SHOTS_PER_P = _shots(700_000)

PROGRAMS = {"cnot": lambda: transversal_cnot(1, "Z"), "chain3": lambda: cnot_chain(1, "Z")}

# Features whose value does not depend on the units of a matching weight. The raw weight and
# the raw gap are in log-odds and move with p; their ratio does not, and neither do counts.
SCALE_FREE = [f for f in ALL_FEATURES if f not in ("w_match", "w_match_norm",
                                                   "gap_min", "gap_mean", "gap_max")]


def load(tag: str):
    df = pd.read_parquet(DATA / f"{tag}.parquet")
    return (df[ALL_FEATURES].to_numpy(np.float32),
            df["y"].to_numpy(np.int8),
            df["y_bposd"].to_numpy(np.int8))


def ler_at(curve: dict, rate: float) -> float:
    return float(np.interp(rate, np.array(curve["rate"]), np.array(curve["ler"])))


def build_mixture(cname: str, d: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    path = DATA / f"trainmix_{cname}_d{d}.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        return df[ALL_FEATURES].to_numpy(np.float32), df["y"].to_numpy(np.int8)
    Xs, ys = [], []
    for i, p in enumerate(MIX_PS):
        ds = generate_dataset(
            PROGRAMS[cname](), d, NoiseModel.uniform(p, loss_ratio=LOSS_RATIO),
            shots=SHOTS_PER_P, seed=seed + i, with_bposd=False, chunk=50_000,
        )
        Xs.append(ds["X"])
        ys.append(ds["y"])
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    df = pd.DataFrame(X, columns=ALL_FEATURES)
    df["y"] = y
    df.to_parquet(path, index=False)
    return X, y


if __name__ == "__main__":
    report: dict = {"mix_ps": list(MIX_PS), "shots_per_p": SHOTS_PER_P, "rows": []}
    print("Weight units move with the noise level:")
    for p in (1e-3, 3e-3):
        print(f"  p = {p:.0e}  ->  an edge of that probability weighs "
              f"log((1-p)/p) = {np.log((1 - p) / p):.2f}")
    print()

    for cname in ("cnot", "chain3"):
        print(f"\n{'=' * 88}\n{cname}\n{'=' * 88}")
        Xmix, ymix = [], []
        for d in (3, 5):
            t0 = time.perf_counter()
            X, y = build_mixture(cname, d, seed=50_000 + 10 * d)
            Xmix.append(X)
            ymix.append(y)
            print(f"  mixture train set d={d}: {len(y):,} shots, {y.mean():.2%} positives "
                  f"({time.perf_counter() - t0:.0f}s)")
        Xmix = np.concatenate(Xmix)
        ymix = np.concatenate(ymix)

        # the two candidate fixes, plus the Phase-6 model retrained here for a fair comparison
        Xhi, yhi = [], []
        for d in (3, 5):
            df = pd.read_parquet(DATA / f"train_{cname}_d{d}.parquet")
            Xhi.append(df[ALL_FEATURES].to_numpy(np.float32))
            yhi.append(df["y"].to_numpy(np.int8))
        Xhi = np.concatenate(Xhi)
        yhi = np.concatenate(yhi)

        m_hi, i_hi = train_lightgbm(Xhi, yhi, ALL_FEATURES, seed=0)
        m_mix, i_mix = train_lightgbm(Xmix, ymix, ALL_FEATURES, seed=0)
        idx_sf = [ALL_FEATURES.index(f) for f in SCALE_FREE]
        m_sf, i_sf = train_lightgbm(Xhi[:, idx_sf], yhi, SCALE_FREE, seed=0)

        for d in (3, 5, 7):
            tag = f"eval_{cname}_d{d}"
            if not (DATA / f"{tag}.parquet").exists():
                continue
            X, y, yb = load(tag)
            models = [
                ("gap only (no training)", gap_scorer(X, ALL_FEATURES)),
                ("LightGBM @ p_train", lgbm_scorer(m_hi, i_hi, X, gap_used=True)),
                ("LightGBM @ mixed p", lgbm_scorer(m_mix, i_mix, X, gap_used=True)),
                ("LightGBM scale-free", lgbm_scorer(m_sf, i_sf, X[:, idx_sf], gap_used=True)),
            ]
            print(f"\n  {tag}   matching LER {y.mean():.3e}   BP+OSD LER {yb.mean():.3e}"
                  + ("   [d held out]" if d == 7 else ""))
            print(f"    {'model':<26} | {'LER @ 1%':>10} {'@ 5%':>10} {'@ 10%':>10}")
            for label, sc in models:
                c = escalation_curve(sc, y, yb)
                vals = [ler_at(c, r) for r in (0.01, 0.05, 0.10)]
                print(f"    {label:<26} | {vals[0]:>10.2e} {vals[1]:>10.2e} {vals[2]:>10.2e}")
                report["rows"].append({
                    "circuit": cname, "d": d, "model": label,
                    "ler_1pct": vals[0], "ler_5pct": vals[1], "ler_10pct": vals[2],
                    "matching_ler": float(y.mean()), "bposd_ler": float(yb.mean()),
                })

    (DATA / "09_mixture.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {DATA / '09_mixture.json'}")
