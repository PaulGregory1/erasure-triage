"""Phase 6 -- train the triage classifier and run it against the baseline ladder.

The ladder of §8.1, in order:

  1. matching alone            latency floor
  2. BP+OSD always             accuracy ceiling (and, it turns out, not even that)
  3. oracle escalation         the unreachable ceiling from Phase 4
  4. gap only                  ** the baseline that matters **
  5. random escalation         sanity check; losing to this means something is broken
  6. LightGBM on all features
  7. a tiny MLP                so we can say we arbitrated instead of following fashion

If (6) does not beat (4), we write that down. "A near-free heuristic captures X% of the
gain" is a stronger and more defensible claim than an inflated number.
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features import ALL_FEATURES, ERASURE, GAP  # noqa: E402
from src.triage import (  # noqa: E402
    escalation_curve,
    gap_scorer,
    lgbm_scorer,
    mlp_scorer,
    ndet_scorer,
    oracle_scorer,
    random_scorer,
    train_lightgbm,
    train_mlp,
    weight_scorer,
)

ROOT = Path(__file__).resolve().parents[1]
DATA, FIGS = ROOT / "data", ROOT / "figs"
FIGS.mkdir(exist_ok=True)

NO_GAP = [f for f in ALL_FEATURES if f not in GAP]
NO_ERASURE = [f for f in ALL_FEATURES if f not in ERASURE]


def load(tag: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    df = pd.read_parquet(DATA / f"{tag}.parquet")
    X = df[ALL_FEATURES].to_numpy(dtype=np.float32)
    y = df["y"].to_numpy(dtype=np.int8)
    yb = df["y_bposd"].to_numpy(dtype=np.int8) if "y_bposd" in df else None
    return X, y, yb


def auc(scores: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, scores))


def gain_at(curve: dict, rate: float) -> float:
    """Logical error achieved when escalating `rate` of shots."""
    r = np.array(curve["rate"])
    ler = np.array(curve["ler"])
    return float(np.interp(rate, r, ler))


if __name__ == "__main__":
    circuits = ["cnot", "chain3"]
    report: dict = {"models": {}, "curves": {}, "auc": {}, "meta": {}}

    # ---- train on d = 3 and d = 5 pooled -------------------------------------------
    for cname in circuits:
        Xs, ys = [], []
        for d in (3, 5):
            X, y, _ = load(f"train_{cname}_d{d}")
            Xs.append(X)
            ys.append(y)
        Xtr = np.concatenate(Xs)
        ytr = np.concatenate(ys)
        print(f"\n{'=' * 78}\n{cname}: training on d=3,5 pooled -- "
              f"{len(ytr):,} shots, {ytr.mean():.2%} positives\n{'=' * 78}")

        t0 = time.perf_counter()
        model, iso = train_lightgbm(Xtr, ytr, ALL_FEATURES, seed=0)
        t_lgbm = time.perf_counter() - t0

        # same model class, but denied the gap: does the gap earn its extra matching?
        idx_ng = [ALL_FEATURES.index(f) for f in NO_GAP]
        t0 = time.perf_counter()
        model_ng, iso_ng = train_lightgbm(Xtr[:, idx_ng], ytr, NO_GAP, seed=0)
        t_ng = time.perf_counter() - t0

        # Ablation: the same model, denied the heralded-loss features. This is the only
        # direct answer to "do the erasure features carry the gain?" -- importance plots
        # tell you what the model *used*, not what it *needed*.
        idx_ne = [ALL_FEATURES.index(f) for f in NO_ERASURE]
        model_ne, iso_ne = train_lightgbm(Xtr[:, idx_ne], ytr, NO_ERASURE, seed=0)

        t0 = time.perf_counter()
        net, stats, n_params = train_mlp(Xtr, ytr, seed=0)
        t_mlp = time.perf_counter() - t0
        print(f"  LightGBM   trained in {t_lgbm:5.1f}s")
        print(f"  LightGBM(no gap) in  {t_ng:5.1f}s")
        print(f"  MLP ({n_params:,} params) in {t_mlp:5.1f}s")

        with open(DATA / f"model_{cname}.pkl", "wb") as f:
            pickle.dump(
                {"lgbm": model, "iso": iso,
                 "lgbm_nogap": model_ng, "iso_nogap": iso_ng,
                 "lgbm_noerasure": model_ne, "iso_noerasure": iso_ne,
                 "features": ALL_FEATURES, "features_nogap": NO_GAP,
                 "features_noerasure": NO_ERASURE,
                 "idx_nogap": idx_ng, "idx_noerasure": idx_ne},
                f,
            )

        # ---- evaluate --------------------------------------------------------------
        for tag_prefix, regime in [("eval", "p_target"), ("evalhi", "p_train")]:
            for d in ((3, 5, 7) if tag_prefix == "eval" else (3, 5)):
                tag = f"{tag_prefix}_{cname}_d{d}"
                if not (DATA / f"{tag}.parquet").exists():
                    continue
                X, y, yb = load(tag)
                if yb is None:
                    continue
                key = f"{cname}_d{d}_{regime}"

                scorers = [
                    oracle_scorer(y),
                    gap_scorer(X, ALL_FEATURES),
                    weight_scorer(X, ALL_FEATURES),
                    ndet_scorer(X, ALL_FEATURES),
                    random_scorer(len(y), seed=1),
                    lgbm_scorer(model, iso, X, gap_used=True),
                    lgbm_scorer(model_ng, iso_ng, X[:, idx_ng], gap_used=False),
                    lgbm_scorer(model_ne, iso_ne, X[:, idx_ne], gap_used=True,
                                label="LightGBM (no erasure)"),
                    mlp_scorer(net, stats, X),
                ]
                report["curves"][key] = [escalation_curve(s, y, yb) for s in scorers]
                report["auc"][key] = {
                    s.name: (auc(s.scores, y) if y.sum() > 5 else float("nan"))
                    for s in scorers if s.name != "oracle"
                }
                report["meta"][key] = {
                    "matching_ler": float(y.mean()),
                    "bposd_ler": float(yb.mean()),
                    "n_shots": int(len(y)),
                    "n_fail": int(y.sum()),
                }

                # "in-distribution" means the model saw this d AND this noise level.
                seen_d = d in (3, 5)
                in_dist = seen_d and regime == "p_train"
                tags = [] if in_dist else (
                    (["new d"] if not seen_d else []) + (["new p"] if regime == "p_target" else [])
                )
                label = "in-distribution" if in_dist else "transfer: " + ", ".join(tags)
                print(f"\n  {tag}  [{label}]  n={len(y):,}  "
                      f"matching LER={y.mean():.3e}  bposd LER={yb.mean():.3e}")
                print(f"    {'scorer':<20} {'AUC':>6} | LER @ escalation rate  1%      5%     10%")
                for s, c in zip(scorers, report["curves"][key]):
                    a = report["auc"][key].get(s.name, float("nan"))
                    print(
                        f"    {s.name:<20} {a:>6.3f} | "
                        f"{gain_at(c, 0.01):>21.2e} {gain_at(c, 0.05):>7.2e} "
                        f"{gain_at(c, 0.10):>7.2e}"
                    )

        report["models"][cname] = {
            "lgbm_train_s": t_lgbm, "mlp_train_s": t_mlp, "mlp_params": n_params,
            "n_train": int(len(ytr)), "pos_rate": float(ytr.mean()),
        }

    (DATA / "06_triage.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {DATA / '06_triage.json'}")
