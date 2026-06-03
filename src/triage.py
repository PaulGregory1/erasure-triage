"""The triage models and their baselines.

The ordering of §8.1 is deliberate and is respected here. The one that matters is
**gap-only**: a threshold on the complementary gap costs one extra matching and no
training. If LightGBM on 34 features does not beat it, that is the result, and it is
a more defensible thing to write than an inflated number.

Every score is "how much do I want to escalate this shot", higher = escalate sooner.
A single threshold tau then sweeps out the Pareto front.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Scorer:
    name: str
    scores: np.ndarray  # higher = more likely that the fast decoder got it wrong
    cost_matchings: float  # extra matching-equivalents per shot, on top of the fast decode
    note: str = ""


# --------------------------------------------------------------------------------------
def gap_scorer(X: np.ndarray, names: list[str]) -> Scorer:
    """Baseline 4: escalate when the complementary gap is small.

    The gap is the weight difference between the best correction and the best correction
    forced into the *other* logical class. A small gap means the two classes are nearly
    tied, i.e. the matcher is guessing. This is the reference uncertainty estimator and
    the bar the learned model has to clear.
    """
    g = X[:, names.index("gap_min")]
    return Scorer("gap only", -g, cost_matchings=1.0,
                  note="threshold on the complementary gap; no training")


def random_scorer(n: int, seed: int = 0) -> Scorer:
    """Baseline 5: escalate at random. If a model cannot beat this, it is broken."""
    rng = np.random.default_rng(seed)
    return Scorer("random", rng.random(n), cost_matchings=0.0)


def weight_scorer(X: np.ndarray, names: list[str]) -> Scorer:
    """Free baseline: escalate on total matching weight. Costs nothing extra."""
    return Scorer("matching weight", X[:, names.index("w_match")], cost_matchings=0.0)


def ndet_scorer(X: np.ndarray, names: list[str]) -> Scorer:
    """Free baseline: escalate when many detectors fired."""
    return Scorer("n detectors", X[:, names.index("n_det")], cost_matchings=0.0)


def oracle_scorer(y: np.ndarray, seed: int = 0) -> Scorer:
    """Baseline 3: escalate exactly on the shots the fast decoder gets wrong.

    Unreachable by construction -- it is the ceiling of the Pareto plot.

    The scores are ``y + u`` with u uniform on [0, 1), not just ``y``. Failures land in
    [1, 2) and successes in [0, 1), so they never interleave -- the oracle stays an oracle.
    But the threshold now sweeps *continuously* through both regimes: above 1 it escalates a
    growing fraction of the true failures, below 1 it escalates all of them plus a growing
    fraction of the (unnecessary) successes. A bare binary score would give the sweep two
    points and a straight line drawn across the gap between them.
    """
    rng = np.random.default_rng(seed)
    return Scorer("oracle", y.astype(float) + rng.random(len(y)), cost_matchings=0.0)


# --------------------------------------------------------------------------------------
def train_lightgbm(
    X: np.ndarray,
    y: np.ndarray,
    names: list[str],
    seed: int = 0,
    n_estimators: int = 400,
    calibrate: bool = True,
) -> tuple[object, object]:
    """LightGBM + isotonic calibration.

    Calibration matters because we threshold the output probability: if we escalate at
    tau = 0.1 we want that to mean "about 10% of these shots really are failures", or the
    Pareto axis is meaningless.
    """
    import lightgbm as lgb
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import train_test_split

    Xtr, Xcal, ytr, ycal = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
    pos = max(int(ytr.sum()), 1)
    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        num_leaves=48,
        max_depth=6,
        learning_rate=0.06,
        min_child_samples=40,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        # The classes are wildly unbalanced (~1-5% positives at p_train, far less at
        # p_target). Reweighting keeps the trees from simply predicting "never escalate".
        scale_pos_weight=(len(ytr) - pos) / pos,
        random_state=seed,
        n_jobs=1,
        verbose=-1,
    )
    model.fit(Xtr, ytr, feature_name=list(names))

    iso = None
    if calibrate:
        raw = model.predict_proba(Xcal)[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(raw, ycal)
    return model, iso


def lgbm_scorer(model, iso, X: np.ndarray, gap_used: bool, label: str | None = None) -> Scorer:
    p = model.predict_proba(X)[:, 1]
    if iso is not None:
        p = iso.predict(p)
    return Scorer(
        label or ("LightGBM" if gap_used else "LightGBM (no gap)"),
        p,
        cost_matchings=1.0 if gap_used else 0.0,
        note="calibrated escalation probability",
    )


# --------------------------------------------------------------------------------------
def train_mlp(X: np.ndarray, y: np.ndarray, seed: int = 0, epochs: int = 25):
    """A deliberately tiny MLP, so we can say we arbitrated instead of following fashion.

    Two hidden layers, a few thousand parameters, CPU. If it does not beat the trees,
    that is worth reporting: inference cost is the binding constraint here, and a model
    that is 100x more expensive at inference has to earn it.
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    torch.set_num_threads(1)

    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn = ((X - mu) / sd).astype(np.float32)

    net = nn.Sequential(
        nn.Linear(X.shape[1], 64), nn.ReLU(),
        nn.Linear(64, 32), nn.ReLU(),
        nn.Linear(32, 1),
    )
    n_params = sum(p.numel() for p in net.parameters())

    pos_weight = torch.tensor([(len(y) - y.sum()) / max(y.sum(), 1)], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)

    xt = torch.from_numpy(Xn)
    yt = torch.from_numpy(y.astype(np.float32)).unsqueeze(1)
    n = len(xt)
    bs = 4096
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            opt.zero_grad()
            loss = loss_fn(net(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
    net.eval()
    return net, (mu, sd), n_params


def mlp_scorer(net, stats, X: np.ndarray) -> Scorer:
    import torch

    mu, sd = stats
    Xn = ((X - mu) / sd).astype(np.float32)
    with torch.no_grad():
        p = torch.sigmoid(net(torch.from_numpy(Xn))).numpy().ravel()
    return Scorer("MLP", p, cost_matchings=1.0)


# --------------------------------------------------------------------------------------
def escalation_curve(
    scorer: Scorer,
    y_fast_wrong: np.ndarray,
    y_slow_wrong: np.ndarray,
    n_points: int = 60,
) -> dict:
    """Sweep the escalation threshold and record (escalation rate, logical error).

    A shot that is escalated is answered by BP+OSD, so it fails iff BP+OSD fails.
    A shot that is not escalated keeps the fast decoder's answer.

        LER(tau) = P(fast wrong AND not escalated) + P(slow wrong AND escalated)

    Note the second term: escalating is not free of risk. BP+OSD occasionally breaks a
    shot that matching had right, and at low escalation rates that can dominate.
    """
    s = scorer.scores
    # Quantiles alone collapse for a heavily-skewed score (the oracle is nearly all zeros),
    # so mix in a linear grid over the score range.
    qs = np.unique(np.concatenate([
        np.quantile(s, np.linspace(0, 1, n_points)),
        np.linspace(float(s.min()), float(s.max()), n_points),
    ]))
    out = {"rate": [], "ler": [], "tau": []}
    for tau in qs:
        esc = s >= tau
        fails = np.where(esc, y_slow_wrong, y_fast_wrong)
        out["rate"].append(float(esc.mean()))
        out["ler"].append(float(fails.mean()))
        out["tau"].append(float(tau))
    # never escalate
    out["rate"].append(0.0)
    out["ler"].append(float(y_fast_wrong.mean()))
    out["tau"].append(float(s.max() + 1))
    order = np.argsort(out["rate"])
    return {
        "name": scorer.name,
        "cost_matchings": scorer.cost_matchings,
        "rate": list(np.array(out["rate"])[order]),
        "ler": list(np.array(out["ler"])[order]),
        "tau": list(np.array(out["tau"])[order]),
    }
