"""Phase 4 -- the test that decides the project. Run this before writing any ML.

Escalation is worthless if BP+OSD is not *right* precisely where matching is *wrong*.
So measure the ceiling before building anything that tries to reach it:

    P(BP+OSD correct | matching wrong)

  > 50%  -> there is real headroom. Build the triage.
  ~ 20%  -> escalation cannot buy much. Pivot: keep erasure reweighting as the
            headline result and turn the triage into "when does a fast decoder
            suffice?", which is still a good paper.

Also reported: the oracle escalation error rate, i.e. what you would get if you
escalated on exactly the shots where matching fails and never otherwise. No
learned model can beat it. It is the ceiling of Figure 2.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.circuits import cnot_chain, ghz, memory, transversal_cnot  # noqa: E402
from src.config import budget_s  # noqa: E402
from src.config import shots as _shots  # noqa: E402
from src.dem_utils import build_problem  # noqa: E402
from src.decoders import BpOsdDecoder, ErasureMatchingDecoder  # noqa: E402
from src.evaluate import Rate  # noqa: E402
from src.noise import NoiseModel  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

# p_target is chosen so matching fails ~1e-3 per shot, the regime §7.3 describes.
# p_train (3x) is where the dataset is generated; the ceiling is measured at both,
# because a classifier trained high and deployed low only helps if the ceiling
# survives the move.
P_TARGET = 1e-3
P_TRAIN = 3e-3
LOSS_RATIO = 1.0
SHOTS = _shots(150_000)
BPOSD_BUDGET_S = budget_s(120.0)

CASES = [
    ("cnot r=1", lambda d: transversal_cnot(1, "Z"), 3),
    ("cnot r=1", lambda d: transversal_cnot(1, "Z"), 5),
    ("cnot r=1", lambda d: transversal_cnot(1, "Z"), 7),
    ("chain3 r=1", lambda d: cnot_chain(1, "Z"), 3),
    ("chain3 r=1", lambda d: cnot_chain(1, "Z"), 5),
    ("memory", lambda d: memory(d, "Z"), 5),
]


def ceiling(name: str, prog, d: int, p: float, shots: int) -> dict:
    noise = NoiseModel.uniform(p, loss_ratio=LOSS_RATIO)
    pr = build_problem(prog, d, noise)

    her, syn, obs = pr.sample(shots, seed=abs(hash((name, d, p))) % 2**31)

    md = ErasureMatchingDecoder(pr, with_gap=False)
    t0 = time.perf_counter()
    pm = md.decode_grouped(syn, her)
    t_match = time.perf_counter() - t0

    # BP+OSD is 2-4 orders of magnitude slower than matching, so it gets a time budget.
    # We *must* run it on the shots where matching failed (those are the ones the ceiling
    # is about) plus a random sample of the rest, then reweight.
    bp = BpOsdDecoder(pr)
    t0 = time.perf_counter()
    bp.decode_batch(syn[:20])
    rate = 20 / max(time.perf_counter() - t0, 1e-9)
    budget = int(np.clip(BPOSD_BUDGET_S * rate, 500, shots))

    wrong_m = (pm != obs).any(axis=1)
    idx_fail = np.flatnonzero(wrong_m)
    rng = np.random.default_rng(0)
    n_rest = max(0, budget - len(idx_fail))
    idx_ok = np.flatnonzero(~wrong_m)
    idx_ok = rng.choice(idx_ok, size=min(n_rest, len(idx_ok)), replace=False)
    idx = np.concatenate([idx_fail, idx_ok])

    t0 = time.perf_counter()
    pb_sub = bp.decode_batch(syn[idx])
    t_bp = (time.perf_counter() - t0) / max(len(idx), 1) * shots

    wrong_b_sub = (pb_sub != obs[idx]).any(axis=1)
    wrong_b = np.zeros(shots, dtype=bool)
    wrong_b[idx] = wrong_b_sub
    # BP+OSD's own LER is estimated on the sampled subset, corrected for the biased
    # inclusion of every matching failure.
    f_fail = wrong_b_sub[: len(idx_fail)].mean() if len(idx_fail) else 0.0
    f_ok = wrong_b_sub[len(idx_fail):].mean() if len(idx_ok) else 0.0
    p_m = len(idx_fail) / shots
    bposd_ler_est = p_m * f_fail + (1 - p_m) * f_ok

    n = len(obs)
    n_mw = int(wrong_m.sum())
    n_both = int((wrong_m & wrong_b).sum())
    n_saved = int((wrong_m & ~wrong_b).sum())
    n_broken = int((wrong_b_sub[len(idx_fail):]).sum())

    p_rescue = n_saved / n_mw if n_mw else float("nan")
    oracle = Rate(n_both, n)  # escalate iff matching is wrong -> fail iff both fail

    return {
        "circuit": name, "d": d, "p": p, "shots": n, "n_cx": pr.n_cx,
        "bposd_shots": int(len(idx)),
        "matching_ler": Rate(n_mw, n).p,
        "matching_ler_err": Rate(n_mw, n).err,
        "bposd_ler": float(bposd_ler_est),
        "bposd_ler_err": float(np.sqrt(max(bposd_ler_est * (1 - bposd_ler_est), 1e-12) / max(len(idx), 1))),
        "oracle_ler": oracle.p,
        "oracle_ler_err": oracle.err,
        "p_bposd_right_given_matching_wrong": p_rescue,
        "n_matching_wrong": n_mw,
        "n_rescued": n_saved,
        "n_bposd_breaks_it": n_broken,
        "escalation_rate_oracle": n_mw / n,
        "t_match_us": 1e6 * t_match / n,
        "t_bposd_us": 1e6 * t_bp / n,
        "slowdown": t_bp / max(t_match, 1e-12),
    }


if __name__ == "__main__":
    rows = []
    print(f"p_loss = {LOSS_RATIO} x p; {SHOTS:,} shots per case")
    print(f"BP+OSD is run on every matching failure + a random sample of the rest,")
    print(f"then reweighted -- otherwise the ceiling would cost hours.\n")
    hdr = (
        f"{'circuit':<12} {'d':>2} | {'LER match':>10} {'LER bposd':>10} {'LER oracle':>11} | "
        f"{'P(bposd ok | match wrong)':>25} | {'rescued':>9} {'broken':>7}"
    )
    for p_label, p in [("p_target", P_TARGET), ("p_train", P_TRAIN)]:
      print(f"\n--- {p_label} = {p:.0e} ---")
      print(hdr)
      print("-" * len(hdr))
      for name, make, d in CASES:
        r = ceiling(name, make(d), d, p, SHOTS)
        r["regime"] = p_label
        rows.append(r)
        print(
            f"{name:<12} {d:>2} | {r['matching_ler']:>10.3e} {r['bposd_ler']:>10.3e} "
            f"{r['oracle_ler']:>11.3e} | {r['p_bposd_right_given_matching_wrong']:>24.1%} | "
            f"{r['n_rescued']:>4}/{r['n_matching_wrong']:<4} {r['n_bposd_breaks_it']:>7}"
        )

    print(f"\n{'circuit':<12} {'d':>2} | {'t_match (us)':>13} {'t_bposd (us)':>13} {'slowdown':>9}")
    for r in rows:
        print(
            f"{r['circuit']:<12} {r['d']:>2} | {r['t_match_us']:>13.2f} "
            f"{r['t_bposd_us']:>13.2f} {r['slowdown']:>8.0f}x"
        )

    vals = [r["p_bposd_right_given_matching_wrong"] for r in rows if r["n_matching_wrong"] > 30]
    med = float(np.median(vals))
    print("\n" + "=" * 78)
    print(f"GO/NO-GO:  median P(BP+OSD right | matching wrong) = {med:.1%}")
    if med > 0.5:
        print("  > 50%  -> real headroom. Build the triage classifier.")
    elif med > 0.3:
        print("  30-50% -> moderate headroom. Triage is worth it but report the ceiling loudly.")
    else:
        print("  < 30%  -> PIVOT. Erasure reweighting becomes the headline; the triage")
        print("            question becomes 'when does a fast decoder suffice?'.")
    print("=" * 78)

    (DATA / "04_oracle.json").write_text(
        json.dumps({"p_target": P_TARGET, "shots": SHOTS, "rows": rows, "median": med}, indent=2)
    )
    print(f"wrote {DATA / '04_oracle.json'}")
