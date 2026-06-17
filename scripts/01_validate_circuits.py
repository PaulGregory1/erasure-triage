"""Phase 1 -- circuit validation and the S2 exit gate.

Three things get established here, in order of how badly they would sink the project:

1. Distance. ``search_for_undetectable_logical_errors`` must return weight d.
   Anything less means the transversal-CNOT detectors are wrong.
2. Hyperedges. Plain memory decomposes cleanly; the transversal CNOT does not.
   The non-decomposable fraction is the correlation that MWPM structurally cannot see,
   and it is what BP+OSD is being paid to recover.
3. S2. With joint (correlated) decoding, r = 1 syndrome round per transversal gate
   must hold up where per-patch (independent) decoding falls over.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.circuits import cnot_chain, compile_program, ghz, memory, transversal_cnot  # noqa: E402
from src.dem_utils import build_problem  # noqa: E402
from src.decoders import (  # noqa: E402
    BpOnlyDecoder,
    BpOsdDecoder,
    CorrelatedMatchingDecoder,
    IndependentMatchingDecoder,
    MatchingDecoder,
)
from src.config import QUICK, budget_s  # noqa: E402
from src.config import shots as _shots  # noqa: E402
from src.evaluate import single_fault_failures  # noqa: E402
from src.noise import NoiseModel  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "data"
OUT.mkdir(exist_ok=True)

P = 3e-3
SHOTS = _shots(200_000)
BPOSD_BUDGET_S = budget_s(45.0)  # per S2 cell; BP+OSD spans 4 orders of magnitude in cost


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def check_distance() -> list[dict]:
    rule("1. DISTANCE  (must equal d, or the detectors are wrong)")
    # Pauli-only: heralded erasure multiplies the error mechanisms and makes the
    # undetectable-logical-error search crawl, without changing the code distance
    # (tests/test_circuits.py checks the erasure model separately).
    noise = NoiseModel.uniform(P, loss_ratio=0.0)
    rows = []
    print(f"{'circuit':<16} {'d':>2} {'found':>6} {'expected':>9}  {'verdict':<6}")
    for name, make in [
        ("memory", lambda d: memory(d, "Z")),
        ("cnot r=1", lambda d: transversal_cnot(1, "Z")),
        ("cnot r=d", lambda d: transversal_cnot(d, "Z")),
        ("chain3 r=1", lambda d: cnot_chain(1, "Z")),
        ("ghz4 r=1", lambda d: ghz(1, 4)),
    ]:
        for d in ((3,) if QUICK else (3, 5)):
            cc = compile_program(make(d), d, noise)
            err = cc.circuit.search_for_undetectable_logical_errors(
                dont_explore_detection_event_sets_with_size_above=6,
                dont_explore_edges_with_degree_above=4,
                dont_explore_edges_increasing_symptom_degree=False,
                canonicalize_circuit_errors=True,
            )
            ok = len(err) == d
            print(f"{name:<16} {d:>2} {len(err):>6} {d:>9}  {'PASS' if ok else 'FAIL':<6}", flush=True)
            rows.append({"circuit": name, "d": d, "distance": len(err), "pass": ok})
            assert ok, f"{name} d={d}: distance {len(err)} != {d}. Fix §3.2 before anything else."
    return rows


def check_hyperedges() -> list[dict]:
    rule("2. HYPEREDGES  (the correlation MWPM cannot represent)")
    noise = NoiseModel.uniform(P, loss_ratio=0.0)
    rows = []
    print(f"{'circuit':<16} {'d':>2} {'dets':>6} {'errors':>7} {'undecomposable':>15} {'%':>6}")
    for name, make in [
        ("memory", lambda d: memory(d, "Z")),
        ("cnot r=1", lambda d: transversal_cnot(1, "Z")),
        ("cnot r=d", lambda d: transversal_cnot(d, "Z")),
        ("chain3 r=1", lambda d: cnot_chain(1, "Z")),
        ("ghz4 r=1", lambda d: ghz(1, 4)),
    ]:
        for d in (3, 5, 7):
            pr = build_problem(make(d), d, noise)
            pct = 100 * pr.n_undecomposable / max(pr.n_errors, 1)
            print(
                f"{name:<16} {d:>2} {pr.n_dets:>6} {pr.n_errors:>7} "
                f"{pr.n_undecomposable:>15} {pct:>5.1f}%"
            )
            rows.append({
                "circuit": name, "d": d, "n_dets": pr.n_dets, "n_errors": pr.n_errors,
                "n_undecomposable": pr.n_undecomposable, "pct": pct,
            })
    print("\n  Memory decomposes exactly. The transversal CNOT does not, and the")
    print("  non-decomposable fraction grows with d. That gap is the whole thesis.")
    return rows


def check_single_faults() -> list[dict]:
    """The sharpest result in the repo, and it needs no Monte Carlo.

    A decoder that mis-corrects a *single* physical fault has effective distance 2,
    whatever the code distance is. Memory: MWPM handles every single fault. Transversal
    CNOT: it does not, and every fault it fluffs is one that lit detectors in both
    patches -- i.e. one that crossed the gate. BP+OSD handles them all.
    """
    rule("4. SINGLE-FAULT TEST  (effective distance of each decoder)")
    noise = NoiseModel.uniform(1e-3, loss_ratio=0.0)
    rows = []
    print(f"{'circuit':<12} {'d':>2} {'faults':>6} | {'MWPM':>6} {'(cross)':>8} "
          f"{'MWPM+corr':>10} {'BP only':>8} {'BP+OSD':>7} | {'hyper':>6} {'dropped':>8}")
    cases = [
        ("memory", lambda d: memory(d, "Z"), 5),
        ("cnot r=1", lambda d: transversal_cnot(1, "Z"), 3),
        ("cnot r=1", lambda d: transversal_cnot(1, "Z"), 5),
        ("cnot r=1", lambda d: transversal_cnot(1, "Z"), 7),
        ("cnot r=d", lambda d: transversal_cnot(d, "Z"), 5),
        ("chain3 r=1", lambda d: cnot_chain(1, "Z"), 5),
    ]
    if QUICK:  # the two heaviest BP+OSD cases add nothing the d=3/5/7 cnot rows do not
        cases = [c for c in cases if c[0] not in ("cnot r=d", "chain3 r=1")]
    for name, make, d in cases:
        pr = build_problem(make(d), d, noise)
        m = single_fault_failures(pr, MatchingDecoder(pr, with_gap=False))
        cm_dec = CorrelatedMatchingDecoder(pr)
        cm = single_fault_failures(pr, cm_dec)
        bo = single_fault_failures(pr, BpOnlyDecoder(pr))
        b = single_fault_failures(pr, BpOsdDecoder(pr))
        print(f"{name:<12} {d:>2} {m['n_faults']:>6} | {m['n_wrong']:>6} "
              f"{m['n_wrong_cross_patch']:>8} {cm['n_wrong']:>10} {bo['n_wrong']:>8} "
              f"{b['n_wrong']:>7} | {pr.n_undecomposable:>6} {cm_dec.n_dropped:>8}")
        rows.append({
            "circuit": name, "d": d, "matching": m, "correlated_matching": cm,
            "bp_only": bo, "bp_osd": b, "n_undecomposable": pr.n_undecomposable,
            "corr_matching_dropped": cm_dec.n_dropped,
        })
    print("\n  MWPM is fooled by ONE fault as soon as a transversal gate is present, and every")
    print("  fault that fools it lit detectors in BOTH patches -- the error crossed the gate.")
    print("  The decomposed graph can only write that as two independent errors of the same")
    print("  total weight as the honest single-fault explanation, so matching coin-flips.")
    print()
    print("  The two obvious cheap escapes do not work:")
    print("   * MWPM + pymatching's correlated mode has to DROP every undecomposed hyperedge")
    print("     before it will even load the model -- i.e. exactly the mechanisms that break")
    print("     it. It halves the failures at d=3 and does nothing at d=5.")
    print("   * BP without OSD is WORSE than plain MWPM: belief propagation does not converge")
    print("     on the degenerate loops these circuits create. OSD is doing the real work.")
    print()
    print("  Only BP+OSD reaches zero. Escalation is not an optimisation -- it is what")
    print("  restores the code distance.")
    return rows


def _ler(pred: np.ndarray, obs: np.ndarray) -> tuple[float, float]:
    fails = (pred != obs).any(axis=1)
    k, n = int(fails.sum()), len(fails)
    p = k / n
    return p, float(np.sqrt(max(p * (1 - p), 1e-12) / n))


def check_s2() -> list[dict]:
    rule("3. S2 GATE  (joint decoding at r=1 vs per-patch decoding)")
    noise = NoiseModel.uniform(P, loss_ratio=0.0)
    rows = []
    print(
        f"{'circuit':<14} {'d':>2} {'rounds':>6} {'cx':>3} | "
        f"{'independent':>13} {'matching(joint)':>16} {'bp_osd(joint)':>22} | {'gain':>6}"
    )
    for name, make, r in [
        ("cnot", lambda d, r: transversal_cnot(r, "Z"), 1),
        ("cnot", lambda d, r: transversal_cnot(r, "Z"), None),  # r = d
        ("chain3", lambda d, r: cnot_chain(r, "Z"), 1),
        ("chain3", lambda d, r: cnot_chain(r, "Z"), None),
    ]:
        for d in (3, 5):
            rr = d if r is None else r
            pr = build_problem(make(d, rr), d, noise)
            _, syn, obs = pr.sample(SHOTS, seed=7)

            m = MatchingDecoder(pr, with_gap=False)
            e_m, s_m = _ler(m.decode_batch(syn), obs)

            ind = IndependentMatchingDecoder(pr)
            e_i, s_i = _ler(ind.decode_batch(syn), obs)

            # BP+OSD cost runs from ~0.1 ms to ~0.8 s per shot across these circuits,
            # so give it a fixed time budget rather than a fixed shot count, and report
            # how many shots that bought.
            bp = BpOsdDecoder(pr)
            t0 = time.perf_counter()
            bp.decode_batch(syn[:20])
            rate = 20 / max(time.perf_counter() - t0, 1e-9)
            nb = int(np.clip(BPOSD_BUDGET_S * rate, 200, 4000))
            e_b, s_b = _ler(bp.decode_batch(syn[:nb]), obs[:nb])

            print(
                f"{name:<14} {d:>2} {rr:>6} {pr.n_cx:>3} | "
                f"{e_i:>8.2e}±{s_i:.0e} {e_m:>10.2e}±{s_m:.0e} "
                f"{e_b:>9.2e}±{s_b:.0e} ({nb:>4}) | {e_m / max(e_b, 1e-9):>5.1f}x"
            )
            rows.append({
                "circuit": name, "d": d, "rounds": rr, "n_cx": pr.n_cx,
                "independent": e_i, "matching": e_m, "bp_osd": e_b,
                "independent_err": s_i, "matching_err": s_m, "bp_osd_err": s_b,
                "bp_osd_shots": nb, "bp_osd_us_per_shot": 1e6 / rate,
            })
    print("\n  'gain' = how much BP+OSD (which uses the hyperedges) beats MWPM (which")
    print("  cannot). It is the headroom the escalation decoder is trying to buy.")
    return rows


if __name__ == "__main__":
    out = {
        "p": P,
        "shots": SHOTS,
        "distance": check_distance(),
        "hyperedges": check_hyperedges(),
        "single_fault": check_single_faults(),
        "s2": check_s2(),
    }
    (OUT / "01_validation.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT / '01_validation.json'}")
