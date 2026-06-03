"""Phase 5 -- build the training and evaluation sets.

Class imbalance is the trap (§7.3). At p_target the fast decoder fails ~1e-3 of the
time, so 1e5 positives would need 1e8 shots. Instead:

  train at p_train = 3 x p_target, where the failure rate is a workable few percent
  evaluate at p_target, and check the model transfers

The eval sets also carry BP+OSD's verdict per shot, because the Pareto curve needs to
know what escalation would actually have bought on that shot -- including the shots
where BP+OSD would have made things *worse*.
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
from src.noise import NoiseModel  # noqa: E402
from src.sampling import generate_dataset  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

P_TARGET = 1e-3
P_TRAIN = 3e-3
LOSS_RATIO = 1.0

TRAIN_SHOTS = _shots(2_000_000)

# The eval sets need BP+OSD on *every* shot, and BP+OSD costs 0.08 ms per shot at d=3 but
# ~23 ms on chain3 at d=5 and ~150 ms at d=7. So the eval budget is set per configuration
# from that cost, not from a round number: enough shots to leave a few hundred fast-decoder
# failures in each set, and no more. The d=7 sets are the held-out generalisation test, so
# they only need to resolve the *shape* of the curve, not its fourth digit.
EVAL_SHOTS = {
    k: _shots(v, floor=10_000) for k, v in {
        ("cnot", 3): 300_000, ("cnot", 5): 300_000, ("cnot", 7): 150_000,
        ("chain3", 3): 200_000, ("chain3", 5): 100_000, ("chain3", 7): 20_000,
    }.items()
}
# Small chunks: BP+OSD is the bottleneck and the pool has 10 workers, so a 50k chunk would
# leave most of them idle at the end of a set.
CHUNK = 10_000

PROGRAMS = {
    "cnot": lambda: transversal_cnot(1, "Z"),
    "chain3": lambda: cnot_chain(1, "Z"),
}
# Train on d = 3, 5. Hold d = 7 out entirely: §9.3 asks whether the model transfers
# across code distance without retraining, and it cannot answer that if it has seen d=7.
TRAIN_D = (3, 5)
EVAL_D = (3, 5, 7)


def build(tag: str, prog, d: int, p: float, shots: int, with_bposd: bool, seed: int) -> dict:
    # Resumable: the BP+OSD eval sets take tens of minutes, and losing them to an
    # interrupted run is not acceptable. Delete data/*.parquet to force a rebuild.
    path = DATA / f"{tag}.parquet"
    if path.exists():
        df = pd.read_parquet(path, columns=["y"])
        print(f"  {tag:<26} {len(df):>9,} shots  {df['y'].mean():>7.3%} positives  "
              f"(cached, skipped)")
        return {"tag": tag, "d": d, "p": p, "shots": int(len(df)),
                "pos_rate": float(df["y"].mean()), "seconds": 0.0, "cached": True}

    t0 = time.perf_counter()
    ds = generate_dataset(
        prog, d, NoiseModel.uniform(p, loss_ratio=LOSS_RATIO),
        shots=shots, seed=seed, with_bposd=with_bposd,
        chunk=CHUNK if with_bposd else 50_000,
    )
    dt = time.perf_counter() - t0
    df = pd.DataFrame(ds["X"], columns=ds["names"])
    df["y"] = ds["y"]
    if with_bposd:
        df["y_bposd"] = ds["y_bposd"]
    path = DATA / f"{tag}.parquet"
    df.to_parquet(path, index=False)
    pos = float(ds["y"].mean())
    print(
        f"  {tag:<26} {shots:>9,} shots  {pos:>7.3%} positives  "
        f"{dt:>6.1f}s  {shots / dt:>8,.0f} sh/s  -> {path.name}"
    )
    row = {"tag": tag, "d": d, "p": p, "shots": shots, "pos_rate": pos, "seconds": dt}
    if with_bposd:
        row["bposd_pos_rate"] = float(ds["y_bposd"].mean())
    return row


if __name__ == "__main__":
    meta = {"p_target": P_TARGET, "p_train": P_TRAIN, "loss_ratio": LOSS_RATIO, "sets": []}
    seed = 10_000

    print(f"TRAIN sets at p_train = {P_TRAIN:.0e} (no BP+OSD needed: labels are free)")
    for cname, mk in PROGRAMS.items():
        for d in TRAIN_D:
            seed += 1
            meta["sets"].append(
                build(f"train_{cname}_d{d}", mk(), d, P_TRAIN, TRAIN_SHOTS, False, seed)
            )

    print(f"\nEVAL sets at p_target = {P_TARGET:.0e} (BP+OSD on every shot: the Pareto needs it)")
    for cname, mk in PROGRAMS.items():
        for d in EVAL_D:
            seed += 1
            meta["sets"].append(
                build(f"eval_{cname}_d{d}", mk(), d, P_TARGET, EVAL_SHOTS[(cname, d)], True, seed)
            )

    print(f"\nEVAL set at p_train (to separate 'model is bad' from 'model does not transfer')")
    for cname, mk in PROGRAMS.items():
        for d in TRAIN_D:
            seed += 1
            meta["sets"].append(
                build(f"evalhi_{cname}_d{d}", mk(), d, P_TRAIN, EVAL_SHOTS[(cname, d)] // 2, True, seed)
            )

    (DATA / "05_datasets.json").write_text(json.dumps(meta, indent=2))
    total = sum(s["shots"] for s in meta["sets"])
    mins = sum(s["seconds"] for s in meta["sets"]) / 60
    print(f"\n{total:,} shots total, {mins:.1f} min of wall time")
    print(f"wrote {DATA / '05_datasets.json'}")
