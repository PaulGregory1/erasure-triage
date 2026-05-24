"""Reproduce everything.

    python run_all.py            # full run
    python run_all.py --quick    # ~10 min, coarser statistics, same conclusions

Every stage writes JSON into data/ and the figures into figs/. Stages are independent
given their inputs, so a crashed stage can be re-run on its own.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = str(ROOT / ".venv" / "bin" / "python")
if not Path(PY).exists():
    PY = sys.executable

STAGES = [
    ("01_validate_circuits", "distance, hyperedges, single-fault test, S2 gate"),
    ("02_thresholds", "memory threshold — is the noise model sane?"),
    ("03_erasure", "atom loss + dynamic reweighting  → figure 1"),
    ("04_oracle", "P(BP+OSD right | matching wrong) — the go/no-go number"),
    ("05_generate_dataset", "shots, features, labels → data/*.parquet"),
    ("06_train_triage", "LightGBM + MLP vs the baseline ladder"),
    ("07_pareto", "per-shot latency, Pareto frontier"),
    ("08_figures", "figures 2, 3, 4"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fewer shots; same conclusions")
    ap.add_argument("--from", dest="start", default=None, help="resume at this stage")
    args = ap.parse_args()

    env = dict(os.environ)
    # Latency numbers are only comparable if nothing fans out behind our back.
    env["OMP_NUM_THREADS"] = "1"
    if args.quick:
        env["ERASURE_TRIAGE_QUICK"] = "1"

    stages = STAGES
    if args.start:
        names = [s[0] for s in STAGES]
        stages = STAGES[names.index(args.start):]

    t_all = time.perf_counter()
    for name, what in stages:
        print(f"\n{'=' * 78}\n>>> {name}  —  {what}\n{'=' * 78}", flush=True)
        t0 = time.perf_counter()
        r = subprocess.run([PY, "-u", str(ROOT / "scripts" / f"{name}.py")], env=env)
        if r.returncode != 0:
            print(f"\n!!! {name} failed (exit {r.returncode})")
            return r.returncode
        print(f"--- {name} done in {time.perf_counter() - t0:.0f}s", flush=True)

    print(f"\nall stages done in {(time.perf_counter() - t_all) / 60:.1f} min")
    print("figures:")
    for f in sorted((ROOT / "figs").glob("*.png")):
        print(f"  {f.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
