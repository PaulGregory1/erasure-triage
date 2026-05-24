"""Shot generation: chunked sampling, parallel sweeps, class-imbalance handling."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from .circuits import LogicalProgram
from .dem_utils import build_problem
from .decoders import BpOsdDecoder, ErasureMatchingDecoder, MatchingDecoder
from .evaluate import Rate
from .noise import NoiseModel

# Keep every worker single-threaded: we parallelise over tasks, not inside them,
# and BLAS/OpenMP fan-out inside a worker would make the latency numbers meaningless.
os.environ.setdefault("OMP_NUM_THREADS", "1")

CHUNK = 200_000


@dataclass(frozen=True)
class SweepPoint:
    program: LogicalProgram
    d: int
    noise: NoiseModel
    variant: str  # "blind" | "pauli" | "reweight"
    shots: int
    seed: int


def _problem_for(pt: SweepPoint):
    decoder_noise = pt.noise.without_loss() if pt.variant == "blind" else None
    return build_problem(pt.program, pt.d, pt.noise, decoder_noise=decoder_noise)


def run_point(pt: SweepPoint) -> dict:
    """One (circuit, d, noise, decoder-variant) cell of a sweep."""
    pr = _problem_for(pt)
    dec = (
        ErasureMatchingDecoder(pr, with_gap=False)
        if pt.variant == "reweight"
        else MatchingDecoder(pr, with_gap=False)
    )
    k = n = 0
    rng_seed = pt.seed
    remaining = pt.shots
    while remaining > 0:
        m = min(CHUNK, remaining)
        her, syn, obs = pr.sample(m, seed=rng_seed)
        rng_seed += 1
        remaining -= m
        if pt.variant == "reweight":
            pred = dec.decode_grouped(syn, her)
        else:
            pred = dec.decode_batch(syn)
        k += int((pred != obs).any(axis=1).sum())
        n += m
    r = Rate(k, n)
    return {
        "circuit": pt.program.name, "d": pt.d, "variant": pt.variant,
        "p_2q": pt.noise.p_2q, "p_loss": pt.noise.p_loss,
        "n_cx": pr.program.n_transversal_cx,
        "shots": n, "fails": k, "ler": r.p, "ler_err": r.err,
        "n_undecomposable": pr.n_undecomposable, "n_dets": pr.n_dets,
    }


def run_sweep(points: list[SweepPoint], workers: int | None = None) -> list[dict]:
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(run_point, points))


# --------------------------------------------------------------------------------------
def train_noise(target: NoiseModel, factor: float) -> NoiseModel:
    """§7.3: train where failures are common, evaluate where they matter.

    At p_target the matcher fails ~1e-3 of the time, so 1e5 positives would need
    1e8 shots. Scaling every rate by 2-3x lifts the failure rate to 1-5%, which
    makes the dataset affordable; whether the classifier *transfers* back down to
    p_target is then an experimental question, and a negative answer is a result.
    """
    return target.scaled(factor)


# --------------------------------------------------------------------------------------
# Dataset generation (Phase 5)
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class DatasetChunk:
    program: LogicalProgram
    d: int
    noise: NoiseModel
    shots: int
    seed: int
    with_bposd: bool
    with_edges: bool = True


def _run_chunk(c: DatasetChunk) -> dict:
    from .features import FeatureExtractor, labels_from

    pr = build_problem(c.program, c.d, c.noise)
    md = ErasureMatchingDecoder(pr, with_gap=True)
    fx = FeatureExtractor(pr, md)

    her, syn, obs = pr.sample(c.shots, seed=c.seed)
    # The deployed fast decoder is matching *with* erasure reweighting -- that is the
    # cheap leg of the triage, so the features and the label describe the same decoder.
    X, pred, names = fx.extract(syn, her, with_gap=True, with_edges=c.with_edges)
    y = labels_from(pred, obs)

    out = {"X": X, "y": y, "names": names}
    if c.with_bposd:
        bp = BpOsdDecoder(pr)
        pb = bp.decode_batch(syn)
        out["y_bposd"] = labels_from(pb, obs)
    return out


def generate_dataset(
    program: LogicalProgram,
    d: int,
    noise: NoiseModel,
    shots: int,
    seed: int = 0,
    with_bposd: bool = False,
    workers: int | None = None,
    chunk: int = 50_000,
) -> dict:
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    chunks = []
    left, s = shots, seed
    while left > 0:
        m = min(chunk, left)
        chunks.append(DatasetChunk(program, d, noise, m, s, with_bposd))
        left -= m
        s += 1
    with ProcessPoolExecutor(max_workers=workers) as ex:
        parts = list(ex.map(_run_chunk, chunks))
    out = {
        "X": np.concatenate([p["X"] for p in parts]),
        "y": np.concatenate([p["y"] for p in parts]),
        "names": parts[0]["names"],
    }
    if with_bposd:
        out["y_bposd"] = np.concatenate([p["y_bposd"] for p in parts])
    return out
