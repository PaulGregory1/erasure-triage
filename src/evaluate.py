"""Error bars, latency measurement, thresholds, Pareto fronts.

Latency discipline (§9.2): ``OMP_NUM_THREADS=1``, warm-up discarded, per-shot
``perf_counter_ns``, median and p99 over >= 10^4 shots, one documented machine.
A mean latency taken over a vectorised batch measures throughput, not latency, and
a decoder with a fine mean and a catastrophic p99 builds a backlog and is useless.
So we report the tail.
"""

from __future__ import annotations

import platform
import time
from dataclasses import dataclass

import numpy as np

MACHINE = f"{platform.machine()} / {platform.system()} {platform.release()} / OMP_NUM_THREADS=1"


# --------------------------------------------------------------------------------------
@dataclass
class Rate:
    """A binomial rate with a standard error."""

    k: int
    n: int

    @property
    def p(self) -> float:
        return self.k / self.n if self.n else float("nan")

    @property
    def err(self) -> float:
        if not self.n:
            return float("nan")
        p = self.p
        # Use a floor so a zero-failure bin still plots a meaningful bar.
        return float(np.sqrt(max(p * (1 - p), 1.0 / self.n) / self.n))

    def __repr__(self) -> str:
        return f"{self.p:.3e} ± {self.err:.1e} ({self.k}/{self.n})"


def logical_error(pred: np.ndarray, obs: np.ndarray) -> Rate:
    """A shot fails if *any* logical observable is mispredicted."""
    fails = (pred.astype(bool) != obs.astype(bool)).any(axis=1)
    return Rate(int(fails.sum()), len(fails))


def per_gate_error(p_shot: float | Rate, n_cx: int) -> float:
    """Logical error per transversal gate -- the metric the field actually quotes.

    Converts a per-shot failure probability into a per-gate one under the usual
    independence approximation: 1 - (1 - p_shot)^(1/n_gates).
    """
    p = p_shot.p if isinstance(p_shot, Rate) else float(p_shot)
    if n_cx <= 0:
        return p
    return float(1.0 - (1.0 - p) ** (1.0 / n_cx))


def bootstrap_ci(fails: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(fails)
    means = fails[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# --------------------------------------------------------------------------------------
@dataclass
class Latency:
    """Per-shot decode latency, in microseconds."""

    samples: np.ndarray

    @property
    def median(self) -> float:
        return float(np.median(self.samples))

    @property
    def mean(self) -> float:
        return float(np.mean(self.samples))

    @property
    def p99(self) -> float:
        return float(np.percentile(self.samples, 99))

    @property
    def p999(self) -> float:
        return float(np.percentile(self.samples, 99.9))

    def summary(self) -> dict:
        return {
            "mean_us": self.mean, "median_us": self.median,
            "p99_us": self.p99, "p999_us": self.p999,
            "n": len(self.samples), "machine": MACHINE,
        }


def time_per_shot(fn, inputs, warmup: int = 50) -> Latency:
    """Time ``fn`` once per shot. Warm-up is run and thrown away."""
    for i in range(min(warmup, len(inputs))):
        fn(inputs[i])
    out = np.empty(len(inputs))
    for i, x in enumerate(inputs):
        t0 = time.perf_counter_ns()
        fn(x)
        out[i] = (time.perf_counter_ns() - t0) * 1e-3  # microseconds
    return Latency(out)


# --------------------------------------------------------------------------------------
def pareto_front(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Indices on the lower-left Pareto front (minimise both x and y)."""
    order = np.argsort(x)
    best = np.inf
    keep = []
    for i in order:
        if y[i] < best:
            keep.append(int(i))
            best = y[i]
    return np.array(keep)


def escalation_latency(
    p_escalate: float, t_match: float, t_bposd: float, t_feature: float = 0.0
) -> float:
    """Mean latency of a triage decoder: always match, sometimes escalate."""
    return t_match + t_feature + p_escalate * t_bposd


def escalation_latency_p99(
    escalated: np.ndarray, t_match: np.ndarray, t_bposd: np.ndarray, t_feature: float = 0.0
) -> float:
    """p99 of the *realised* per-shot latency of a triage decoder.

    A shot that escalates pays matching *and* BP+OSD; the tail is what matters,
    because a real-time decoder's backlog is built by the tail, not the mean.
    """
    total = t_match + t_feature + escalated.astype(float) * t_bposd
    return float(np.percentile(total, 99))


# --------------------------------------------------------------------------------------
def single_fault_failures(problem, decoder) -> dict:
    """Feed every single-fault error mechanism to a decoder and see which it gets wrong.

    This is a *deterministic* measurement of a decoder's effective distance -- no Monte
    Carlo, no error bars. If a decoder mis-corrects even one single fault, its effective
    distance is 2, whatever the code distance is.

    Memory: MWPM gets all of them right. Transversal CNOT: it does not, because one
    physical fault before the gate propagates into both patches, and the decomposed
    matching graph can only express that as two independent errors of equal total weight.
    Picking between the two explanations is a coin flip, and half the time it flips the
    logical class. BP+OSD keeps the hyperedge and is not fooled.
    """
    syn, obs = [], []
    for inst in problem.dem_raw.flattened():
        if inst.type != "error":
            continue
        s = np.zeros(problem.n_dets, dtype=np.uint8)
        o = np.zeros(problem.n_obs, dtype=bool)
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                s[t.val] = 1
            elif t.is_logical_observable_id():
                o[t.val] ^= True
        syn.append(s)
        obs.append(o)
    S = np.array(syn)
    O = np.array(obs)
    pred = decoder.decode_batch(S)
    wrong = (pred != O).any(axis=1)
    idx = np.flatnonzero(wrong)

    cross = 0
    for i in idx:
        lit = np.flatnonzero(S[i])
        if len(set(problem.det_patch[lit].tolist())) > 1:
            cross += 1
    return {
        "n_faults": int(len(S)),
        "n_wrong": int(wrong.sum()),
        "n_wrong_cross_patch": int(cross),
        "effective_distance_at_most_2": bool(wrong.any()),
    }
