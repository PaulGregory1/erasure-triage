"""The decoder zoo.

* :class:`MatchingDecoder` -- MWPM on the *decomposed* DEM. Fast, and blind to
  every correlation that the decomposition threw away. This is the "cheap" leg.
* :class:`ErasureMatchingDecoder` -- same graph, but each shot's heralded atom
  losses zero the weight of the edges they touch. This is §5's dynamic reweighting.
* :class:`BpOsdDecoder` -- BP + ordered-statistics decoding on the *undecomposed*
  hypergraph. Uses the correlations. This is the "slow" leg.
* :class:`TriageDecoder` -- run matching, decide, escalate.

The complementary gap comes from a second matching graph per observable in which
the observable is promoted to a detector node: force it to 0, force it to 1, take
|w0 - w1|. It costs one extra matching per observable, and we measure that cost
rather than assume it away.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pymatching
import stim
from ldpc.bposd_decoder import BpOsdDecoder as _LdpcBpOsd

from .dem_utils import DEM_KW, DecodingProblem, Edge, dem_matrices, observable_as_detector


# --------------------------------------------------------------------------------------
@dataclass
class ShotResult:
    prediction: np.ndarray  # (n_obs,) bool
    weight: float
    gaps: np.ndarray  # (n_obs,) complementary gap per observable
    n_hyper_edges: int  # matched edges that are halves of a non-graphlike mechanism
    seconds: float


class MatchingDecoder:
    """MWPM via pymatching's sparse blossom, on the decomposed DEM."""

    name = "matching"

    def __init__(self, problem: DecodingProblem, with_gap: bool = True) -> None:
        self.problem = problem
        self.matching = pymatching.Matching.from_detector_error_model(problem.dem)
        self.n_obs = problem.n_obs
        self.n_dets = problem.n_dets
        self.with_gap = with_gap

        # One extra graph per observable, with that observable promoted to a detector.
        self.gap_matchers: list[pymatching.Matching] = []
        if with_gap:
            for k in range(self.n_obs):
                circ = observable_as_detector(problem.dec.circuit, k)
                dem_k = circ.detector_error_model(decompose_errors=True, **DEM_KW)
                self.gap_matchers.append(pymatching.Matching.from_detector_error_model(dem_k))

        self._hyper_edges = _hyperedge_half_set(problem)

    # -- batch (fast path, for labels and error rates) ----------------------------------
    def decode_batch(self, syndrome: np.ndarray) -> np.ndarray:
        return self.matching.decode_batch(syndrome).astype(bool).reshape(len(syndrome), self.n_obs)

    # -- per shot (for latency and features) -------------------------------------------
    def decode_shot(self, s: np.ndarray) -> ShotResult:
        t0 = time.perf_counter_ns()
        pred, w = self.matching.decode(s, return_weight=True)
        gaps = np.zeros(self.n_obs)
        if self.with_gap:
            for k, mk in enumerate(self.gap_matchers):
                forced = np.empty(self.n_dets + 1, dtype=np.uint8)
                forced[: self.n_dets] = s
                forced[self.n_dets] = 1 - int(pred[k])  # force the *other* logical class
                _, wc = mk.decode(forced, return_weight=True)
                gaps[k] = wc - w
        t1 = time.perf_counter_ns()
        return ShotResult(
            prediction=pred.astype(bool),
            weight=float(w),
            gaps=gaps,
            n_hyper_edges=0,
            seconds=(t1 - t0) * 1e-9,
        )

    def matched_edges(self, s: np.ndarray) -> np.ndarray:
        return self.matching.decode_to_edges_array(s)

    def n_hyper_edges_used(self, s: np.ndarray) -> int:
        """How much of this shot's solution rests on edges that were once hyperedges."""
        if not self._hyper_edges:
            return 0
        edges = self.matching.decode_to_edges_array(s)
        return sum(1 for u, v in edges if (int(u), int(v)) in self._hyper_edges)


def _hyperedge_half_set(problem: DecodingProblem) -> set[Edge]:
    """Edges that appear as a component of a DEM error with >2 symptoms.

    These are exactly the places where the decomposition discarded a correlation:
    matching sees two independent edges where the physics says one mechanism.
    """
    from .dem_utils import _components

    out: set[Edge] = set()
    for inst in problem.dem.flattened():
        if inst.type != "error":
            continue
        comps = _components(inst)
        if len(comps) < 2 and all(len(c) <= 2 for c in comps):
            continue
        for c in comps:
            if len(c) == 2:
                out.add((min(c), max(c)))
                out.add((max(c), min(c)))
            elif len(c) == 1:
                out.add((c[0], -1))
    return out


# --------------------------------------------------------------------------------------
class ErasureMatchingDecoder(MatchingDecoder):
    """Matching + per-shot dynamic reweighting on heralded atom loss.

    An erased atom is returned in the maximally mixed state: its Pauli frame is a
    coin flip. In matching units that is probability 1/2, i.e. **weight 0** -- the
    decoder may route a correction through that qubit for free. Setting the weight
    of the touched edges to 0 (rather than leaving them at their nominal weight) is
    the whole difference between baseline (b) and baseline (c).

    Cost control: erasures are rare, so most shots erase nothing and reuse one
    untouched matcher. Shots that *do* erase pay only for the edges they touch --
    we mutate those edges in place and restore them afterwards, instead of
    rebuilding the graph.
    """

    name = "erasure_matching"

    def __init__(self, problem: DecodingProblem, with_gap: bool = True) -> None:
        super().__init__(problem, with_gap=with_gap)
        # Graph 0 is the base matcher; graph k+1 is the complementary-gap matcher for
        # observable k. All of them must be reweighted, or the gap would be measured on
        # a graph that disagrees with the one that produced the prediction.
        self.graphs = [self.matching] + list(self.gap_matchers)
        self.tables = [problem.herald_edges] + list(problem.herald_edges_gap[: len(self.gap_matchers)])
        self._saved: list[dict[Edge, tuple]] = [{} for _ in self.graphs]
        for gi, (g, table) in enumerate(zip(self.graphs, self.tables)):
            for edges in table:
                for e in edges:
                    if e in self._saved[gi]:
                        continue
                    data = _edge_data(g, e)
                    if data is not None:
                        self._saved[gi][e] = data

    def _set_weight(self, gi: int, edge: Edge, weight: float, prob: float) -> None:
        u, v = edge
        fault_ids, _, _ = self._saved[gi][edge]
        g = self.graphs[gi]
        if v == -1:
            g.add_boundary_edge(
                u, fault_ids=fault_ids, weight=weight,
                error_probability=prob, merge_strategy="replace",
            )
        else:
            g.add_edge(
                u, v, fault_ids=fault_ids, weight=weight,
                error_probability=prob, merge_strategy="replace",
            )

    def _erased_edges(self, gi: int, heralds: np.ndarray) -> list[Edge]:
        table, saved = self.tables[gi], self._saved[gi]
        hit: set[Edge] = set()
        for h in np.flatnonzero(heralds):
            for e in table[h]:
                if e in saved:
                    hit.add(e)
        return sorted(hit)

    def _apply(self, heralds: np.ndarray) -> list[list[Edge]]:
        applied = []
        for gi in range(len(self.graphs)):
            edges = self._erased_edges(gi, heralds)
            for e in edges:
                self._set_weight(gi, e, 0.0, 0.5)  # p = 1/2 -> weight 0
            applied.append(edges)
        return applied

    def _restore(self, applied: list[list[Edge]]) -> None:
        for gi, edges in enumerate(applied):
            for e in edges:
                _, w0, p0 = self._saved[gi][e]
                self._set_weight(gi, e, w0, p0)

    # ----------------------------------------------------------------------------------
    def decode_grouped(self, syndrome: np.ndarray, heralds: np.ndarray) -> np.ndarray:
        pred, _, _ = self.decode_grouped_full(syndrome, heralds, with_gap=False)
        return pred

    def decode_grouped_full(
        self, syndrome: np.ndarray, heralds: np.ndarray, with_gap: bool = True
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decode a batch, grouping shots by identical erasure pattern.

        Returns (prediction, matching weight, complementary gaps).

        The zero-erasure group is the overwhelming majority and runs on the untouched
        graphs. Every other pattern pays one graph mutation, shared across all the shots
        that have that pattern -- which is why the cost stays bounded even though the
        reweighting is nominally per shot.
        """
        n = len(syndrome)
        syn = np.ascontiguousarray(syndrome.astype(np.uint8))
        pred = np.zeros((n, self.n_obs), dtype=bool)
        weight = np.zeros(n, dtype=np.float32)
        gaps = np.zeros((n, self.n_obs), dtype=np.float32)

        def run(idx: np.ndarray) -> None:
            p, w = self.matching.decode_batch(syn[idx], return_weights=True)
            p = p.astype(bool).reshape(len(idx), self.n_obs)
            pred[idx] = p
            weight[idx] = w
            if with_gap and self.gap_matchers:
                forced = np.zeros((len(idx), self.n_dets + 1), dtype=np.uint8)
                forced[:, : self.n_dets] = syn[idx]
                for k, mk in enumerate(self.gap_matchers):
                    forced[:, self.n_dets] = 1 - p[:, k].astype(np.uint8)
                    _, wk = mk.decode_batch(forced, return_weights=True)
                    gaps[idx, k] = wk - w

        if heralds.size == 0 or not heralds.any():
            run(np.arange(n))
            return pred, weight, gaps

        any_loss = heralds.any(axis=1)
        clean = np.flatnonzero(~any_loss)
        if clean.size:
            run(clean)

        groups: dict[bytes, list[int]] = {}
        for i in np.flatnonzero(any_loss):
            groups.setdefault(heralds[i].tobytes(), []).append(int(i))
        for idx_list in groups.values():
            idx = np.array(idx_list)
            applied = self._apply(heralds[idx[0]])
            try:
                run(idx)
            finally:
                self._restore(applied)
        return pred, weight, gaps


def _edge_data(g: pymatching.Matching, edge: Edge):
    u, v = edge
    try:
        data = g.get_boundary_edge_data(u) if v == -1 else g.get_edge_data(u, v)
    except ValueError:
        return None  # this component has no single edge in the decomposed graph
    return (
        data.get("fault_ids", set()),
        data.get("weight", 1.0),
        data.get("error_probability", -1.0),
    )


# --------------------------------------------------------------------------------------
class BpOsdDecoder:
    """BP + OSD on the undecomposed hypergraph. Slow, and uses the correlations."""

    name = "bp_osd"

    def __init__(
        self,
        problem: DecodingProblem,
        max_iter: int = 30,
        osd_order: int = 10,
        osd_method: str = "osd_cs",
    ) -> None:
        self.problem = problem
        mats = dem_matrices(problem.dem_raw)
        self.H = mats.check_matrix
        self.L = mats.observables_matrix
        self.priors = mats.priors
        self.n_obs = problem.n_obs
        self.dec = _LdpcBpOsd(
            self.H,
            error_channel=list(self.priors),
            max_iter=max_iter,
            bp_method="ms",
            ms_scaling_factor=0.625,
            schedule="parallel",
            osd_method=osd_method,
            osd_order=osd_order,
        )

    def decode_shot(self, s: np.ndarray) -> ShotResult:
        t0 = time.perf_counter_ns()
        corr = self.dec.decode(s.astype(np.uint8))
        pred = (self.L @ corr) % 2
        t1 = time.perf_counter_ns()
        return ShotResult(
            prediction=np.asarray(pred).ravel().astype(bool),
            weight=float("nan"), gaps=np.zeros(self.n_obs),
            n_hyper_edges=0, seconds=(t1 - t0) * 1e-9,
        )

    def decode_batch(self, syndrome: np.ndarray) -> np.ndarray:
        out = np.zeros((len(syndrome), self.n_obs), dtype=bool)
        for i, s in enumerate(syndrome):
            out[i] = self.decode_shot(s).prediction
        return out


# --------------------------------------------------------------------------------------
class IndependentMatchingDecoder:
    """Straw man: decode each patch's detectors on its own sub-graph.

    Errors that cross a transversal CNOT flip detectors in two patches at once.
    A per-patch decoder cannot represent that; it pays for the two halves twice.
    This is the baseline that shows why joint (correlated) decoding is needed at all
    once you go to r = 1 round per gate.
    """

    name = "independent_matching"

    def __init__(self, problem: DecodingProblem) -> None:
        self.problem = problem
        self.n_obs = problem.n_obs
        self.n_dets = problem.n_dets
        self.patch_of_det = problem.det_patch
        self.matchers: dict[int, pymatching.Matching] = {}
        self.det_idx: dict[int, np.ndarray] = {}
        for p in sorted(set(problem.det_patch.tolist())):
            sub, idx = _restrict_dem_to_patch(problem, p)
            self.det_idx[p] = idx
            self.matchers[p] = pymatching.Matching.from_detector_error_model(sub)

    def decode_batch(self, syndrome: np.ndarray) -> np.ndarray:
        out = np.zeros((len(syndrome), self.n_obs), dtype=bool)
        for p, m in self.matchers.items():
            pred = m.decode_batch(syndrome[:, self.det_idx[p]])
            out ^= pred.astype(bool).reshape(len(syndrome), self.n_obs)
        return out


def _restrict_dem_to_patch(problem: DecodingProblem, patch: int):
    """Sub-DEM whose detectors live on one patch; cross-patch errors get truncated."""
    keep = np.flatnonzero(problem.det_patch == patch)
    remap = {int(g): i for i, g in enumerate(keep)}
    out = stim.DetectorErrorModel()
    for inst in problem.dem.flattened():
        if inst.type != "error":
            continue
        p = inst.args_copy()[0]
        targets = []
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                if t.val in remap:
                    targets.append(stim.target_relative_detector_id(remap[t.val]))
            elif t.is_logical_observable_id():
                targets.append(t)
            elif t.is_separator():
                targets.append(t)
        # drop leading/duplicate separators produced by the truncation
        cleaned: list = []
        for t in targets:
            if t.is_separator() and (not cleaned or cleaned[-1].is_separator()):
                continue
            cleaned.append(t)
        while cleaned and cleaned[-1].is_separator():
            cleaned.pop()
        if not any(t.is_relative_detector_id() for t in cleaned):
            continue
        out.append("error", p, cleaned)
    return out, keep


# --------------------------------------------------------------------------------------
class CorrelatedMatchingDecoder:
    """pymatching's own correlated matching. The obvious objection, answered.

    "Why not just turn on correlated matching instead of paying for BP+OSD?" -- because it
    does not fix the problem. pymatching's correlation handling reweights the matching graph
    using the *decomposed* model, and it flatly refuses a DEM containing an undecomposed
    hyperedge. So the very mechanisms that break MWPM (47 of them at d=5, 203 on the 3-CNOT
    chain) have to be dropped from its model before it will even load.

    Measured: it halves the single-fault failures at d=3 and does nothing at all at d=5. Its
    effective distance is still 2. It is cheap and it is not enough, which is exactly why
    escalation to a hypergraph decoder is the thing being studied.
    """

    name = "correlated_matching"

    def __init__(self, problem: DecodingProblem) -> None:
        graphlike, self.n_dropped = _graphlike_only(problem.dem)
        self.matching = pymatching.Matching.from_detector_error_model(
            graphlike, enable_correlations=True
        )
        self.n_obs = problem.n_obs

    def decode_batch(self, syndrome: np.ndarray) -> np.ndarray:
        s = np.ascontiguousarray(syndrome.astype(np.uint8))
        pred = self.matching.decode_batch(s, enable_correlations=True)
        return pred.astype(bool).reshape(len(s), self.n_obs)


def _graphlike_only(dem: stim.DetectorErrorModel) -> tuple[stim.DetectorErrorModel, int]:
    """The DEM restricted to what a matching graph can physically hold.

    Every component must have 1 or 2 detectors. Anything else -- a surviving hyperedge, or a
    purely undetectable component -- is dropped, and we count how many, because that count is
    the size of the blind spot.
    """
    out = stim.DetectorErrorModel()
    dropped = 0
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        comps: list[list] = [[]]
        for t in inst.targets_copy():
            if t.is_separator():
                comps.append([])
            else:
                comps[-1].append(t)
        degrees = [sum(1 for t in c if t.is_relative_detector_id()) for c in comps]
        if any(k > 2 or k == 0 for k in degrees):
            dropped += 1
            continue
        out.append(inst)
    return out, dropped


class BpOnlyDecoder:
    """BP on the hypergraph, without OSD. A control, and a negative result.

    If BP+OSD's advantage came purely from *seeing* the hyperedges, BP alone would already
    fix the single-fault failures. It does not -- it is measurably *worse* than plain MWPM
    (59 vs 16 single-fault failures at d=5), because belief propagation does not converge on
    the highly degenerate loops these circuits create. The OSD post-processing is doing real
    work, and it is what the escalation is paying for.
    """

    name = "bp_only"

    def __init__(self, problem: DecodingProblem, max_iter: int = 60) -> None:
        from ldpc.bp_decoder import BpDecoder

        mats = dem_matrices(problem.dem_raw)
        self.L = mats.observables_matrix
        self.n_obs = problem.n_obs
        self.dec = BpDecoder(
            mats.check_matrix, error_channel=list(mats.priors), max_iter=max_iter,
            bp_method="ms", ms_scaling_factor=0.625,
        )

    def decode_batch(self, syndrome: np.ndarray) -> np.ndarray:
        out = np.zeros((len(syndrome), self.n_obs), dtype=bool)
        for i, s in enumerate(syndrome):
            out[i] = (self.L @ self.dec.decode(s.astype(np.uint8))) % 2
        return out
