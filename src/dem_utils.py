"""DEM plumbing: matrices, hyperedge accounting, and the erasure -> edge table.

Two circuits are compiled from every program, with *identical* noise:

* ``full``  -- herald detectors declared. This is the one we **sample**.
* ``dec``   -- herald detectors suppressed. This is the one we **decode**.

Detectors are emitted in the same order in both, so the columns of ``full``'s
detector array restricted to ``syndrome_dets`` are exactly ``dec``'s detectors.
That alignment is what lets us read the heralds in the clear while decoding on a
graph that has no herald nodes in it.

Suppressing the herald detector is not a hack -- it *is* baseline (b) of §5.3.
An erasure with an unobserved herald is just extra Pauli noise on that qubit,
and the DEM says so. Baseline (c) then reuses the same graph but sets the
weight of every edge touched by an erased qubit to 0 (probability 1/2), which is
what "we know nothing about this qubit's Pauli frame" means in matching units.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import stim
from beliefmatching import detector_error_model_to_check_matrices

from .circuits import CompiledCircuit, LogicalProgram, compile_program
from .noise import NoiseModel

DEM_KW = dict(approximate_disjoint_errors=True, ignore_decomposition_failures=True)

Edge = tuple[int, int]  # (node, node); -1 means the boundary


@dataclass
class DecodingProblem:
    """Everything a decoder needs for one (program, d, noise) triple."""

    program: LogicalProgram
    d: int
    noise: NoiseModel

    full: CompiledCircuit  # sampled from; has herald detectors
    dec: CompiledCircuit  # decoded on; no herald detectors

    dem: stim.DetectorErrorModel  # decomposed, from `dec`
    dem_raw: stim.DetectorErrorModel  # undecomposed, from `dec` (for BP+OSD)

    herald_dets: np.ndarray  # column indices into the sampled detector array
    syndrome_dets: np.ndarray  # column indices into the sampled detector array
    herald_edges: list[list[Edge]]  # herald i -> edges of the base matching graph it erases
    herald_edges_gap: list[list[list[Edge]]]  # [observable k][herald i] -> edges of gap graph k
    herald_qubit: list[int]
    herald_round: list[int]
    herald_patch: list[int]

    n_undecomposable: int
    n_errors: int
    det_kind: np.ndarray  # per syndrome detector: 0 = X, 1 = Z
    det_round: np.ndarray
    det_patch: np.ndarray
    det_xy: np.ndarray  # (n_syndrome_dets, 2)

    @property
    def n_dets(self) -> int:
        return len(self.syndrome_dets)

    @property
    def n_obs(self) -> int:
        return self.dec.n_observables

    @property
    def n_cx(self) -> int:
        return self.program.n_transversal_cx

    def sampler(self, seed: int | None = None):
        return self.full.circuit.compile_detector_sampler(seed=seed)

    def sample(self, shots: int, seed: int | None = None):
        """Returns (heralds, syndrome, observables) as bool arrays."""
        det, obs = self.sampler(seed).sample(shots, separate_observables=True)
        return det[:, self.herald_dets], det[:, self.syndrome_dets], obs


def _components(inst: stim.DemInstruction) -> list[list[int]]:
    """Split a decomposed DEM error into its graphlike components (detector ids only)."""
    comps: list[list[int]] = [[]]
    for t in inst.targets_copy():
        if t.is_separator():
            comps.append([])
        elif t.is_relative_detector_id():
            comps[-1].append(t.val)
    return [c for c in comps if c]


def count_undecomposable(dem: stim.DetectorErrorModel) -> tuple[int, int]:
    """(# errors with a component of degree > 2, # errors). Zero for plain memory."""
    bad = tot = 0
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        tot += 1
        if any(len(c) > 2 for c in _components(inst)):
            bad += 1
    return bad, tot


def build_problem(
    program: LogicalProgram,
    d: int,
    noise: NoiseModel,
    decoder_noise: NoiseModel | None = None,
) -> DecodingProblem:
    """Compile the sampling and decoding circuits and wire up the erasure table.

    ``decoder_noise`` lets the decoder's DEM disagree with reality on purpose --
    that is baseline (a), "erasures ignored": pass ``noise.without_loss()``.
    """
    full = compile_program(program, d, noise, herald_detectors=True)
    dec_noise = noise if decoder_noise is None else decoder_noise
    dec = compile_program(program, d, dec_noise, herald_detectors=False)

    dem = dec.circuit.detector_error_model(decompose_errors=True, **DEM_KW)
    dem_raw = dec.circuit.detector_error_model(decompose_errors=False, **DEM_KW)
    n_und, n_err = count_undecomposable(dem)

    syndrome_dets = np.array(full.syndrome_dets, dtype=np.int64)
    herald_dets = np.array(full.herald_dets, dtype=np.int64)
    assert len(syndrome_dets) == dem.num_detectors, (
        f"detector alignment broken: {len(syndrome_dets)} vs {dem.num_detectors}"
    )

    # full detector index -> decoding-graph index
    remap = {int(g): i for i, g in enumerate(syndrome_dets)}

    herald_edges = _herald_edge_table(full.circuit, full.herald_dets, remap)

    # One table per complementary-gap graph. Node n_dets is the promoted observable,
    # which is declared last, so it is detector `full.n_detectors` of the augmented circuit.
    n_dets = len(syndrome_dets)
    herald_edges_gap: list[list[list[Edge]]] = []
    for k in range(dec.n_observables):
        circ_k = observable_as_detector(full.circuit, k)
        remap_k = dict(remap)
        remap_k[full.n_detectors] = n_dets
        herald_edges_gap.append(_herald_edge_table(circ_k, full.herald_dets, remap_k))

    herald_qubit = [h["qubit"] for h in full.herald_info]
    herald_round = [h["round"] for h in full.herald_info]
    herald_patch = [h["patch"] for h in full.herald_info]

    det_kind = np.array([0 if dec.kind_of_det[i] == "X" else 1 for i in range(dem.num_detectors)],
                        dtype=np.int8)
    det_round = np.array([dec.round_of_det[i] for i in range(dem.num_detectors)], dtype=np.int32)
    det_patch = np.array([dec.patch_of_det[i] for i in range(dem.num_detectors)], dtype=np.int32)
    det_xy = np.array([dec.det_coords[i][:2] for i in range(dem.num_detectors)], dtype=np.float32)

    return DecodingProblem(
        program=program, d=d, noise=noise, full=full, dec=dec,
        dem=dem, dem_raw=dem_raw,
        herald_dets=herald_dets, syndrome_dets=syndrome_dets,
        herald_edges=herald_edges, herald_edges_gap=herald_edges_gap,
        herald_qubit=herald_qubit,
        herald_round=herald_round, herald_patch=herald_patch,
        n_undecomposable=n_und, n_errors=n_err,
        det_kind=det_kind, det_round=det_round, det_patch=det_patch, det_xy=det_xy,
    )


def _herald_edge_table(
    circuit: stim.Circuit,
    herald_dets: list[int],
    remap: dict[int, int],
) -> list[list[Edge]]:
    """For each herald detector, the matching-graph edges its erasure can flip.

    Read straight out of the DEM: the errors that flip herald ``h`` *are* the four
    erasure branches (I/X/Y/Z) on that qubit at that round. Strip the herald node
    from each and what remains is exactly the edge set to zero-weight. No coordinate
    parsing, no guessing which qubit owns which edge.

    ``remap`` maps a detector index of ``circuit`` to a node of the target matching
    graph. It is also used as the membership test: anything not in ``remap`` and not a
    herald is not part of that graph. This is what lets the same routine serve the base
    graph and each complementary-gap graph, where the promoted observable is an extra
    node and boundary edges carrying that observable become edges *to* it.
    """
    if not herald_dets:
        return []
    dem = circuit.detector_error_model(decompose_errors=True, **DEM_KW)
    herald_set = set(herald_dets)
    table: dict[int, set[Edge]] = {h: set() for h in herald_dets}

    for inst in dem.flattened():
        if inst.type != "error":
            continue
        comps = _components(inst)
        flat = [dt for c in comps for dt in c]
        hits = [dt for dt in flat if dt in herald_set]
        if len(hits) != 1:
            continue  # not a single-qubit erasure branch
        h = hits[0]
        for c in comps:
            dets = sorted(remap[dt] for dt in c if dt not in herald_set)
            if not dets:
                continue
            if len(dets) == 1:
                table[h].add((dets[0], -1))  # boundary edge
            elif len(dets) == 2:
                table[h].add((dets[0], dets[1]))
            # degree > 2 components have no single matching edge; the BP+OSD path
            # sees them anyway, and matching cannot represent them by construction.
    return [sorted(table[h]) for h in herald_dets]


def dem_matrices(dem: stim.DetectorErrorModel):
    """Check matrix / observables / priors over the full hypergraph, for BP+OSD."""
    return detector_error_model_to_check_matrices(dem, allow_undecomposed_hyperedges=True)


def observable_as_detector(circuit: stim.Circuit, k: int) -> stim.Circuit:
    """Copy of ``circuit`` where observable ``k`` is *also* declared as a detector.

    Forcing that extra detector to 0 and to 1 and taking the two matching weights
    gives the complementary gap. The observable is deterministic in the noiseless
    circuit, so it is a legal detector; and every DEM component carrying an
    observable has exactly one detector, so the promoted node keeps the graph
    graphlike -- MWPM still applies.

    The new detector is the last one declared, so its index is ``num_detectors``
    of the original circuit.
    """
    out = stim.Circuit()
    for inst in circuit.flattened():
        out.append(inst)
        if inst.name == "OBSERVABLE_INCLUDE" and int(inst.gate_args_copy()[0]) == k:
            out.append("DETECTOR", inst.targets_copy(), (0, 0, 0, 2))
    return out
