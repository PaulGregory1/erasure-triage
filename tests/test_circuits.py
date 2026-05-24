"""Phase-1 validation. Nothing else in this repo is meaningful if these fail.

1. Noiseless circuit fires zero detectors.
2. The circuit distance equals d. A distance below d means the detectors are wrong;
   this is the test that saves the project.
3. The DEM builds. If ``decompose_errors=True`` fails, that is a *signal*, not a bug:
   it means non-decomposable hyperedges, which we count and use as a feature.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import stim

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.circuits import (  # noqa: E402
    cnot_chain,
    compile_program,
    ghz,
    memory,
    transversal_cnot,
)
from src.noise import NoiseModel  # noqa: E402

UNIFORM = NoiseModel.uniform(1e-3, loss_ratio=0.0)

PROGRAMS = [
    ("memory_z", lambda d: memory(d, "Z")),
    ("memory_x", lambda d: memory(d, "X")),
    ("cnot_rd", lambda d: transversal_cnot(d, "Z")),
    ("cnot_r1", lambda d: transversal_cnot(1, "Z")),
    ("cnot_r1_x", lambda d: transversal_cnot(1, "X")),
    ("chain3_r1", lambda d: cnot_chain(1, "Z")),
    ("chain3_rd", lambda d: cnot_chain(d, "Z")),
    ("ghz4_r1", lambda d: ghz(1, 4)),
]


@pytest.mark.parametrize("d", [3, 5])
@pytest.mark.parametrize("name,make", PROGRAMS, ids=[p[0] for p in PROGRAMS])
def test_noiseless_has_no_detections(d, name, make):
    cc = compile_program(make(d), d)
    sample = cc.circuit.compile_detector_sampler().sample(256)
    assert not sample.any(), f"{name} d={d}: noiseless circuit fires detectors"


@pytest.mark.parametrize("d", [3, 5])
@pytest.mark.parametrize("name,make", PROGRAMS, ids=[p[0] for p in PROGRAMS])
def test_distance_equals_d(d, name, make):
    cc = compile_program(make(d), d, UNIFORM)
    err = cc.circuit.search_for_undetectable_logical_errors(
        dont_explore_detection_event_sets_with_size_above=6,
        dont_explore_edges_with_degree_above=4,
        dont_explore_edges_increasing_symptom_degree=False,
        canonicalize_circuit_errors=True,
    )
    assert len(err) == d, f"{name} d={d}: found undetectable logical error of weight {len(err)}"


@pytest.mark.parametrize("d", [3, 5])
@pytest.mark.parametrize("name,make", PROGRAMS, ids=[p[0] for p in PROGRAMS])
def test_dem_builds(d, name, make):
    cc = compile_program(make(d), d, UNIFORM)
    dem = cc.circuit.detector_error_model(decompose_errors=False)
    assert dem.num_detectors == cc.n_detectors
    assert dem.num_observables == cc.n_observables


def test_heralded_erasure_produces_herald_detectors():
    noise = NoiseModel.uniform(1e-3, loss_ratio=1.0)
    cc = compile_program(transversal_cnot(1, "Z"), 3, noise)
    assert cc.herald_dets, "no herald detectors emitted"
    # Heralds must never fire without loss, and must be deterministic-zero when noiseless.
    clean = compile_program(transversal_cnot(1, "Z"), 3)
    assert not clean.herald_dets, "noiseless model should emit no HERALDED_ERASE"
    sample = cc.circuit.compile_detector_sampler().sample(4096)
    herald_cols = sample[:, cc.herald_dets]
    assert herald_cols.any(), "heralds never fire despite p_loss > 0"


def test_erasure_does_not_break_distance():
    noise = NoiseModel.uniform(1e-3, loss_ratio=1.0)
    for d in (3, 5):
        cc = compile_program(transversal_cnot(1, "Z"), d, noise)
        err = cc.circuit.search_for_undetectable_logical_errors(
            dont_explore_detection_event_sets_with_size_above=6,
            dont_explore_edges_with_degree_above=4,
            dont_explore_edges_increasing_symptom_degree=False,
            canonicalize_circuit_errors=True,
        )
        assert len(err) == d, f"d={d}: erasure model gives distance {len(err)}"


def test_matches_stim_reference_memory():
    """Our hand-built memory_z must be equivalent to stim's generated circuit.

    Not literally identical (noise placement differs) but same detector count,
    same observable count, same distance.
    """
    for d in (3, 5):
        ours = compile_program(memory(d, "Z"), d, UNIFORM)
        ref = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            distance=d,
            rounds=d,
            after_clifford_depolarization=1e-3,
        )
        assert ours.n_detectors == ref.num_detectors, (
            f"d={d}: {ours.n_detectors} detectors vs stim's {ref.num_detectors}"
        )
        assert ours.n_observables == ref.num_observables


def test_matching_effective_distance_collapses_on_transversal_cnot():
    """The project's central claim, as a regression test.

    MWPM on the decomposed graph must handle every single fault in a memory circuit,
    and must FAIL on at least one single fault once a transversal CNOT is present.
    BP+OSD, which keeps the hyperedges, must handle them all. If this ever stops being
    true, either the circuits changed or a decoder got fixed -- and the whole framing
    of the repo needs revisiting.
    """
    from src.decoders import BpOsdDecoder, MatchingDecoder
    from src.dem_utils import build_problem
    from src.evaluate import single_fault_failures

    noise = NoiseModel.uniform(1e-3, loss_ratio=0.0)

    mem = build_problem(memory(5, "Z"), 5, noise)
    m = single_fault_failures(mem, MatchingDecoder(mem, with_gap=False))
    assert m["n_wrong"] == 0, "MWPM should handle every single fault in a memory circuit"

    cx = build_problem(transversal_cnot(1, "Z"), 5, noise)
    m = single_fault_failures(cx, MatchingDecoder(cx, with_gap=False))
    b = single_fault_failures(cx, BpOsdDecoder(cx))
    assert m["n_wrong"] > 0, "MWPM is expected to be fooled by single faults across a CNOT"
    assert m["n_wrong"] == m["n_wrong_cross_patch"], (
        "every fault that fools MWPM should be one that crossed the gate into both patches"
    )
    assert b["n_wrong"] == 0, "BP+OSD keeps the hyperedges and should not be fooled"
