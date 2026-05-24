"""Compile logical circuits (transversal CNOT networks) on rotated surface code patches.

The whole project turns on getting the detectors right across a transversal CNOT,
so that logic lives in one place: :func:`_pull_back` / :func:`_push_forward`.

Heisenberg propagation through CNOT (c = control, t = target):

    X_c -> X_c X_t      Z_c -> Z_c
    X_t -> X_t          Z_t -> Z_c Z_t

Because the gate is transversal and both patches share a geometry, a stabilizer
of patch ``p`` maps to *the same* stabilizer on a **set** of patches. So the
propagation is a linear map on F_2^(n_patches), and a detector that straddles a
gate layer compares one measurement against the XOR of a *set* of measurements
from the previous round. For a single CNOT A->B this reproduces the table

    X_A(after) = X_A(before) ^ X_B(before)     X_B: unchanged
    Z_B(after) = Z_B(before) ^ Z_A(before)     Z_A: unchanged

which is exactly the three-term detector structure that generates the
non-decomposable hyperedges we exploit later.

CNOT is self-inverse, so pull-back and push-forward use the same elementary map;
only the order differs (pull-back walks the gate list in reverse).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import stim

from .noise import NOISELESS, NoiseModel
from .patches import RotatedSurfaceCode

PatchSet = frozenset[int]


# --------------------------------------------------------------------------------------
# Propagation of a stabilizer/logical support across patches
# --------------------------------------------------------------------------------------
def _conjugate(support: PatchSet, kind: str, control: int, target: int) -> PatchSet:
    """Conjugate an operator of the given ``kind`` supported on ``support`` by one CNOT."""
    s = set(support)
    if kind == "X":
        if control in s:
            s ^= {target}
    elif kind == "Z":
        if target in s:
            s ^= {control}
    else:
        raise ValueError(kind)
    return frozenset(s)


def _pull_back(support: PatchSet, kind: str, gates: list[tuple[int, int]]) -> PatchSet:
    """Rewrite an operator measured *after* ``gates`` as an operator *before* them."""
    for control, target in reversed(gates):
        support = _conjugate(support, kind, control, target)
    return support


def _push_forward(support: PatchSet, kind: str, gates: list[tuple[int, int]]) -> PatchSet:
    """Evolve a stabilizer of the initial state forward through ``gates``."""
    for control, target in gates:
        support = _conjugate(support, kind, control, target)
    return support


# --------------------------------------------------------------------------------------
# Program description
# --------------------------------------------------------------------------------------
@dataclass
class LogicalProgram:
    """A logical circuit over ``n_patches`` surface-code patches.

    ``ops`` is a flat list of either
      * ``("rounds", k)``  -- k rounds of syndrome extraction on every patch
      * ``("cx", [(c, t), ...])`` -- one layer of simultaneous transversal CNOTs
    """

    name: str
    n_patches: int
    init_basis: list[str]
    meas_basis: list[str]
    ops: list[tuple]
    note: str = ""

    def __post_init__(self) -> None:
        assert len(self.init_basis) == self.n_patches
        assert len(self.meas_basis) == self.n_patches
        for op in self.ops:
            if op[0] == "cx":
                seen: set[int] = set()
                for c, t in op[1]:
                    assert c != t, "a patch cannot be its own control"
                    assert c not in seen and t not in seen, "CNOT layer must be disjoint"
                    seen |= {c, t}

    @property
    def gate_layers(self) -> list[list[tuple[int, int]]]:
        return [op[1] for op in self.ops if op[0] == "cx"]

    @property
    def n_transversal_cx(self) -> int:
        return sum(len(layer) for layer in self.gate_layers)


@dataclass
class CompiledCircuit:
    """A stim circuit plus the bookkeeping the decoders need."""

    circuit: stim.Circuit
    program: LogicalProgram
    d: int
    noise: NoiseModel
    n_detectors: int
    herald_dets: list[int]  # detector indices that are atom-loss heralds
    herald_info: list[dict]  # parallel to herald_dets: {patch, data_index, round, qubit}
    syndrome_dets: list[int]  # every non-herald detector index
    det_coords: dict[int, tuple[float, ...]]
    n_observables: int
    round_of_det: dict[int, int]
    kind_of_det: dict[int, str]  # "X", "Z", or "herald"
    patch_of_det: dict[int, int]

    @property
    def n_syndrome_dets(self) -> int:
        return len(self.syndrome_dets)


# --------------------------------------------------------------------------------------
# Compiler
# --------------------------------------------------------------------------------------
class Compiler:
    """Turns a :class:`LogicalProgram` into a stim circuit with correct detectors."""

    def __init__(
        self,
        program: LogicalProgram,
        d: int,
        noise: NoiseModel = NOISELESS,
        herald_detectors: bool = True,
    ) -> None:
        self.p = program
        self.d = d
        self.noise = noise
        self.herald_detectors = herald_detectors
        self.code = RotatedSurfaceCode(d)
        self.n = program.n_patches

        self.per_patch = self.code.n_qubits
        self.patch_dx = 2 * d + 2

        self.circuit = stim.Circuit()
        self.n_meas = 0
        # (patch, stab_index) -> absolute measurement index, for the most recent round
        self.last_round_rec: dict[tuple[int, int], int] = {}
        self.n_rounds_done = 0

        self.herald_dets: list[int] = []
        self.herald_info: list[dict] = []
        self.syndrome_dets: list[int] = []
        self.det_coords: dict[int, tuple[float, ...]] = {}
        self.round_of_det: dict[int, int] = {}
        self.kind_of_det: dict[int, str] = {}
        self.patch_of_det: dict[int, int] = {}
        self.n_det = 0

        # Gates seen since the previous syndrome round (chronological).
        self.pending_gates: list[tuple[int, int]] = []
        # Every gate applied so far (chronological), for observable propagation.
        self.all_gates: list[tuple[int, int]] = []

    # -- qubit indexing ----------------------------------------------------------------
    def data_q(self, patch: int, local: int) -> int:
        return patch * self.per_patch + local

    def anc_q(self, patch: int, stab_index: int) -> int:
        return patch * self.per_patch + self.code.n_data + stab_index

    def _global_coord(self, patch: int, coord: tuple[int, int]) -> tuple[int, int]:
        return (coord[0] + patch * self.patch_dx, coord[1])

    def all_data(self) -> list[int]:
        return [self.data_q(p, i) for p in range(self.n) for i in range(self.code.n_data)]

    def all_anc(self) -> list[int]:
        return [self.anc_q(p, s) for p in range(self.n) for s in range(self.code.n_anc)]

    # -- measurement bookkeeping -------------------------------------------------------
    def _rec(self, absolute_index: int) -> stim.GateTarget:
        return stim.target_rec(absolute_index - self.n_meas)

    def _measure(self, targets: list[int]) -> list[int]:
        idx = list(range(self.n_meas, self.n_meas + len(targets)))
        self.n_meas += len(targets)
        return idx

    def _detector(
        self,
        recs: list[int],
        coord: tuple[float, ...],
        kind: str,
        rnd: int,
        patch: int,
    ) -> int:
        self.circuit.append("DETECTOR", [self._rec(r) for r in recs], coord)
        i = self.n_det
        self.n_det += 1
        self.det_coords[i] = coord
        self.round_of_det[i] = rnd
        self.kind_of_det[i] = kind
        self.patch_of_det[i] = patch
        if kind == "herald":
            self.herald_dets.append(i)
        else:
            self.syndrome_dets.append(i)
        return i

    # -- noise helpers -----------------------------------------------------------------
    def _dep1(self, qubits: list[int], p: float) -> None:
        if p > 0 and qubits:
            self.circuit.append("DEPOLARIZE1", sorted(qubits), p)

    def _dep2(self, pairs: list[int], p: float) -> None:
        if p > 0 and pairs:
            self.circuit.append("DEPOLARIZE2", pairs, p)

    def _erase(self, qubits: list[int], p: float, rnd: int) -> None:
        """Heralded atom loss. Emits one measurement record per qubit."""
        if p <= 0 or not qubits:
            return
        self.circuit.append("HERALDED_ERASE", qubits, p)
        recs = self._measure(qubits)
        if not self.herald_detectors:
            return
        for q, r in zip(qubits, recs):
            patch = q // self.per_patch
            local = q % self.per_patch
            x, y = self._global_coord(patch, self.code.data[local])
            i = self._detector([r], (x, y, rnd, 1), "herald", rnd, patch)
            self.herald_info.append(
                {"det": i, "patch": patch, "data_index": local, "round": rnd, "qubit": q}
            )

    # -- circuit pieces ----------------------------------------------------------------
    def _prepare(self) -> None:
        nm = self.noise
        for patch, basis in enumerate(self.p.init_basis):
            qs = [self.data_q(patch, i) for i in range(self.code.n_data)]
            self.circuit.append("RX" if basis == "X" else "R", qs)
            self.circuit.append("Z_ERROR" if basis == "X" else "X_ERROR", qs, nm.p_r)
        anc = self.all_anc()
        self.circuit.append("R", anc)
        self.circuit.append("X_ERROR", anc, nm.p_r)
        self.circuit.append("TICK")

    def _syndrome_round(self) -> None:
        nm = self.noise
        rnd = self.n_rounds_done
        data = self.all_data()
        anc = self.all_anc()

        self._erase(data, nm.p_loss, rnd)
        self._dep1(data, nm.p_idle)

        x_anc = [self.anc_q(p, s.index) for p in range(self.n) for s in self.code.x_stabilizers]
        self.circuit.append("H", x_anc)
        self._dep1(x_anc, nm.p_1q)
        self._dep1([q for q in data], nm.p_idle)
        self.circuit.append("TICK")

        for step in range(4):
            pairs: list[int] = []
            for patch in range(self.n):
                for s in self.code.stabilizers:
                    coord = s.data[step]
                    if coord is None:
                        continue
                    dq = self.data_q(patch, self.code.data_index(coord))
                    aq = self.anc_q(patch, s.index)
                    if s.kind == "X":
                        pairs += [aq, dq]  # ancilla controls
                    else:
                        pairs += [dq, aq]  # data controls
            self.circuit.append("CX", pairs)
            self._dep2(pairs, nm.p_2q)
            busy = set(pairs)
            self._dep1([q for q in data + anc if q not in busy], nm.p_idle)
            self.circuit.append("TICK")

        self.circuit.append("H", x_anc)
        self._dep1(x_anc, nm.p_1q)
        self._dep1(data, nm.p_idle)
        self.circuit.append("TICK")

        self.circuit.append("X_ERROR", anc, nm.p_m)
        recs = self._measure(anc)
        self.circuit.append("MR", anc)
        self.circuit.append("X_ERROR", anc, nm.p_r)
        self._dep1(data, nm.p_idle)

        rec_of: dict[tuple[int, int], int] = {}
        k = 0
        for patch in range(self.n):
            for s_index in range(self.code.n_anc):
                rec_of[(patch, s_index)] = recs[k]
                k += 1

        self._declare_round_detectors(rec_of, rnd)
        self.last_round_rec = rec_of
        self.pending_gates = []
        self.n_rounds_done += 1
        self.circuit.append("TICK")

    def _declare_round_detectors(self, rec_of: dict[tuple[int, int], int], rnd: int) -> None:
        first = rnd == 0
        for patch in range(self.n):
            for s in self.code.stabilizers:
                if first:
                    # Only stabilizers matching the init basis are deterministic.
                    if s.kind != self.p.init_basis[patch]:
                        continue
                    recs = [rec_of[(patch, s.index)]]
                else:
                    support = _pull_back(frozenset({patch}), s.kind, self.pending_gates)
                    recs = [rec_of[(patch, s.index)]] + [
                        self.last_round_rec[(q, s.index)] for q in sorted(support)
                    ]
                x, y = self._global_coord(patch, s.anc)
                self._detector(recs, (x, y, rnd, 0), s.kind, rnd, patch)

    def _transversal_cx(self, layer: list[tuple[int, int]]) -> None:
        nm = self.noise
        rnd = self.n_rounds_done
        involved = sorted({p for pair in layer for p in pair})
        moving_data = [self.data_q(p, i) for p in involved for i in range(self.code.n_data)]

        # Atoms are physically transported to bring the two patches together.
        self._erase(moving_data, nm.p_loss_transport, rnd)
        self._dep1(moving_data, nm.p_transport)

        pairs: list[int] = []
        for c, t in layer:
            for i in range(self.code.n_data):
                pairs += [self.data_q(c, i), self.data_q(t, i)]
        self.circuit.append("CX", pairs)
        self._dep2(pairs, nm.p_2q)

        busy = set(pairs)
        idle = [q for q in self.all_data() + self.all_anc() if q not in busy]
        self._dep1(idle, nm.p_idle)
        self.circuit.append("TICK")

        self.pending_gates += layer
        self.all_gates += layer

    def _final_measure(self) -> None:
        nm = self.noise
        code = self.code
        data = self.all_data()
        for patch, basis in enumerate(self.p.meas_basis):
            qs = [self.data_q(patch, i) for i in range(code.n_data)]
            self.circuit.append("Z_ERROR" if basis == "X" else "X_ERROR", qs, nm.p_m)
        recs = self._measure(data)
        # One M/MX per patch keeps the record order aligned with ``data``.
        base = 0
        data_rec: dict[tuple[int, int], int] = {}
        for patch, basis in enumerate(self.p.meas_basis):
            qs = [self.data_q(patch, i) for i in range(code.n_data)]
            self.circuit.append("MX" if basis == "X" else "M", qs)
            for i in range(code.n_data):
                data_rec[(patch, i)] = recs[base + i]
            base += code.n_data

        rnd = self.n_rounds_done
        for patch in range(self.n):
            basis = self.p.meas_basis[patch]
            for s in code.stabilizers:
                if s.kind != basis:
                    continue  # only same-type stabilizers are reconstructible from the data
                recs_d = [
                    data_rec[(patch, code.data_index(c))] for c in s.data if c is not None
                ]
                support = _pull_back(frozenset({patch}), s.kind, self.pending_gates)
                recs_prev = [self.last_round_rec[(q, s.index)] for q in sorted(support)]
                x, y = self._global_coord(patch, s.anc)
                self._detector(recs_d + recs_prev, (x, y, rnd, 0), s.kind, rnd, patch)

        self._declare_observables(data_rec)

    def _declare_observables(self, data_rec: dict[tuple[int, int], int]) -> None:
        """Every initial logical stabilizer that survives to a same-basis readout."""
        code = self.code
        obs = 0
        for patch in range(self.n):
            kind = self.p.init_basis[patch]
            support = _push_forward(frozenset({patch}), kind, self.all_gates)
            if any(self.p.meas_basis[q] != kind for q in support):
                continue  # not readable in the final basis; skip it
            recs: list[int] = []
            for q in sorted(support):
                for c in code.logical(kind):
                    recs.append(data_rec[(q, code.data_index(c))])
            self.circuit.append(
                "OBSERVABLE_INCLUDE", [self._rec(r) for r in recs], obs
            )
            obs += 1
        self.n_obs = obs

    # -- entry point -------------------------------------------------------------------
    def build(self) -> CompiledCircuit:
        self._prepare()
        for op in self.p.ops:
            if op[0] == "rounds":
                for _ in range(op[1]):
                    self._syndrome_round()
            elif op[0] == "cx":
                self._transversal_cx(op[1])
            else:
                raise ValueError(f"unknown op {op[0]}")
        if self.n_rounds_done == 0:
            raise ValueError("program has no syndrome rounds")
        self._final_measure()

        return CompiledCircuit(
            circuit=self.circuit,
            program=self.p,
            d=self.d,
            noise=self.noise,
            n_detectors=self.n_det,
            herald_dets=self.herald_dets,
            herald_info=self.herald_info,
            syndrome_dets=self.syndrome_dets,
            det_coords=self.det_coords,
            n_observables=self.n_obs,
            round_of_det=self.round_of_det,
            kind_of_det=self.kind_of_det,
            patch_of_det=self.patch_of_det,
        )


def compile_program(
    program: LogicalProgram,
    d: int,
    noise: NoiseModel = NOISELESS,
    herald_detectors: bool = True,
) -> CompiledCircuit:
    return Compiler(program, d, noise, herald_detectors).build()


# --------------------------------------------------------------------------------------
# The four benchmark programs
# --------------------------------------------------------------------------------------
def memory(rounds: int, basis: str = "Z") -> LogicalProgram:
    return LogicalProgram(
        name=f"memory_{basis.lower()}",
        n_patches=1,
        init_basis=[basis],
        meas_basis=[basis],
        ops=[("rounds", rounds)],
        note="Single patch, no gates. Threshold calibration.",
    )


def transversal_cnot(rounds_between: int, basis: str = "Z") -> LogicalProgram:
    """A -> B transversal CNOT with ``rounds_between`` syndrome rounds around it."""
    return LogicalProgram(
        name=f"cnot_r{rounds_between}",
        n_patches=2,
        init_basis=[basis, basis],
        meas_basis=[basis, basis],
        ops=[
            ("rounds", rounds_between),
            ("cx", [(0, 1)]),
            ("rounds", rounds_between),
        ],
        note="The textbook case: shows O(d) rounds -> O(1) rounds.",
    )


def cnot_chain(rounds_between: int, basis: str = "Z") -> LogicalProgram:
    """Three transversal CNOTs in sequence: 0->1, 1->2, 2->0. Correlations accumulate."""
    return LogicalProgram(
        name=f"chain3_r{rounds_between}",
        n_patches=3,
        init_basis=[basis] * 3,
        meas_basis=[basis] * 3,
        ops=[
            ("rounds", rounds_between),
            ("cx", [(0, 1)]),
            ("rounds", rounds_between),
            ("cx", [(1, 2)]),
            ("rounds", rounds_between),
            ("cx", [(2, 0)]),
            ("rounds", rounds_between),
        ],
        note="Regime where correlation accumulates across gates.",
    )


def ghz(rounds_between: int, n_patches: int = 4) -> LogicalProgram:
    """Logical GHZ: |+>0 then a CNOT ladder. Exercises X-type correlations."""
    ops: list[tuple] = [("rounds", rounds_between)]
    for t in range(1, n_patches):
        ops.append(("cx", [(0, t)]))
        ops.append(("rounds", rounds_between))
    return LogicalProgram(
        name=f"ghz{n_patches}_r{rounds_between}",
        n_patches=n_patches,
        init_basis=["X"] + ["Z"] * (n_patches - 1),
        meas_basis=["X"] * n_patches,
        ops=ops,
        note="GHZ ladder; all-X readout gives the X_0 X_1 ... X_n stabilizer.",
    )
