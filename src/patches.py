"""Rotated surface code patch geometry.

Conventions reproduce ``stim.Circuit.generated("surface_code:rotated_memory_*")``:

* data qubits sit at odd coordinates ``(2i+1, 2j+1)``, ``0 <= i, j < d``
* measure qubits sit at even coordinates ``(2a, 2b)``, ``0 <= a, b <= d``
* a measure qubit is X-type iff ``a + b`` is odd, Z-type iff ``a + b`` is even
* boundary measure qubits are weight-2, interior ones weight-4
* Z_L = the bottom data row (y = 1), X_L = the left data column (x = 1)

The CX schedules are the standard "Z" (X-ancilla) and "N" (Z-ancilla) orders;
using anything else silently costs you a factor of 2 in effective distance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# (dx, dy) offsets from the measure qubit to its data neighbours, in CX time order.
X_ORDER = [(+1, +1), (-1, +1), (+1, -1), (-1, -1)]
Z_ORDER = [(+1, +1), (+1, -1), (-1, +1), (-1, -1)]

Coord = tuple[int, int]


@dataclass
class Stabilizer:
    """One stabilizer generator of a single patch."""

    index: int  # position within the patch's stabilizer list (shared across patches)
    kind: str  # "X" or "Z"
    anc: Coord  # measure-qubit coordinate, patch-local
    data: list[Coord | None]  # patch-local data coords in CX time order; None = idle step


@dataclass
class RotatedSurfaceCode:
    """Geometry of a single distance-``d`` rotated surface code patch."""

    d: int
    data: list[Coord] = field(init=False)
    stabilizers: list[Stabilizer] = field(init=False)
    x_stabilizers: list[Stabilizer] = field(init=False)
    z_stabilizers: list[Stabilizer] = field(init=False)

    def __post_init__(self) -> None:
        if self.d < 2:
            raise ValueError("distance must be >= 2")
        d = self.d
        self.data = [(2 * i + 1, 2 * j + 1) for j in range(d) for i in range(d)]
        data_set = set(self.data)

        stabs: list[Stabilizer] = []
        for b in range(d + 1):
            for a in range(d + 1):
                anc = (2 * a, 2 * b)
                kind = "X" if (a + b) % 2 == 1 else "Z"
                interior = 0 < a < d and 0 < b < d
                if not interior:
                    # Only weight-2 boundary plaquettes of the correct type survive.
                    on_vertical_edge = a in (0, d) and 0 < b < d
                    on_horizontal_edge = b in (0, d) and 0 < a < d
                    if on_vertical_edge and kind != "Z":
                        continue
                    if on_horizontal_edge and kind != "X":
                        continue
                    if not (on_vertical_edge or on_horizontal_edge):
                        continue  # corner: at most one data neighbour
                order = X_ORDER if kind == "X" else Z_ORDER
                neighbours = [
                    (anc[0] + dx, anc[1] + dy) if (anc[0] + dx, anc[1] + dy) in data_set else None
                    for dx, dy in order
                ]
                if sum(n is not None for n in neighbours) not in (2, 4):
                    raise AssertionError(f"stabilizer at {anc} has bad weight")
                stabs.append(Stabilizer(index=len(stabs), kind=kind, anc=anc, data=neighbours))

        if len(stabs) != d * d - 1:
            raise AssertionError(f"expected {d * d - 1} stabilizers, built {len(stabs)}")

        self.stabilizers = stabs
        self.x_stabilizers = [s for s in stabs if s.kind == "X"]
        self.z_stabilizers = [s for s in stabs if s.kind == "Z"]

    @property
    def n_data(self) -> int:
        return self.d * self.d

    @property
    def n_anc(self) -> int:
        return self.d * self.d - 1

    @property
    def n_qubits(self) -> int:
        return 2 * self.d * self.d - 1

    def logical(self, basis: str) -> list[Coord]:
        """Representative logical operator support (patch-local data coords)."""
        if basis == "Z":
            return [(2 * i + 1, 1) for i in range(self.d)]  # bottom row
        if basis == "X":
            return [(1, 2 * j + 1) for j in range(self.d)]  # left column
        raise ValueError(basis)

    def data_index(self, coord: Coord) -> int:
        return self.data.index(coord)
