"""Circuit-level noise model for neutral-atom surface-code patches.

Every channel is a parameter, never a hard-coded constant. Defaults sit in the
middle of the ranges quoted for Rydberg-based neutral-atom processors.

The channel that makes this project different is ``p_loss``: atom loss is
*heralded*. ``HERALDED_ERASE`` replaces the qubit with the maximally mixed state
*and* emits a measurement record, so the decoder knows exactly which qubit was
lost. That free information is the lever the whole triage story rests on.

Note on ``p_idle``: a transversal gate on neutral atoms requires physically
moving atoms, which takes hundreds of microseconds against a T2 of a few
milliseconds. The idle error accumulated during transport is therefore *not*
negligible and it depends on the layout, so it gets its own parameter rather
than being folded into the gate error.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class NoiseModel:
    p_2q: float = 3e-3  # Rydberg two-qubit gate (DEPOLARIZE2)
    p_1q: float = 3e-4  # single-qubit gate (DEPOLARIZE1), ~p_2q / 10
    p_idle: float = 3e-4  # per-round idling on qubits not being acted on
    p_transport: float = 1e-3  # extra idling while atoms are moved for a transversal gate
    p_m: float = 3e-3  # measurement flip (X_ERROR before M)
    p_r: float = 3e-3  # reset flip (X_ERROR after R)
    p_loss: float = 0.0  # heralded atom loss per data qubit per round (HERALDED_ERASE)
    p_loss_transport: float = 0.0  # extra heralded loss during a transversal-gate move

    def scaled(self, factor: float) -> "NoiseModel":
        """Scale every Pauli/loss rate by ``factor`` (used for the p_train sweep)."""
        return replace(
            self,
            p_2q=self.p_2q * factor,
            p_1q=self.p_1q * factor,
            p_idle=self.p_idle * factor,
            p_transport=self.p_transport * factor,
            p_m=self.p_m * factor,
            p_r=self.p_r * factor,
            p_loss=self.p_loss * factor,
            p_loss_transport=self.p_loss_transport * factor,
        )

    def without_loss(self) -> "NoiseModel":
        """The model a decoder sees when it is blind to atom loss (baseline (a))."""
        return replace(self, p_loss=0.0, p_loss_transport=0.0)

    @staticmethod
    def uniform(p: float, loss_ratio: float = 1.0) -> "NoiseModel":
        """Single-parameter model used for threshold sweeps.

        ``loss_ratio`` sets ``p_loss = loss_ratio * p``, so a sweep in ``p`` at
        fixed ratio keeps the loss/Pauli balance constant.
        """
        return NoiseModel(
            p_2q=p,
            p_1q=p / 10,
            p_idle=p / 10,
            p_transport=p / 3,
            p_m=p,
            p_r=p,
            p_loss=loss_ratio * p,
            p_loss_transport=loss_ratio * p / 3,
        )


NOISELESS = NoiseModel(0, 0, 0, 0, 0, 0, 0, 0)
