"""Shared knobs. One place, so a reviewer can find them.

``ERASURE_TRIAGE_QUICK=1`` divides every shot count by ``QUICK_DIVISOR``. The
conclusions are the same; the error bars are wider. Used by ``run_all.py --quick``.
"""

from __future__ import annotations

import os

QUICK = os.environ.get("ERASURE_TRIAGE_QUICK") == "1"
QUICK_DIVISOR = 8

# The physical regime everything is quoted at.
P_TARGET = 1e-3  # matching fails ~1e-3 per shot here: the deployment regime
P_TRAIN = 3e-3  # 3x, where failures are common enough to train on (§7.3)
LOSS_RATIO = 1.0  # p_loss / p_2q; heralded atom loss, the neutral-atom lever


def shots(n: int, floor: int = 2_000) -> int:
    return max(floor, n // QUICK_DIVISOR) if QUICK else n


def budget_s(seconds: float, floor: float = 4.0) -> float:
    """Scale a wall-clock time budget (e.g. for BP+OSD sampling) down in quick mode."""
    return max(floor, seconds / QUICK_DIVISOR) if QUICK else seconds
