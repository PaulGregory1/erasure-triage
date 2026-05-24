"""Per-shot features for the triage classifier.

Labels are free (§7.1): ``label = (matching prediction != true logical class)``.
Stim hands us the true observable, so no slow decode is needed to *label* a shot.
That is the single fact that makes this project fit on a laptop.

Features are graded by what they cost:

  free       -- read straight off the syndrome and the heralds, vectorised over shots
  1x match   -- the matching weight, which we were computing anyway
  ~2x match  -- the complementary gap: decode again with the observable forced to the
                other class and take the weight difference. This is the reference
                uncertainty estimator, and it is the baseline the model must beat.
  ~1x match  -- edge-level features, only if `with_edges` (per-shot Python loop)

Everything that scales with the code is *also* emitted normalised by d or d^2, so a
model trained at d = 3,5 has a chance of transferring to d = 7 (§9.3).
"""

from __future__ import annotations

import numpy as np

from .dem_utils import DecodingProblem
from .decoders import ErasureMatchingDecoder

FREE = [
    "n_det", "n_det_norm", "n_det_x", "n_det_z", "det_xz_imbalance",
    "det_round_mean", "det_round_std", "det_round_max_count", "det_round_max_count_norm",
    "det_depth_frac",
    "det_x_std", "det_y_std", "det_spread", "det_spread_norm",
    "n_patches_lit", "max_det_per_patch", "det_patch_imbalance",
    "n_erase", "n_erase_norm", "n_patches_erased",
    "erase_round_mean", "erase_bdist_min", "erase_bdist_mean", "erase_on_logical",
    "erase_det_overlap",
]
MATCH = ["w_match", "w_match_norm"]
GAP = ["gap_min", "gap_mean", "gap_max", "gap_min_over_w"]
EDGE = ["n_edges", "n_hyper_edges", "frac_hyper_edges"]

ALL_FEATURES = FREE + MATCH + GAP + EDGE

# The heralded-atom-loss block. Figure 3 asks whether these actually carry the gain, so
# they need to be nameable in one place -- both to colour the importance plot and to train
# a model with them removed, which is the only way to answer the question directly.
ERASURE = [
    "n_erase", "n_erase_norm", "n_patches_erased", "erase_round_mean",
    "erase_bdist_min", "erase_bdist_mean", "erase_on_logical", "erase_det_overlap",
]
NO_GAP_FEATURES = [f for f in ALL_FEATURES if f not in GAP]
NO_ERASURE_FEATURES = [f for f in ALL_FEATURES if f not in ERASURE]


class FeatureExtractor:
    """Builds the shot-level design matrix for one decoding problem."""

    def __init__(self, problem: DecodingProblem, matcher: ErasureMatchingDecoder) -> None:
        self.pr = problem
        self.m = matcher
        d = problem.d
        n_dets = problem.n_dets

        # ---- detector-side lookup tables (one-hot, so features are matrix products) ----
        self.det_round = problem.det_round.astype(np.float32)
        self.n_rounds = int(problem.det_round.max()) + 1
        self.R = np.zeros((n_dets, self.n_rounds), dtype=np.float32)
        self.R[np.arange(n_dets), problem.det_round] = 1.0

        n_patches = problem.program.n_patches
        self.P = np.zeros((n_dets, n_patches), dtype=np.float32)
        self.P[np.arange(n_dets), problem.det_patch] = 1.0

        self.is_x = (problem.det_kind == 0).astype(np.float32)
        self.is_z = (problem.det_kind == 1).astype(np.float32)
        self.det_x = problem.det_xy[:, 0].astype(np.float32)
        self.det_y = problem.det_xy[:, 1].astype(np.float32)

        # ---- herald-side lookup tables ----
        nh = len(problem.herald_dets)
        self.n_heralds = nh
        code = problem.dec.code if hasattr(problem.dec, "code") else None
        from .patches import RotatedSurfaceCode

        code = RotatedSurfaceCode(d)
        logical_z = set(code.logical("Z"))
        bdist, onlog, hround, hpatch = [], [], [], []
        for info in problem.full.herald_info:
            x, y = code.data[info["data_index"]]
            # distance to the nearest patch boundary, in plaquettes
            bdist.append(min(x - 1, 2 * d - 1 - x, y - 1, 2 * d - 1 - y) / 2.0)
            onlog.append(1.0 if (x, y) in logical_z else 0.0)
            hround.append(float(info["round"]))
            hpatch.append(info["patch"])
        self.h_bdist = np.array(bdist, dtype=np.float32) if nh else np.zeros(0, np.float32)
        self.h_onlog = np.array(onlog, dtype=np.float32) if nh else np.zeros(0, np.float32)
        self.h_round = np.array(hround, dtype=np.float32) if nh else np.zeros(0, np.float32)
        self.HP = np.zeros((nh, n_patches), dtype=np.float32)
        if nh:
            self.HP[np.arange(nh), np.array(hpatch)] = 1.0

        # herald -> the detectors its erasure can flip (used for erase/detector overlap)
        self.HD = np.zeros((nh, n_dets), dtype=np.float32)
        for i, edges in enumerate(problem.herald_edges):
            for u, v in edges:
                self.HD[i, u] = 1.0
                if v != -1:
                    self.HD[i, v] = 1.0

    # ----------------------------------------------------------------------------------
    def extract(
        self,
        syndrome: np.ndarray,
        heralds: np.ndarray,
        with_gap: bool = True,
        with_edges: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Returns (X, matching_predictions, feature_names)."""
        d = self.pr.d
        S = syndrome.astype(np.float32)
        N = len(S)
        eps = 1e-9

        n_det = S.sum(1)
        safe = np.maximum(n_det, 1.0)

        rc = S @ self.R  # (N, n_rounds): lit detectors per round
        pc = S @ self.P  # (N, n_patches)

        mean_r = (S @ self.det_round) / safe
        var_r = (S @ (self.det_round**2)) / safe - mean_r**2
        mx = (S @ self.det_x) / safe
        my = (S @ self.det_y) / safe
        vx = np.maximum((S @ (self.det_x**2)) / safe - mx**2, 0)
        vy = np.maximum((S @ (self.det_y**2)) / safe - my**2, 0)
        spread = np.sqrt(vx + vy)

        nx = S @ self.is_x
        nz = S @ self.is_z

        if self.n_heralds:
            H = heralds.astype(np.float32)
            n_er = H.sum(1)
            safe_e = np.maximum(n_er, 1.0)
            hpc = H @ self.HP
            bd = np.where(heralds, self.h_bdist[None, :], np.inf)
            bd_min = np.where(n_er > 0, bd.min(1), -1.0)
            bd_mean = (H @ self.h_bdist) / safe_e
            er_round = (H @ self.h_round) / safe_e
            on_log = H @ self.h_onlog
            n_pe = (hpc > 0).sum(1).astype(np.float32)
            # how much of the lit syndrome sits on edges an erased atom could have flipped
            overlap = ((H @ self.HD) * S).sum(1) / safe
        else:
            z = np.zeros(N, dtype=np.float32)
            n_er = z
            bd_min = np.full(N, -1.0, dtype=np.float32)
            bd_mean = z
            er_round = z
            on_log = z
            n_pe = z
            overlap = z

        cols: dict[str, np.ndarray] = {
            "n_det": n_det,
            "n_det_norm": n_det / (d * d),
            "n_det_x": nx,
            "n_det_z": nz,
            "det_xz_imbalance": (nx - nz) / safe,
            "det_round_mean": mean_r,
            "det_round_std": np.sqrt(np.maximum(var_r, 0)),
            "det_round_max_count": rc.max(1),
            "det_round_max_count_norm": rc.max(1) / (d * d),
            "det_depth_frac": mean_r / max(self.n_rounds - 1, 1),
            "det_x_std": np.sqrt(vx),
            "det_y_std": np.sqrt(vy),
            "det_spread": spread,
            "det_spread_norm": spread / d,
            "n_patches_lit": (pc > 0).sum(1).astype(np.float32),
            "max_det_per_patch": pc.max(1),
            "det_patch_imbalance": (pc.max(1) - pc.min(1)) / safe,
            "n_erase": n_er,
            "n_erase_norm": n_er / (d * d),
            "n_patches_erased": n_pe,
            "erase_round_mean": er_round,
            "erase_bdist_min": bd_min,
            "erase_bdist_mean": bd_mean,
            "erase_on_logical": on_log,
            "erase_det_overlap": overlap,
        }

        syn_u8 = np.ascontiguousarray(syndrome.astype(np.uint8))
        # The prediction, weight and gap must all come from the decoder we actually
        # deploy -- erasure-aware matching. Measuring the gap on the un-reweighted graph
        # would describe a decoder that is not the one being triaged.
        pred, w, gaps = self.m.decode_grouped_full(syn_u8, heralds, with_gap=with_gap)
        cols["w_match"] = w.astype(np.float32)
        cols["w_match_norm"] = (w / d).astype(np.float32)

        if with_gap:
            cols["gap_min"] = gaps.min(1)
            cols["gap_mean"] = gaps.mean(1)
            cols["gap_max"] = gaps.max(1)
            # +1 in the denominator: a clean shot has w = 0, and gap/0 is not a feature.
            cols["gap_min_over_w"] = gaps.min(1) / (w + 1.0)
        else:
            for f in GAP:
                cols[f] = np.zeros(N, dtype=np.float32)

        if with_edges and self.m._hyper_edges:
            ne = np.zeros(N, dtype=np.float32)
            nh = np.zeros(N, dtype=np.float32)
            hyper = self.m._hyper_edges
            for i in range(N):
                edges = self.m.matching.decode_to_edges_array(syn_u8[i])
                ne[i] = len(edges)
                nh[i] = sum(1 for u, v in edges if (int(u), int(v)) in hyper)
            cols["n_edges"] = ne
            cols["n_hyper_edges"] = nh
            cols["frac_hyper_edges"] = nh / np.maximum(ne, 1.0)
        else:
            for f in EDGE:
                cols[f] = np.zeros(N, dtype=np.float32)

        names = ALL_FEATURES
        X = np.column_stack([cols[n] for n in names]).astype(np.float32)
        return X, pred, names


def labels_from(pred: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """1 where the fast decoder got the logical class wrong. This is the target."""
    return (pred.astype(bool) != obs.astype(bool)).any(axis=1).astype(np.int8)
