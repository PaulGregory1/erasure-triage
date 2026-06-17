# ErasureTriage

**Escalation decoding for transversal logical circuits on neutral atoms.**

Correlated decoding makes transversal gates O(1) in syndrome rounds, but it moves the
bottleneck onto the classical decoder. This repo does not try to decode *better*. It
learns **when the exact decoder is actually needed** — and on neutral atoms, where atom
loss is *heralded*, that decision turns out to be cheap to make.

Deliverable: a Pareto frontier of **logical error per transversal CNOT vs decode latency**
(mean *and* p99).

---

## The result that matters

Minimum-weight perfect matching on the decomposed detector graph has **effective distance
2** on any circuit containing a transversal CNOT — at any code distance, at any number of
syndrome rounds.

That is not a statistical statement. Feed every single-fault error mechanism of the DEM to
the decoder and check the logical class it returns:

| circuit | d | single faults | MWPM wrong | …of which cross-patch | BP+OSD wrong |
|---|---|---|---|---|---|
| memory | 5 | 1677 | **0** | 0 | 0 |
| cnot r=1 | 3 | 295 | 23 | **23** | 2 |
| cnot r=1 | 5 | 1125 | 16 | **16** | 0 |
| cnot r=1 | 7 | 2487 | 24 | **24** | 0 |
| chain3 r=1 | 5 | 4466 | 68 | **68** | 0 |

Every single fault that fools MWPM is one that lit detectors in **both** patches — i.e. an
error that crossed the transversal gate. One physical fault before the CNOT propagates into
two patches; the decomposed graph can only write that as two independent errors, whose total
weight equals that of the honest single-fault explanation. Matching coin-flips, and half the
time it flips the logical class. BP+OSD keeps the hyperedge and is never fooled.

Confirmed independently by the error scaling. Fitting LER ∝ p^s (slope should be ⌈d/2⌉ for a
distance-*d* decoder), same shots, only the decoder differs:

| circuit | MWPM slope (d = 3 / 5 / 7) | BP+OSD slope | verdict |
|---|---|---|---|
| memory | 1.99 / 2.87 / 3.89 | tracks ⌈d/2⌉ | matching scales fine |
| cnot r=1 | **1.19 / 1.31 / 1.28** | 1.76 / 2.74 / 3.05 | **matching is flat — effective distance 2** |

On memory, matching's slope grows with *d*. On the transversal CNOT it is flat at ~1.3 for
every distance — adding rows of qubits buys nothing. The code has distance *d* (Stim confirms
it); it is the **decoder** that loses the distance.

### And there is no cheap escape (`cnot r=1`, d = 5)

| decoder | single faults wrong | LER | latency | hyperedges dropped |
|---|---|---|---|---|
| MWPM | 16 / 1125 | 4.7e-3 | 0.2 µs | 47 |
| MWPM + pymatching's correlated mode | **16 / 1125** | 4.5e-3 | 0.6 µs | 47 |
| BP alone (hypergraph, no OSD) | **59 / 1125** | 2.4e-2 | 193 µs | 0 |
| BP + OSD (hypergraph) | **0 / 1125** | < 3e-4 | 370 µs | 0 |

*"Just turn on correlated matching."* It refuses to load a model containing an undecomposed
hyperedge, so the 47 mechanisms that break it must be **dropped from its model first**. It
halves the failures at d = 3 and does *nothing* at d = 5.

*"So it's about seeing the hyperedges."* No — BP alone, which sees the whole hypergraph, is
**worse than plain MWPM**. It doesn't converge on these degenerate loops. OSD is doing the
real work, and that is what makes the slow decoder slow.

**So escalation is not an optimisation. It is what restores the code distance.** The question
is not "can we shave latency", it is "how few shots can we escalate and still get distance-*d*
scaling".

Answer: **~0.5%** — and the cheapest possible trigger gets you there.

## The triage, and the honest version of the result

Labels are free: a shot is "positive" iff the fast decoder got the logical class wrong, and
Stim hands over the truth, so no slow decode is needed to label. The escalation trigger is
then a score per shot, swept into a Pareto front. The baseline that matters is **gap only**: a
threshold on the *complementary gap* (decode twice, once with the observable forced each way,
take the weight difference) — one extra matching, no model, no training.

| cnot, d = 5, at deployment noise | escalate | logical error |
|---|---|---|
| never (matching alone) | 0% | 4.7e-3 |
| always (BP+OSD) | 100% | 1.8e-4 |
| **gap only** | **5%** | **1.9e-4** |
| LightGBM (400 trees, 34 features) | 5% | 6.2e-4 |

The near-free heuristic reaches BP+OSD's accuracy escalating 5% of shots. A *naively* trained
LightGBM does **not** beat it at deployment noise — it wins in-distribution and fails to
transfer, because the matching weight is in log-odds units that move with the noise rate. That
failure is diagnosable and fixable: retrain on a **blend** of noise levels and the model beats
the gap by 3–7× (`scripts/09_mixture.py`; cnot d=5 goes from 6.2e-4 to 2.1e-4 at 1% escalation).
So the honest arc is *the complementary gap is most of the signal, and a units-aware model beats
it.* A tiny MLP was also tried; it does not beat the trees, and its inference is no cheaper in
practice — LightGBM's own per-shot inference (16 µs) already exceeds the gap it improves on, and
inference is the binding constraint. The model transfers across code *distance* cleanly too:
trained on d = 3, 5, it hits BP+OSD's floor on held-out d = 7 (Figure 4).

---

## The atom-loss lever

Atom loss on neutral atoms is *heralded* — you know which qubit vanished. An erased atom's
Pauli frame is a coin flip, which in matching units is **weight 0**. So the right move is to
zero the weight of the touched edges **per shot**, not to fold the loss rate into the noise
model. Memory threshold, three decoders:

| p_loss / p_2q | (a) erasure ignored | (b) erasure as Pauli noise | (c) per-shot reweighting |
|---|---|---|---|
| 0.0 | 0.77% | 0.78% | 0.77% |
| 1.0 | 0.68% | 0.67% | **0.75%** |
| 2.0 | 0.59% | 0.60% | **0.73%** |

The striking line is that **(b) ≈ (a)**: telling the decoder the *average* loss rate, without
telling it *which* atom was lost on *this* shot, buys nothing at all. Only the per-shot herald
keeps the threshold flat.

A subtlety the ablation makes precise (Figure 3): the herald pays off in the **decoder**
(reweighting), not in the **triage classifier**. The erasure features contribute ~1% of the
model's attribution, because the complementary gap — computed on the already-reweighted graph —
already reflects whatever uncertainty a lost atom introduces. The herald makes the fast decoder
better; the gap then reads out the residual uncertainty regardless of where it came from.

## Figures

| | |
|---|---|
| `figs/00_distance_collapse.png` | MWPM's d = 3, 5, 7 curves collapse onto one line; BP+OSD's don't |
| `figs/01_erasure_threshold.png` | threshold vs atom-loss rate, with and without per-shot reweighting |
| `figs/02_pareto.png` | **the figure** — logical error per transversal CNOT vs latency (mean + p99) |
| `figs/03_feature_importance.png` | what the model *used* (SHAP) and what it *needed* (ablation) |
| `figs/04_generalization.png` | train at d = 3, 5; evaluate at d = 7 without retraining |

---

## Reproduce

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # macOS also needs: brew install libomp
python run_all.py                      # ~1 h on a laptop (Apple M3 Pro)
python run_all.py --quick              # ~20 min, wider error bars, same conclusions
python run_all.py --from 06_train_triage   # resume from a stage
pytest tests/                          # the distance check and the single-fault check
```

No GPU. No cluster. The whole thing runs on one laptop core-hour budget, which is the point:
compare against a repo that needs 40 h of A100 and ask which one a reviewer will actually run.
The wall time is dominated by BP+OSD (the slow decoder itself), not by anything that would go
faster on better hardware — which is rather the point of the project.

**Machine of record for every latency number:** Apple M3 Pro, `OMP_NUM_THREADS=1`, per-shot
`perf_counter_ns`, warm-up discarded, median and p99 over ≥10⁴ shots. A mean taken over a
vectorised batch measures throughput, not latency, and a decoder with a good mean and a bad
p99 builds a backlog and is useless — so the tail is reported everywhere.

---

## What is in here

```
src/
  patches.py     rotated surface code geometry (reproduces stim's layout exactly)
  circuits.py    the transversal-CNOT compiler — detectors, observables, hyperedges
  noise.py       neutral-atom noise model, incl. HERALDED_ERASE for atom loss
  dem_utils.py   DEM surgery: herald→edge tables, the observable-as-detector trick
  decoders.py    MWPM · erasure-aware MWPM · BP+OSD · per-patch (straw man)
  features.py    34 per-shot features, graded by what they cost
  triage.py      the escalation models and the full baseline ladder
  evaluate.py    error bars, per-shot latency, the single-fault test
scripts/01..09   one phase each; each writes JSON into data/
```

The core piece is `circuits.py`. A transversal CNOT makes a stabilizer of patch *p* map to
the same stabilizer on a **set** of patches, so detectors that straddle the gate must compare
against a *product* of prior measurements:

```
X_A(after) = X_A(before) ⊕ X_B(before)      Z_B(after) = Z_B(before) ⊕ Z_A(before)
```

Generalised to arbitrary CNOT networks, this is a linear map on F₂^(n_patches), applied in
reverse chronological order. Get it wrong and the code distance silently drops below *d* —
which is why `tests/test_circuits.py` checks the distance before anything else runs.

## Honest limitations

- The "magic state distillation" circuit of the original plan is **not** here. Stim is a
  stabilizer simulator; a T gate is not Clifford. The circuits that *are* here (memory,
  transversal CNOT, a 3-CNOT chain, a 4-patch GHZ ladder) already exhibit the effect and are
  fully verifiable. Adding a distillation block means committing to a specific Clifford
  skeleton, and that deserves its own justification rather than a hand-wave.
- BP+OSD is the "exact" decoder here, and it is not exact — it is a good heuristic. The
  ceiling in Phase 4 is BP+OSD's ceiling, not the true ML decoder's.
- Latencies are single-core Python/C++ library calls, not an FPGA. The *ratios* are the
  meaningful thing, not the absolute microseconds.
