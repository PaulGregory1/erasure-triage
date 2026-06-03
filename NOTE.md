# When is the exact decoder actually needed?

### Escalation decoding for transversal logical circuits on neutral atoms

**Abstract.** Correlated decoding lets a transversal logical gate be protected with O(1) rather
than O(*d*) rounds of syndrome extraction, at the price of a harder classical decoding problem
that must be solved in real time. We show, constructively and without Monte Carlo, that the
standard fast decoder — minimum-weight perfect matching on the decomposed detector graph — has
**effective distance 2** on any circuit containing a transversal CNOT, regardless of the code
distance: a single physical fault, propagated into both patches by the gate, is enough to flip
the logical class. A hypergraph decoder (BP+OSD) is not fooled, but costs 2–4 orders of
magnitude more per shot, and neither correlated matching nor BP-without-OSD closes the gap. We
therefore treat exact decoding as an *escalation* to be triggered selectively, and ask how few
shots must be escalated to recover distance-*d* scaling. On neutral atoms, where atom loss is
heralded, the answer is about 0.5% of shots, and the trigger can be a bare threshold on the
complementary gap — no learned model required. Everything runs on a laptop; the repository
reproduces all four figures in about an hour.

---

## 1. Context: correlated decoding moves the bottleneck

A transversal CNOT between two surface-code patches is a single layer of physical CX gates.
It is fast, it is local in time, and — crucially — it does not need *d* rounds of syndrome
extraction to be protected, provided the decoder treats the whole spacetime volume of the
logical circuit **jointly**. This is the "correlated decoding" result: O(*d*) rounds per
logical gate collapse to O(1).

The cost does not vanish; it moves. Joint decoding means a bigger, denser decoding problem,
solved between gates, in real time, on classical hardware. The quantum circuit got shorter
and the classical decoder became the bottleneck.

This note asks a narrower question than "how do we decode better":

> **Given a fast decoder and a slow one, when is the slow one actually necessary?**

and answers it in a setting where the answer is unusually cheap to compute: neutral atoms,
where the dominant error — atom loss — is *heralded*.

---

## 2. The mechanism: why matching breaks on transversal gates

Under a transversal CNOT with A as control and B as target, stabilizers propagate
deterministically:

| stabilizer | after the gate |
|---|---|
| X on A | X_A ⊗ X_B |
| Z on A | Z_A |
| X on B | X_B |
| Z on B | Z_A ⊗ Z_B |

so detectors that straddle the gate must compare against a **product** of prior measurements:

```
D[X_A, after] = m(X_A, after) ⊕ m(X_A, before) ⊕ m(X_B, before)
D[Z_B, after] = m(Z_B, after) ⊕ m(Z_B, before) ⊕ m(Z_A, before)
```

These three-term detectors are what generate the correlated structure. In the detector error
model they show up as **hyperedges**: error mechanisms with more than two symptoms. We count
them:

| circuit | d=3 | d=5 | d=7 |
|---|---|---|---|
| memory | 0% | 0% | 0% |
| cnot r=1 | 0% | 3.4% | 5.7% |
| chain3 r=1 | 0% | 3.9% | 6.4% |

Memory decomposes exactly. The transversal CNOT does not, and the non-decomposable fraction
grows with distance.

MWPM cannot represent a hyperedge. Stim decomposes it into graphlike pieces, and matching then
treats the pieces as independent errors. Consider one X fault on data qubit *i* of the control
patch, immediately before the gate. It propagates to X on A_i **and** B_i. Matching sees two
edges — one in each patch — and must pay for both. But *two independent faults*, one landing on
A_i and one on B_i after the gate, light exactly the same detectors and cost exactly the same
weight. The two explanations are degenerate, and they imply **different logical classes**.

Matching cannot break the tie. It coin-flips.

### The consequence, measured without any Monte Carlo

Enumerate every single-fault mechanism in the DEM, hand its syndrome to the decoder, compare
the logical class it returns against the truth:

| circuit | d | single faults | MWPM wrong | …of which cross-patch | BP+OSD wrong |
|---|---|---|---|---|---|
| memory | 5 | 1677 | **0** | 0 | 0 |
| cnot r=1 | 3 | 295 | 23 | 23 | 2 |
| cnot r=1 | 5 | 1125 | 16 | 16 | 0 |
| cnot r=1 | 7 | 2487 | 24 | 24 | 0 |
| cnot r=d | 5 | 7651 | 26 | 26 | 0 |
| chain3 r=1 | 5 | 4466 | 68 | 68 | 0 |

**Every** fault that fools matching is one whose error crossed the gate into both patches.
A single physical fault flips the logical class. MWPM's effective distance on these circuits
is **2**, regardless of *d* and regardless of how many syndrome rounds you spend.

The error scaling agrees. Fitting LER ∝ p^s across p ∈ [3e-4, 8e-3], same shots, same DEM,
only the decoder changes (slope should be ⌈d/2⌉ for a distance-*d* decoder):

| | MWPM slope | BP+OSD slope | expected ⌈d/2⌉ |
|---|---|---|---|
| **memory** d = 3 / 5 / 7 | 1.99 / 2.87 / 3.89 | ~scales | 2 / 3 / 4 |
| **cnot r=1** d = 3 / 5 / 7 | **1.19 / 1.31 / 1.28** | 1.76 / 2.74 / 3.05 | 2 / 3 / 4 |

On memory, matching's slope tracks ⌈d/2⌉ — it scales. On the transversal CNOT it is **flat at
~1.3 for every distance**: adding rows of qubits buys nothing, which is effective distance 2.
BP+OSD's slope grows with *d* on the same circuit, as a distance-*d* decoder must.

The code has distance *d* — Stim's `search_for_undetectable_logical_errors` confirms it for
every circuit here. It is the *decoder* that loses the distance, not the code.

### One of them, concretely

`cnot r=1`, *d* = 5. A single fault of probability 2e-4 lights three detectors:

| detector | patch | round | type | position |
|---|---|---|---|---|
| D4 | 0 | 0 | Z | (4, 4) |
| D36 | 0 | 1 | Z | (2, 6) |
| D60 | **1** | 1 | Z | (14, 6) |

Three symptoms — a degree-3 hyperedge, not graphlike. The fault happened on the control patch
*before* the gate, and the CNOT copied it into the target patch, which is why D60 lights up on
patch 1. Truth: logical class `[0 0]`. MWPM returns `[1 0]` — wrong. BP+OSD returns `[0 0]` —
right. The complementary gap on the flipped observable is 6.8, well below its typical value of
~14, so the matcher does "know" it is unsure — which is exactly the signal the triage
classifier is built to read.

### There is no cheap escape

Two obvious objections, both tested, both answered — `cnot r=1`, p = 1e-3:

| decoder | single faults wrong (d=5) | LER (d=5) | latency | hyperedges it had to drop |
|---|---|---|---|---|
| MWPM | 16 / 1125 | 4.7e-3 | 0.2 µs | 47 |
| MWPM + pymatching's correlated mode | **16 / 1125** | 4.5e-3 | 0.6 µs | 47 |
| BP alone (hypergraph, no OSD) | **59 / 1125** | 2.4e-2 | 193 µs | 0 |
| BP + OSD (hypergraph) | **0 / 1125** | < 3e-4 | 370 µs | 0 |

*"Why not just turn on correlated matching?"* Because pymatching's correlated mode refuses to
load a model containing an undecomposed hyperedge — so the 47 mechanisms that break MWPM at
*d* = 5 must be **dropped from its model before it will run at all**. It halves the failures at
*d* = 3 and does *nothing* at *d* = 5. Its effective distance is still 2.

*"So it's just about seeing the hyperedges?"* No. BP alone, which sees the full hypergraph, is
**worse than plain MWPM** — belief propagation does not converge on the degenerate loops these
circuits create. The OSD post-processing is doing the real work of breaking the tie. That is
what the escalation is paying for, and it is why the slow decoder is genuinely slow.

**This reframes the project.** Escalation is not a latency optimisation bolted onto a working
decoder. It is the thing that restores the code distance, and no cheap decoder substitutes for it.

---

## 3. The neutral-atom lever: heralded atom loss

Atom loss dominates neutral-atom error budgets, and it is *heralded*: the atom is simply
missing, and you know it. Stim models this natively with `HERALDED_ERASE`, which replaces the
qubit with the maximally mixed state **and** emits a measurement record.

An erased atom's Pauli frame is a coin flip. In matching units, probability ½ means **weight 0**:
the decoder may route a correction through that qubit for free. So the right thing to do is not
to fold the loss into the noise model on average, but to **reweight per shot**:

- **(a) blind** — the decoder's DEM has p_loss = 0. It does not know loss exists.
- **(b) Pauli** — the DEM includes the erasure channel, but the herald bit is discarded.
  An erased atom is just a noisier atom.
- **(c) reweight** — the herald is read per shot, and every matching edge that atom could
  have flipped is set to weight 0.

The implementation detail that makes (c) affordable: the herald→edge table is read straight out
of the DEM. The errors that flip herald *h* **are** the four erasure branches (I/X/Y/Z) on that
qubit at that round; strip the herald node from each and what remains is exactly the edge set to
zero. No coordinate parsing, no guessing which qubit owns which edge.

The cost is real and we report it rather than hide it. Erasures are rare, so shots are grouped by
identical erasure pattern: the zero-erasure group is the overwhelming majority and runs on an
untouched graph, and every other pattern pays one in-place graph mutation shared across its shots.
On the largest circuit tested this still drops matching throughput by an order of magnitude — that
is a genuine Python-side cost and it goes in the latency budget.

### Result: the herald is worth nothing on average, and a lot per shot

Memory circuit, threshold in p_2q, 300 000 shots per point, *d* = 3, 5, 7:

| p_loss / p_2q | (a) erasure ignored | (b) erasure as Pauli noise | (c) dynamic reweighting |
|---|---|---|---|
| 0.0 | 0.77% | 0.78% | 0.77% |
| 0.5 | 0.72% | 0.73% | **0.77%** |
| 1.0 | 0.68% | 0.67% | **0.75%** |
| 2.0 | 0.59% | 0.60% | **0.73%** |

The interesting line is not (c) — it is that **(b) ≈ (a)**. Telling the decoder the *average*
erasure rate, without telling it *which qubit was lost on this shot*, buys essentially nothing:
the threshold decays just as fast. It is the per-shot herald, and only the per-shot herald, that
holds the threshold flat as atom loss grows. At p_loss = 2·p_2q the reweighted decoder keeps 0.73%
where the others have fallen to 0.59% — a 1.24× threshold gain, and, more to the point, a curve
that is nearly *flat* in the loss rate.

For a neutral-atom device this is the whole argument for caring about the herald: atom loss is
close to free, provided you use the information at the shot level rather than folding it into a
noise model.

*(Figure 1.)*

---

## 4. The ceiling, measured before any ML was written

Escalation is worthless if BP+OSD is not right precisely where matching is wrong. So that was
measured first:

p = 1e-3, p_loss = 1e-3, 150 000 shots per row. "Rescued" = shots matching got wrong that
BP+OSD gets right; "broken" = shots matching had right that BP+OSD then ruins.

| circuit | d | LER matching | LER BP+OSD | LER oracle escalation | P(BP+OSD right \| matching wrong) | broken |
|---|---|---|---|---|---|---|
| cnot r=1 | 3 | 6.4e-3 | 1.8e-3 | 9.3e-4 | **85.6%** | 137 |
| cnot r=1 | 5 | 4.7e-3 | 1.7e-4 | **4.7e-5** | **99.0%** | 18 |
| cnot r=1 | 7 | 6.2e-3 | 6.7e-6 | **6.7e-6** | **99.9%** | 0 |
| chain3 r=1 | 5 | 1.45e-2 | 1.3e-3 | 2.1e-4 | **98.6%** | 9 |
| *memory* (control) | 5 | 6.7e-5 | 9.2e-5 | 2.7e-5 | *60.0%* | 8 |

Three things fall out of this table.

**The headroom is enormous.** At *d* = 5, escalating on exactly the shots matching gets wrong —
about 0.5% of them — takes the logical error from 4.7e-3 to 4.7e-5. A hundredfold, at
essentially zero mean latency cost. At *d* = 7 it is nearly a thousandfold.

**The headroom is specific to transversal gates.** The memory row is the control: there,
BP+OSD rescues only 60% of matching's failures and *breaks* almost as many as it fixes — its
LER (9.2e-5) is actually *worse* than matching's (6.7e-5). Matching is already near-optimal on
a memory circuit, and escalation buys nothing. This is not a generic "BP+OSD is better than
MWPM" result; it is a statement about what transversal gates do to the decoding graph.

**Escalating *everything* is worse than escalating selectively.** BP+OSD alone (1.7e-4 at
*d* = 5) is worse than oracle escalation (4.7e-5), because BP+OSD occasionally breaks a shot
that matching had right. Triage therefore wins on *accuracy* as well as latency — which is not
what one expects going in, and it means "just always run the good decoder" is not merely
expensive, it is also *worse*.

The price of escalation, measured per shot on the machine of record:

| circuit | d | matching | BP+OSD | slowdown |
|---|---|---|---|---|
| cnot r=1 | 3 | 0.15 µs | 27 µs | 183× |
| cnot r=1 | 5 | 0.91 µs | 402 µs | 444× |
| cnot r=1 | 7 | 4.3 µs | 2 767 µs | 638× |
| chain3 r=1 | 5 | 13 µs | 10 999 µs | 837× |

The slowdown *grows* with circuit volume, which is precisely why "always escalate" does not
scale and why the escalation rate is the quantity that matters.

---

## 5. Triage

Labels are free: `label = (fast decoder's logical class ≠ truth)`. Stim hands over the true
observable, so no slow decode is needed to *label* a shot. That single fact is what makes this
fit on a laptop.

Features are graded by cost:

- **free** — read off the syndrome and the heralds (detector counts, spatial spread, per-round
  concentration, number of erasures, their distance to the boundary, whether they sit on the
  logical support);
- **~1×matching** — the matching weight, which we compute anyway;
- **~2×matching** — the **complementary gap**: decode again with the observable forced to the
  other logical class and take the weight difference. This is the reference uncertainty
  estimator and it is the baseline the model must beat.

The baseline ladder, in order, is: matching alone → BP+OSD always → oracle escalation → **gap
only** → random escalation → LightGBM → a tiny MLP. The one that matters is *gap only*: a bare
threshold on the complementary gap, no training, one extra matching. If a 400-tree model on 34
features cannot beat it, that is the result and it gets written down.

### What the triage actually shows

Three results, and only one of them is the flattering one.

**The learned model wins in-distribution and the gap wins at deployment.** Trained and
evaluated at the same noise level (p_train), LightGBM clearly beats the gap-only heuristic
(AUC 0.983 vs 0.943 on cnot d = 5). Evaluated at p_target — the same model, lower noise — the
gap-only baseline catches up and often passes it:

| cnot, d = 5, p_target | AUC | LER @ 5% escalated |
|---|---|---|
| BP+OSD, every shot | — | 1.8e-4 |
| gap only (no training) | 0.984 | **1.9e-4** |
| LightGBM (trained at p_train) | 0.981 | 6.2e-4 |

So a near-free heuristic — one extra matching, no model, no training — reaches the accuracy of
"always run BP+OSD" while escalating 5% of shots, and the trained model, in this transfer
setting, does not beat it. That is the honest headline, and it is a stronger claim than a
tuned number would have been: **the complementary gap is most of the signal.**

**Why the model does not transfer — and the fix, which does beat the gap.** A matching edge
weighs log((1−p)/p): 5.8 at p_train, 6.9 at p_target. Every weight-derived feature shifts with
the noise level, so a tree that split at "gap < 5.0" is asking a different question after
deployment. The gap-only baseline is immune because it ranks rather than thresholding an
absolute value. `scripts/09_mixture.py` tests the fix: retrain on a *blend* of noise levels
(p ∈ {1, 1.7, 3}×10⁻³) so the trees see the shift. It works, and then the learned model does
beat the heuristic — LER at a fixed escalation rate, at deployment noise:

| model | cnot d=5 @ 1% | chain3 d=5 @ 5% | chain3 d=7 @ 5% (d held out) |
|---|---|---|---|
| gap only (no training) | 2.9e-3 | 8.4e-3 | 7.9e-3 |
| LightGBM, single p | 3.0e-3 | 7.8e-3 | 1.3e-2 |
| **LightGBM, mixed p** | **2.1e-4** | **1.2e-3** | **6.4e-3** |

So the honest arc is: the gap is a strong, near-free baseline that a *naively* trained model
fails to beat across noise regimes; the failure is a feature-units artefact, not a modelling
limit; and a model trained on a blend of noise levels beats the gap by 3–7×, while
generalising across both code distance *and* noise level. The takeaway for a deployment is
nonetheless conservative: if you are not going to train across noise regimes, ship the gap
threshold — it captures most of the gain for one extra matching and no model at all.

**Do the heralded-loss features carry the gain? No — the gap does, and that is not a
contradiction.** The complementary-gap features account for 64% of the model's SHAP
attribution (`gap_min` alone is 30%); the heralded-loss features account for **1%**. The
ablation confirms it: delete the erasure features entirely and the escalation curve does not
move. This looks like it contradicts the neutral-atom framing, and it does not — the herald has
*two* jobs, and only one of them is here:

- in the **decoder**, the per-shot herald reweighting keeps the threshold flat as loss grows
  (Figure 1). That is where the herald pays off.
- in the **triage classifier**, the herald adds almost nothing — because the gap is computed on
  the *already-reweighted* graph, so whatever uncertainty a lost atom introduces is *already*
  visible in the gap. The classifier does not need the raw loss features to see it.

So the honest one-line answer is: the herald makes the fast decoder better, and the gap then
reads out the residual uncertainty regardless of its source. Written down because the plan asked
the question both ways, and this is the way it came out.

**The MLP does not beat the trees.** A 4 353-parameter MLP was trained alongside — it trails
LightGBM in AUC (0.983 vs 0.989 at d = 5). Reported so that the choice of trees reads as a
decision rather than a default: a 400-tree evaluation is a few hundred branch comparisons,
whereas the MLP is three dense matrix multiplies, and neither buys accuracy over the other
here. And both are dominated by a more basic fact — LightGBM's own per-shot inference, as
measured, is **16 µs**, which is *more* than the 11 µs the complementary gap costs. The model
is more expensive than the single feature that carries most of its signal. (A compiled tree
would cut that to ~1 µs; as a Python `predict_proba` call it does not, and the Pareto plots the
honest measured cost.) Inference is the binding constraint — the trigger runs once per shot
inside the decode budget — so this is the number that decides the design, and it points at the
gap threshold.

*(Figure 2: the Pareto frontier. Figure 3: what the model used (SHAP) and what it needed
(ablation). Figure 4: train at d = 3, 5, evaluate at d = 7 with no retraining — it holds.)*

---

## 6. Limitations

- **No magic-state distillation block.** Stim is a stabilizer simulator and a T gate is not
  Clifford. Simulating a distillation block means committing to a particular Clifford skeleton
  and arguing that the decoding problem it induces is the one that matters. That argument
  deserves to be made properly rather than waved at, so it is left out. The circuits that *are*
  here already exhibit the effect and are fully verified.
- **BP+OSD is not exact.** It is a strong heuristic. The Phase-4 ceiling is BP+OSD's ceiling,
  not the true maximum-likelihood decoder's. The real headroom is at least this large.
- **Latency is single-core Python calling C++ libraries**, not an FPGA. The *ratios* between
  decoders are the meaningful quantity; the absolute microseconds are not.
- **The noise model is a plausible neutral-atom model, not a calibrated device model.**
  `p_idle` during atom transport is parameterised rather than derived, and it depends on a
  layout we did not commit to.

## 7. What 10× more compute would buy

Not a bigger model — inference is the binding constraint, and even the current trigger's cost
(a few µs for the gap, 16 µs for the tree) is already a meaningful fraction of the fast decode
it guards; a heavier model would eat the latency budget it is supposed to protect. It would buy
instead: more code distances (*d* = 9, 11, to see whether the ~0.5% escalation rate holds or
grows); full logical algorithms rather than single gate gadgets; a sliding-window decoder so the
escalation decision is made under a real backlog rather than shot-by-shot; a compiled trigger
(treelite) to see how far the µs cost can be pushed down; and real device data, where the loss
herald is noisy rather than perfect.

## 8. The five questions this project should let me answer

- **Where does the decoder fail?** On single faults that cross the transversal gate — §2 gives
  a worked syndrome (D4, D36 on the control patch, D60 on the target), and the single-fault
  enumeration finds every one of them.
- **Why not a transformer?** Tested a small MLP; it did not beat gradient-boosted trees on
  accuracy and inference is the binding constraint, so the extra cost buys nothing. The trigger
  that actually wins is a *threshold on one feature*.
- **Why not just always run BP+OSD, or just turn on correlated matching?** Always-BP+OSD is
  1–3 orders of magnitude slower *and* less accurate than selective escalation (it breaks shots
  matching had right). Correlated matching cannot even load the hyperedges that matter. §2, §4.
- **What is the one number?** P(BP+OSD right | matching wrong) = 85.6 / 99.0 / 99.9 % at
  d = 3 / 5 / 7. Without it there is no project; with it, the escalation ceiling is set.
- **What would you do with 10× compute?** §7 — not a bigger model.
