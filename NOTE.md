# When is the exact decoder actually needed?

### Escalation decoding for transversal logical circuits on neutral atoms

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

The error scaling agrees. Fitting LER ∝ p^s on the same shots and the same DEM, changing only
the decoder:

| decoder | s | implied effective distance |
|---|---|---|
| MWPM, memory d=5 | 2.98 | 5 ✓ |
| MWPM, cnot r=1 d=5 | **1.16** | **2** |
| BP+OSD, cnot r=1 d=5 | **5.90** | **5** ✓ |

The code has distance *d* — Stim's `search_for_undetectable_logical_errors` confirms it for
every circuit here. It is the *decoder* that loses the distance, not the code.

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

*(Figure 1: threshold vs loss rate, for (a), (b), (c).)*

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

*(Figure 2: the Pareto frontier. Figure 3: feature importance — does the heralded-loss
information actually carry the gain? Figure 4: train at d = 3,5, evaluate at d = 7 with no
retraining.)*

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

Not a bigger model — inference is the binding constraint, and the model is already
microseconds against a decoder that costs milliseconds. It would buy: more code distances
(*d* = 9, 11, to see whether the ~0.5% escalation rate holds or grows); full logical
algorithms rather than gate gadgets; a sliding-window decoder so the escalation decision is
made under a real backlog rather than shot-by-shot; and real device data, where the loss
herald is noisy rather than perfect.
