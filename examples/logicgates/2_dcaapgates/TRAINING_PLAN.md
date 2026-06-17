# Training plan — end-to-end multi-step dCaAP logic circuits

How to get from hand-configured single-neuron gates to **trained, multi-layer spiking
logic circuits** using reward-modulated STDP (no backprop). Written against the
`dcap_logic_framework.py` toolkit and the snn2 R-STDP engine.

---

## 0. Goal

Train a circuit of dCaAP neurons to implement a target boolean function (XNOR →
half-adder → full-adder → 2-bit adder) from reward alone, such that the learned circuit
is **stable** — its inter-layer noise margins stay open and it generalizes across seeds.

Two regimes, used in sequence:
- **Configure** (done): place bands/weights analytically. Proves realizability, gives a
  known-good target for training to recover.
- **Train** (this plan): learn the synaptic weights (and, later, effective band
  placement) from reward.

---

## 1. The three hard problems

1. **Credit assignment across layers.** R-STDP broadcasts one global reward; a deep
   circuit must apportion it to the right synapses without a backward pass. Eligibility
   traces give per-synapse credit for *recent* pre/post coincidence, but a layer-2 error
   may be caused by a layer-1 weight.
2. **Cascade stability (the dominant issue).** Gate outputs are rates, not bits. If an
   upstream gate's true/false rates aren't matched to the downstream band, the noise
   margin collapses with depth (shown empirically: matched chain holds margin ≈ 1.0 to
   depth 10; a detuned interface collapses to 0 by depth 3). Training must *preserve*
   margins, not just fit truth tables.
3. **The band-entry-order trap.** With a band-pass neuron, pure-LTP can push the wrong
   input pattern's drive through the band first and crush all weights before the right
   pattern climbs in. Already seen in single-neuron XOR; it compounds in a cascade.

---

## 2. Three training strategies (with trade-offs)

| strategy | what's trained | pro | con |
|---|---|---|---|
| **A. Modular / greedy** | each gate alone, then frozen and wired | simple, robust, reuses single-gate R-STDP that already works | no inter-layer adaptation; interfaces must be matched by hand |
| **B. Interface-only** | gates frozen; only the inter-layer input weights learned end-to-end with global reward | directly targets the stability problem; small parameter count | assumes gates are already correct |
| **C. Full end-to-end** | all weights, global reward + eligibility traces | most general; can discover non-obvious solutions | hardest credit assignment; most prone to the trap and to margin collapse |

**Recommended path: A → B → C as a curriculum.** Configure for the target (known-good
init), train gates modularly to confirm each learns, then learn interfaces (B) to make
the cascade self-match, then optionally unfreeze everything (C) for fine-tuning. Each
stage starts from the previous stage's stable solution — this is the single most
important design decision, because it keeps every training run inside a good basin.

---

## 3. Stability mechanisms to build in (not optional)

- **Reward-only learning** (`punish_mult = 0`) + **weight cap inside the target window**
  — defuses the band-entry-order trap. Proven necessary for single-neuron XOR.
- **Homeostatic self-limiting** — the dCaAP band already gives reward-modulated plasticity
  a stable interior fixed point (the `florian2` result); rely on it so weights settle
  *in-band* instead of running to a ceiling. A small homeostatic weight decay sharpens
  this into a true attractor.
- **Interface normalization** — constrain each gate's *total* input drive (sum of
  weights × upstream rates) to a target range, so a gate self-tunes to sit in its band
  regardless of how many upstream lines it has. This is the trainable version of
  `match_interface`.
- **Margin-aware reward** — add a term that rewards *separation* between a node's
  true-pattern and false-pattern rates, not just final-output correctness. This makes
  the optimizer protect noise margins explicitly (otherwise it will fit the truth table
  with margins as small as 0.05 and the circuit fails one layer deeper).

---

## 4. Milestones

1. **XNOR (depth 2).** NOT ∘ XOR. Smallest non-single-neuron function. Train XOR (works),
   then learn the NOT interface weight with global reward (strategy B). Success: truth
   `[1,0,0,1]`, margin > 0.5 across seeds.
2. **Half-adder (depth 1, two outputs).** SUM=XOR, CARRY=AND in parallel. Tests two
   read-outs sharing inputs. Mostly a modular-training check.
3. **Full-adder (depth ~3).** Two half-adders + OR on the carries — the first real carry
   chain. First place the trap and margin-collapse problems bite together. Train A→B.
4. **2-bit ripple adder (depth ~6).** Carry propagates across bits — the deepest cascade;
   the stability mechanisms (§3) are what make or break it.
5. **2:1 multiplexer / 1-of-n selector.** Different topology (control line gates data);
   tests the framework beyond adders.

---

## 5. Metrics & validation

- **Per-gate accuracy** (truth-table match) — confirms each unit learned.
- **Composed accuracy** — truth-table match of the whole circuit.
- **Noise margin vs depth** — `cascade_margin`; the stability signal. Track per layer.
- **Trap incidence** — fraction of seeds that collapse to all-silent / all-firing.
- **Seed robustness** — converged-seeds / total; spread of final weights.
- **Negative control** — detune one interface and *show* the margin collapse, to prove
  the matched result isn't accidental.

Validation ladder per milestone: gate truth tables → composed truth table → margin holds
at target depth → collapse reproduced under detuning → holds across ≥8 seeds.

---

## 6. First concrete experiments (week-one)

1. **Interface-learning on XNOR (strategy B).** Freeze a trained XOR and a configured
   NOT; learn only the NOT input weight `w_if` from global reward on the XNOR truth table.
   Expected: `w_if` converges into the `match_interface` range; margin > 0.5. This
   validates the whole "learn the interface" idea on the smallest case.
2. **Margin-aware reward ablation.** Repeat (1) with and without the separation term;
   measure final margin. Hypothesis: the plain reward fits the table with a thin margin;
   the margin term widens it. This decides whether §3's margin reward is needed downstream.
3. **Trap reproduction + fix in a 2-layer circuit.** Train full-adder carry path with
   `punish_mult ≠ 0` (expect collapse) vs `= 0` + cap (expect success). Confirms the
   single-neuron fix scales.

---

## 7. Open questions / risks

- **Is global-reward credit assignment enough at depth ≥ 4**, or is a layer-local reward
  (per-gate target signals during a pretraining phase) required? Likely need local
  targets early, global reward for fine-tuning.
- **Learned vs fixed bands.** Here the band is fixed and weights move. A metaplasticity
  rule that *adapts* `[lo, hi]` to input statistics could remove most interface matching —
  worth a parallel investigation (see snn2 `dcaap.md`, open directions).
- **Rate vs spike-timing codes between layers.** This plan uses rate interfaces. A
  timing-based interface (dCaAPs are fast, precise events) could carry more bits per
  spike but complicates stability analysis.
- **Scaling group size / noise.** Margins depend on `GROUP` and `RATE`; quantify the
  margin-vs-noise trade-off before committing to deep circuits.

---

## 8. Definition of done

A 2-bit ripple adder, trained (gates modularly + interfaces end-to-end), that produces
the correct 4-bit truth table with noise margin > 0.4 at every node across ≥8 seeds, and
whose margin collapse is reproducible under interface detuning. That demonstrates the
full claim: dCaAP neurons compose into trainable, stable, multi-step spiking logic.
