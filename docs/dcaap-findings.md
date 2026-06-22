# FINDINGS — dCaAP dendritic logic, learning, and the non-monotonic band

A synthesis of the dCaAP (non-monotonic dendritic neuron) work in `snn2`: what was
built, what was learned, what is genuinely novel, and where the boundaries are. Written
to be calibrated — claims are tagged by how surprising they are, and the honest scope
limits are stated alongside the results.

---

## 0. The one idea everything traces back to

The dCaAP neuron is a **non-monotonic band-pass unit**: it fires when its dendritic drive
lands in a window `[lo, hi]` and falls silent both *below* and *above* it. The biological
basis is the dendritic Ca²⁺ action potential reported by Gidon et al. (2020, *Science*
367:83–87; see `dcaap.md` §8 for the full citation). Nearly every result here is a
consequence of that single geometric fact. The central conceptual
contribution is recognizing that several seemingly-separate learning phenomena are **the
same thing** — the sign-flip of `dw/dt` produced by the band's upper edge:

- **Homeostatic cap** — in closed loop, weights settle at a stable fixed point instead of
  running away; the upper band edge pushes back. (`florian2/`)
- **Band-entry-order trap** — XOR only learns with punishment *off*, because punishing
  over-firing during the weight ascent crushes the weight before it lands in the band.
  (`dcapLogicGates/`, `dcapRStdpRewards/`)
- **Silent dead-zone** — between "single-input drive exits the band" and "double-input
  drive enters it" there is a gap where the neuron is silent and the reward gradient is
  exactly zero. (`dcapRStdpRewards/`, `dcapGlobalLearn/`)

Cap, trap, dead-zone — **one cause**. This unification is the spine of the project.

---

## 1. Results by how surprising they are

### [MOST NOVEL] The band-pass reverses credit assignment, gate-specifically
Changing the reward from `+1/0` (reward-only) to `+1/−x` (reward + punishment) exposes an
opposite optimum for different gates:

| gate | reached by | reward-only | + punishment |
|---|---|---|---|
| XOR | raising weights into the band | **learns (≈1.0)** | hurts (≈0.5) |
| OR  | raising weights into the band | **learns (≈0.82)** | hurts (≈0.45) |
| AND | lowering weights from over-driven | fails (≈0.44) | helps (≈0.72), then **walls** |

AND walls because punishment drives the weight straight **through** the narrow window to
zero (silence): near that window the wrong-firing input sits closer to the band than the
correct one, so wrong-firing is punished harder than correct-firing is rewarded — the band
**reverses the local gradient**. 

**Conclusion:** *no single Hebbian reward structure can learn all three gates on a dCaAP.*
This is not a tuning failure; it is forced by the non-monotonic geometry. It is a real,
mechanistically-explained limit on reward-modulated STDP for this class of unit.
(Evidence + figure: `dcapRStdpRewards/`.)

### [MOST NOVEL] The control-theory fix: oscillatory extremum-seeking
Diagnosing AND's failure as a textbook **deadband + non-monotonic plant** problem points
to the right tool — **extremum-seeking control (ESC)**. A probing *oscillation* (dither)
does three things at once:

1. **crosses the deadband** — dither is the classical fix for deadbands in servo control;
   it periodically pushes drive into the band so feedback exists where the static gradient
   is zero;
2. **recovers the correct local gradient sign through the non-monotonicity** — via
   phase-sensitive demodulation of the reward's response to the dither;
3. **locks at the target** (does not overshoot) — because the dither amplitude is
   **reward-gated**, shrinking to zero as reward → 1.

Add **reward-gated annealing** (Metropolis acceptance, to cross reward *valleys*) and a
**relaxation restart** (to escape genuinely dead configurations), and the target becomes a
**global attractor**:

> **AND converges to 100% from every pathological start tested — including a fully-dead
> neuron — using one rule, without breaking XOR (97%) or OR (100%).**

The biological mapping is the suggestive part: the dither is a neural oscillator, the
reward×phase correlation is a **three-factor (neuromodulated) signal**, and the
reward-gated amplitude is exploration variability that **shrinks as a task is mastered**
(as observed in motor learning). This is **oscillation-gated three-factor plasticity** —
theta/gamma-flavored. (Evidence + figure: `dcapGlobalLearn/`.)

### [STRONG, MORE EXPECTED] End-to-end multi-step learning with a width-independent tile
The same oscillator optimizer, lifted from one gate to a whole circuit's parameter vector,
learns composed circuits **end-to-end from one circuit-level reward**:

- **3-input XOR (parity)** — forces a 2-layer circuit (a 2-input XOR cannot be one
  neuron), both gates + interface scales learned jointly → 1.0;
- **full adder** (5 gates) → 1.0;
- **ripple adder learned all at once** via **parameter sharing**: all XORs share one
  `(w, band)`, all ANDs share, all ORs share, plus three interface scales = **12
  parameters regardless of width**. Learned on 4-bit → **exhaustively correct, 256/256**;
  the *same 12 numbers* dropped into an 8-bit adder handle every carry-propagate case;
  direct 8-bit learning also succeeds.

The tile is **width-independent**: learn it small, deploy at any width. (Evidence + figure:
`dcapMultistep/`.)

### [FOUNDATIONAL] Single-neuron dendritic logic
A single dCaAP computes all six 2-input gates — including the inverters (NOT/NAND/NOR) via
tonic-baseline *suppression* — and, crucially, **XOR / negative-patterning in one neuron**,
which a monotonic point neuron cannot. Composed and cascaded into a half-adder and an 8-bit
ripple adder, **proven correct over all 65,536 input pairs** and verified spiking on every
carry-propagate case. (`dcapLogicGates/`, `dcapNegativePatterning/`, `dcap8bitAdder/`.)

---

## 2. The rigor result that makes the rest trustworthy

Midway, the "trained adder" claim collapsed under audit:
- a frozen network (no learning, random weights) already scored AND 1.00 and OR 1.00 — the
  hand-set bands pre-solved them; only XOR genuinely moved (0.65 → 1.0);
- the trained weights were never plugged into the verified adder — correctness was inferred,
  not checked;
- from a genuinely broken (silent) init, reward-only STDP could not recover at all.

This was documented, not hidden (`dcap8bitAdder/LEARNING_AUDIT.md`), and the pipeline was
rebuilt so training is genuinely connected to the 65,536/65,536-verified circuit. The
negative STDP result and the positive ESC result are both **more credible because that
audit happened**.

---

## 3. Honest boundaries (so "groundbreaking" stays calibrated)

- **The learning that works is gradient-free reward optimization** (extremum-seeking /
  annealing / restart), **not spike-timing plasticity.** The STDP story is a *negative*
  result: we showed *why* Hebbian rules cannot do it, plus a plausible-but-**unimplemented**
  biological mapping for the ESC rule.
- **Topology is hand-given.** Only continuous parameters (weights, bands, interface scales)
  are learned; the wiring is specified.
- **Parameter sharing is a modeling choice** exploiting the adder's regularity — not a claim
  that 120 *independent* gate parameters were learned from one scalar reward.
- **8-bit learned-adder correctness** rests on 4-bit exhaustive (256/256) + the
  width-independent tile + large carry-propagate samples. The full 65,536 *spiking* sweep is
  verified only for the **hand-built** adder.
- Several individual ingredients (ESC for deadbands, non-monotonic credit-assignment
  difficulty, weight tiling) exist in the literature in other contexts. The contribution is
  their **application to dCaAP dendritic logic** and the **unification** in §0 — not
  inventing them from scratch.

---

## 4. The single most defensible "this is new" claim

> **The non-monotonic dendritic band reverses reward-credit assignment in a gate-specific
> way, which makes Hebbian learning provably insufficient and makes an oscillatory
> extremum-seeking rule the natural — and biologically suggestive — fix.**

Everything else in the project is scaffolding around, or consequences of, that statement.

---

## 5. Map of the evidence

| topic | folder |
|---|---|
| Two-compartment dCaAP f–I (non-monotonic) | `dcap_two_compartment/` |
| Negative patterning / single-neuron XOR | `dcapNegativePatterning/` |
| Homeostatic cap (closed-loop fixed point) | `florian2/` |
| All six gates on one dCaAP (suppression inverters) | `dcapLogicGates/` |
| Compose / cascade gates, half-adder | `dcap-logic-circuits.skill/` |
| 8-bit adder, proven 65,536/65,536 | `dcap8bitAdder/` |
| Learning audit + genuine reward learning | `dcap8bitAdder/LEARNING_AUDIT.md` |
| Reward-structure study (Hebbian frontier) | `dcapRStdpRewards/` |
| Oscillator global learner (AND from any start) | `dcapGlobalLearn/` |
| Multi-step circuits + all-at-once adder | `dcapMultistep/` |
| Concept overview / early experiments | `dcaap.md` |
| Installable package | `snn2/` |
