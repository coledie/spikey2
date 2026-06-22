# dCaAPs in snn2 — concept and experiments

A working reference for the dendritic calcium action potential (dCaAP) and the
experiments built around it in the `snn2` framework.

---

## 1. What a dCaAP is

Gidon et al. (2020), *"Dendritic action potentials and computation in human layer 2/3
cortical neurons"* (Science 367:83–87), reported a dendritic Ca²⁺ action potential in
human cortical neurons with an unusual property: its **amplitude is maximal for
threshold-level input and *decreases* for stronger input**. The activation is
**band-pass / non-monotonic** — the opposite of a normal neuron, whose response grows
(then saturates) with input.

The headline computational consequence: a **single neuron can compute XOR**. Fire for
moderate drive (one input), stay silent for weak drive (no input) *and* for strong
drive (two inputs). No point neuron — no single `(a,b,c,d)` or LIF unit — can do this,
because for a point neuron more input always means more (or equal) output.

The dCaAP is therefore best thought of as a tunable **"moderate-drive / exactly-one"
detector**, and more generally an **"exactly-k-of-n" detector**.

---

## 2. How it's modeled here

Two complementary models live in the package:

**(a) Memoryless band-pass unit** (`snn2/parts.py`, neuron `"dendritic"`).
Fires on a step iff the instantaneous drive `I` falls in a band `[dcap_lo, dcap_hi]`.
Cheap, vectorized, used inside the learning experiments (logic gates, negative
patterning). It captures the *computational* signature without dynamics.

**(b) Two-compartment neuron** (`snn2/dcap_neuron.py`).
A dendritic compartment produces a dCaAP whose amplitude is a non-monotonic ("hump")
function of dendritic drive (ON above a low threshold, OFF again above a high one),
low-pass filtered by a Ca-gate time constant; it injects current into a **somatic
Izhikevich (2003) RS** compartment that produces the Na⁺ output spikes. This is the
biophysically-flavored version, used to reproduce the f-I signature.

---

## 3. Experiments

### 3.1 Two-compartment f-I curve  → `dcap_two_compartment/`
Reproduces the Gidon signature as a **non-monotonic f-I curve**.

| measurement | result |
|---|---|
| dCaAP f-I peak | **41 Hz at drive ≈ 6.5** (interior) |
| strong-drive firing | **0 Hz** (dCaAP suppressed) |
| comparison RS neuron | monotonic, saturating (~39 Hz) |

The somatic rate rises, peaks at threshold-level drive, then falls — next to a
standard RS neuron that only ever increases. This is the shared primitive behind the
other two experiments. *5/5 validation checks pass.*

### 3.2 Dendritic XOR + the band-entry-order trap  → `snn2/logic_gates.py`
In an R-STDP gate-learning curriculum, a monotonic neuron hits the expected walls
(AND ceilings at 0.75, XOR stuck at 0.50), while a **dCaAP neuron learns XOR to 1.00
(16/16 seeds)**. Getting there exposed the **band-entry-order trap**: with a punish
signal and no weight cap, the two-input pattern over-fires, drives weights down in the
wrong order, and the network locks into a bad equilibrium. Fixes that work:
`punish_mult = 0`, a weight cap inside the XOR window, and feedforward-only updates.
This trap reappears below as the flip side of the homeostatic cap.

### 3.3 Negative patterning + k-of-n gates  → `dcapNegativePatterning/`
**Negative patterning** (A+, B+, AB−) is a classic configural-learning paradigm and is
formally XOR. A single dCaAP neuron solves it; an elemental (monotonic) learner can't.

| claim | result |
|---|---|
| dCaAP learns negative patterning | accuracy **1.00** |
| monotonic learner | accuracy **0.50** (chance) |
| dCaAP compound response | AB / single-cue = **0.02** (suppressed) |
| elemental compound response | AB / single-cue = **1.73** (**summation**) |
| exactly-1-of-3 detector | ✓ |
| exactly-2-of-3 detector | ✓ |

The "summation vs suppression" contrast is the heart of the phenomenon: the elemental
learner responds to the compound *more* than to a single cue (wrong), the dCaAP
suppresses it (right). The second figure shows the **k-of-n** generalization — one
dCaAP neuron tuned by its weight scale into AND/XOR (2 inputs) or exactly-1/exactly-2
(3 inputs), gates a point neuron cannot realize. *6/6 validation checks pass.*

### 3.4 florian2 — the homeostatic cap  → `florian2/`
Florian (2007) MSTDPET (reward-modulated STDP with eligibility traces) run in **closed
loop**: the postsynaptic neuron's firing depends on the weight being learned.

| postsynaptic neuron | outcome |
|---|---|
| monotonic | weight **runs away** to the ceiling (≈1.97–1.99 of 2.0) |
| dCaAP | **stable interior fixed point** at **w\* ≈ 1.25** (drive ≈ 6.25, band edge) |

All four initial weights converge to the dCaAP fixed point; the phase portrait shows
`dw/dt` crossing zero with negative slope (dCaAP) versus staying positive (monotonic).
*5/5 validation checks pass.*

---

## 4. The unifying object: a sign-flip in dw/dt

The single most useful idea to come out of these experiments is that **the trap and the
cap are the same geometric object** — `dw/dt` changing sign as a function of the weight,
a direct consequence of the dCaAP's non-monotonic transfer function.

**Homeostatic cap (benign face).** A monotonic neuron under Hebbian/reward plasticity is
a positive-feedback loop (more weight → more firing → more coincidence → more weight) and
runs away; the field's homeostatic mechanisms (synaptic scaling, BCM sliding threshold,
normalization, hard bounds) are separate, slower add-ons that fix this. A dCaAP folds the
stabilizer into the transfer function: past the band, more weight means *less* firing, so
the feedback reverses sign and the weight settles at a fixed point. What it means:

- **Homeostasis without a second mechanism** — the nonlinearity that computes also
  regulates.
- **Structural / instantaneous BCM** — same stabilizing effect as a sliding modification
  threshold, but built into the f-I curve and acting on a spike timescale.
- **Set-point, not a wall** — an interior attractor (converged from both sides, dynamic
  range preserved), unlike a weight clip where selectivity piles up against the ceiling.
- **Reward-modulated learning is intrinsically bounded** — eligibility self-extinguishes
  past the band, so positive reward cannot cause runaway. (Directly relevant to RL-STDP
  stability.)

**Band-entry-order trap (hazard face).** With multiple input patterns competing for the
same band (XOR: single-in-band, double-above), naive potentiation can push the wrong
pattern's drive through the band in the wrong order; the same restoring pull then locks a
bad equilibrium. This is why dendritic XOR needed `punish_mult=0` and a weight cap.

**Control knob.** Both faces are governed by where the band sits relative to the input
drive statistics. "Design the band relative to the input distribution" is the single
lever for harnessing the cap and avoiding the trap.

---

## 5. Honest limitations

- The dCaAP models are **phenomenological** (band-pass activation), not full HH Ca²⁺
  channel kinetics. They reproduce the computational signature (non-monotonic f-I, XOR),
  not the detailed conductance waveforms. The soma in the two-compartment model *is* a
  faithful Izhikevich RS unit.
- The florian2 fixed point is a true two-sided attractor only with a small homeostatic
  **weight-decay** term. Without it the dCaAP still self-limits (growth stops past the
  band) but the equilibrium is a soft ceiling rather than a sharp attractor.
- The cap regulates a neuron's **total drive**, not representational usefulness. A
  mis-scaled band can self-pin the neuron to silence (drive below band) or to the soft
  ceiling.
- k-of-n panels are a **capability map** (firing vs weight scale), computed directly;
  the 2-input gates are also learned end-to-end via R-STDP.

---

## 6. Files

| path | what |
|---|---|
| `dcap_two_compartment/` | two-compartment neuron, f-I figure, README, runner |
| `dcapNegativePatterning/` | negative patterning + k-of-n, two figures, README, runner |
| `florian2/` | closed-loop MSTDPET homeostatic cap, figure, README, runner |
| `snn2/snn2/dcap_neuron.py` | two-compartment dCaAP neuron |
| `snn2/snn2/dcap_negative_patterning.py` | negative patterning + k-of-n |
| `snn2/snn2/florian2.py` | self-limiting closed-loop MSTDPET |
| `snn2/snn2/parts.py` | the memoryless `"dendritic"` band-pass unit |
| `snn2/snn2/logic_gates.py` | dendritic XOR + the band-entry-order trap |

Run any module: `cd snn2 && PYTHONPATH=. python -m snn2.<module>`
(e.g. `snn2.dcap_neuron`, `snn2.dcap_negative_patterning`, `snn2.florian2`).

---

## 7. Open directions

- **The trap/cap as a paper-worthy object.** It recurs across logic gates (learning
  hazard), Farries (gradient non-convexity), and florian2 (homeostatic cap). A focused
  treatment of "non-monotonic single-neuron plasticity geometry" may stand on its own.
- **Band as learned, not fixed.** Make `[dcap_lo, dcap_hi]` adapt to input statistics —
  a metaplasticity rule that keeps the operating point in-band automatically.
- **Networks of dCaAPs.** Everything so far is single-neuron or single-synapse. The
  k-of-n result suggests layers of dCaAPs as configural feature detectors.

---

## 8. References

- Gidon, A., Zolnik, T. A., Fidzinski, P., Bolduan, F., Papoutsi, A., Poirazi, P.,
  Holtkamp, M., Vida, I., & Larkum, M. E. (2020). Dendritic action potentials and
  computation in human layer 2/3 cortical neurons. *Science, 367*(6473), 83–87.
  https://doi.org/10.1126/science.aax6239
- Izhikevich, E. M. (2003). Simple model of spiking neurons. *IEEE Transactions on
  Neural Networks, 14*(6), 1569–1572. https://doi.org/10.1109/TNN.2003.820440
- Florian, R. V. (2007). Reinforcement learning through modulation of spike-timing-
  dependent synaptic plasticity. *Neural Computation, 19*(6), 1468–1502.
  https://doi.org/10.1162/neco.2007.19.6.1468

