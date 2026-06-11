# Learning logic gates with reward-modulated STDP — and dendritic XOR

A spiking neural network that **starts knowing nothing**, then learns Boolean
logic gates in order of difficulty, and finally solves **XOR** with a single
neuron using the dendritic mechanism described in Gidon et al. 2020, *“Dendritic
action potentials and computation in human layer 2/3 cortical neurons”* (Science
367:83–87).

This document explains the task, the encoding, the learning rule, every result,
*why* each result happens, the two non-obvious pitfalls that had to be solved,
and how to run and extend it.

- Module: `snn2/logic_gates.py`
- Example: `examples/logic_gates.py`
- Run validation: `python -m snn2.logic_gates`
- Generate figure: `python -c "from snn2.logic_gates import make_figure; make_figure('logic_gates.png')"`

---

## 1. The one-paragraph version

One output neuron sees the four input cases of two binary operands `a, b` and is
rewarded when its spiking output matches a target gate. Trained from random
weights by reward-modulated STDP, an ordinary (monotonic) point neuron learns
**OR** instantly, **stalls on AND** at ~0.75, and is **stuck at chance on XOR**
(~0.5) because XOR is not linearly separable. Swapping *only* the neuron’s
activation for a non-monotonic **dCaAP (dendritic calcium spike)** unit — which
fires for *threshold-level* input and is *suppressed* for stronger input — lets a
single neuron **learn XOR to ~1.0**. The synapses, encoding, reward, and learning
rule are identical between the failing and succeeding XOR runs; the dendrite is
the only change.

---

## 2. The task and encoding

**Operands → spikes.** The two operands `a, b ∈ {0,1}` are rate-coded by two
groups of input neurons (20 each). Group A fires (at `operand_rate`, default 0.5
per step) iff `a = 1`; group B fires iff `b = 1`. So each trial presents one of
four input patterns, indexed by `combo = 2a + b`:

| combo | a b | group A | group B |
|:-----:|:---:|:-------:|:-------:|
|  0    | 0 0 | silent  | silent  |
|  1    | 0 1 | silent  | firing  |
|  2    | 1 0 | firing  | silent  |
|  3    | 1 1 | firing  | firing  |

**Output → action.** The single output neuron is run for `processing_time` steps.
Its mean firing rate over the window is thresholded (`action_threshold`) into a
binary action: 1 = “fire”, 0 = “stay silent”.

**Reward.** Action is compared to the target gate’s truth table; reward is `+1` on
a match and `punish_mult` on a mismatch (per-trial).

**Truth tables** (indexed by `combo`, in `snn2/spec.py:GATES`):

```
OR  = [0, 1, 1, 1]      AND = [0, 0, 0, 1]      XOR = [0, 1, 1, 0]
NAND= [1, 1, 1, 0]      NOR = [1, 0, 0, 0]
```

> **Why only OR / AND / XOR are used here.** NAND and NOR require the neuron to
> fire when *no* operand is present (`combo 00 → 1`). With purely excitatory
> inputs there is no drive at `00`, so firing there is impossible without a tonic
> “bias” input. They are left out to keep the demo honest and minimal.

---

## 3. The learning rule

Weights are updated by **reward-modulated STDP with a per-trial eligibility
trace** (the engine in `snn2/engine.py`). Within a trial, pre→post coincidences
accumulate an eligibility tensor; at trial end the weights move by

```
ΔW = learning_rate · reward · eligibility        (then clipped to [0, max_weight])
```

Two engine constraints matter for this task:

- **Feedforward only.** After each update the body→body weight is forced to zero
  (`W[:, n_in:, :] = 0`). Without this, the output neuron can build a recurrent
  self-excitation loop and run away out of any useful operating regime.
- **Per-trial credit.** Eligibility is scoped to the current trial and scaled by
  *that* trial’s reward. A single trace carried across trials misassigns reward
  to the next trial’s activity and collapses learning to chance — this is the
  same credit-assignment fix used in the instrumental-conditioning demo.

---

## 4. The curriculum, result by result

All numbers are mean final accuracy over 12 seeds (200-trial window), reproduced
by `python -m snn2.logic_gates`.

### Stage 1 — OR (monotonic neuron): **1.00, solved immediately**

OR is the easiest gate: “fire if any input is present.” A low firing threshold
already satisfies it, so near-random initial weights are essentially correct from
trial 0 and the curve sits at 1.0. There is little to *learn* — OR is the trivial
warm-up that proves the plumbing (encoding, reward, readout) is wired correctly.

### Stage 2 — AND (monotonic neuron): **~0.75, a genuine ceiling**

AND needs *coincidence detection*: fire for two inputs, **not** for one. For a
monotonic point neuron the drive from two inputs is always exactly twice the
drive from one input, so success requires a firing threshold wedged into that
narrow 1×–2× window. With a noisy rate code and reward-modulated STDP, the neuron
cannot reliably place and hold the threshold there: the reward for the lone
“fire on 11” case (25% of trials) is too sparse to overcome the punishment from
single-input firing, so the weights settle low and the neuron learns to **stay
silent**. Silence is correct on `00, 01, 10` and wrong only on `11` → **0.75**.

This is not a bug to be tuned away — it was confirmed across a dozen
configurations (threshold, gain, punishment strength, both ascending and
descending weight initializations, sharper input rates). It is an honest
illustration that *even a linearly separable gate can exceed what a single
rate-coded point neuron learns with simple R-STDP.*

### Stage 3 — XOR (monotonic neuron): **~0.5, the wall**

XOR (“fire for exactly one input”) is **not linearly separable**. No single
threshold on a monotonic drive can fire for one input yet stay silent for two,
because two inputs always drive *harder* than one. The point neuron is pinned at
chance. This is the classic limitation that historically motivated hidden layers.

### Stage 4 — XOR (dendritic dCaAP neuron): **~1.00, 16/16 seeds, solved**

Here we change **only the neuron**. Gidon et al. (2020) found that human layer 2/3
pyramidal dendrites generate calcium action potentials (dCaAPs) whose amplitude is
**maximal for threshold-level input and decreases for stronger input** — a
band-pass / *anti-coincidence* activation, the opposite of a normal monotonic
neuron. A single dendrite with this property computes XOR directly:

- `00` — no drive → below the band → **silent** (correct, target 0)
- `01 / 10` — one input → drive lands **in** the band → **fires** (correct, target 1)
- `11` — two inputs → drive **overshoots** above the band → **suppressed** (correct, target 0)

Our model (`snn2/parts.py:dendritic`) is memoryless: the neuron spikes on any step
whose instantaneous drive `I` lies in a band `[dcap_lo, dcap_hi]`. Learning’s only
job is to scale the synapses so that the one-input drive sits inside the band while
the two-input drive (always 2×) clears it. R-STDP does this, and the trained
neuron’s truth table is exactly `[0, 1, 1, 0]`. It starts at ~0.83 (the dendrite
has an XOR-favorable inductive bias, so it begins above chance) and climbs to ~1.0.

**The headline contrast:** identical task, encoding, synapses, and learning rule;
monotonic neuron 0.49, dendritic neuron 1.00.

---

## 5. Why the dendrite works — the firing map

The bottom-left panel of `logic_gates.png` plots dendritic firing rate against
synaptic scale for each input case. Reading it left to right:

- The **two-input** curve (red) enters the band first (at low weight, because its
  drive is 2× larger), peaks, then **exits above** the band as weight grows.
- The **one-input** curve (green) enters the band later and peaks where two-input
  has already been suppressed.
- The shaded **XOR window** is the range of weights where *one input fires and two
  inputs do not*. Learning’s job is to land the weight in that window.

The “no input” curve stays at zero throughout — `00` is never in the band.

---

## 6. Two pitfalls that had to be solved (and how)

These are the non-obvious failures that the validation methodology surfaced; both
are baked into the `logic_dendritic` preset.

### Pitfall A — the band-entry-order trap (weight collapse)

With pure potentiation **and** punishment, the dendritic XOR collapses to chance
and the weights go to **zero**. Why: at low initial weight the *two-input* drive
is the largest and is the first to enter the band, so `11` fires — and, being the
wrong answer, gets **punished**, which drives *all* weights down (both groups are
active on `11`). The punishment crushes the weights to zero before the one-input
drive can ever climb into the band. You can see this trap directly in the firing
map: the red (two-input) curve owns the low-weight region.

**Fix:** reward-only learning (`punish_mult = 0`). Wrong-but-firing trials then
cannot crush the weights; only correct one-input firing moves them, ratcheting the
weight *up* toward the band. A `max_weight` cap inside the XOR window stops the
ratchet from overshooting out the top.

### Pitfall B — the dead-start trap (no exploration)

A deterministic threshold neuron initialized fully sub-threshold never fires, so
there is no post-synaptic spike, so eligibility is zero, so **nothing can learn**.
The same is true at the band’s edge if drive never reaches it.

**Fix:** initialize the operating point near the relevant boundary so the
stochastic rate code produces occasional exploratory spikes that learning can
reinforce. For the monotonic gates this means setting the gain so the initial
single-input drive sits around threshold; for the dendrite it means the initial
one-input drive sits near the band.

> Both pitfalls are exactly the kind of thing the “validation ladder”
> (see the `neural-model-validation` skill) is meant to catch: the equations
> *ran* fine in every case — it was the *learning behavior* that was wrong, and
> only property-style checks (does it converge? do weights stay sane? does the
> truth table match?) expose that.

---

## 7. The validated claims

`snn2/logic_gates.py:validate()` asserts all of the following (all pass):

1. **Starts not knowing** — the dendritic XOR learner is below mastery at the
   start (start < 0.92).
2. **OR is learned** — monotonic OR final > 0.9.
3. **AND plateaus** — monotonic AND final in (0.6, 0.85): learned-ish, not solved.
4. **XOR fails on a point neuron** — monotonic XOR final < 0.8 (the wall).
5. **XOR is solved on a dendrite** — dendritic XOR final > 0.9.
6. **Robust** — dendritic XOR converges (>0.9) for ≥80% of seeds.
7. **Dendrite beats point neuron on XOR by > 0.3.**
8. **Learning happened** — dendritic XOR end > start + 0.05.
9. **Exact truth table** — the trained dendritic neuron implements `[0,1,1,0]`.

---

## 8. Configuration reference

The two presets live in `snn2/spec.py`. Override any field per run, e.g.
`expand({"preset": "logic_dendritic", "target_map": GATES["XOR"], "lr": 0.05})`.

**`logic_monotonic`** (point neuron, gates OR / AND / XOR)

| field | value | role |
|---|---|---|
| `neuron` | `lif` | monotonic leaky integrate-and-fire |
| `firing_threshold` | 10.0 | spike threshold |
| `input_gain` | 0.8 | scales drive; set so init ≈ threshold (exploration) |
| `max_weight` | 0.5 | weight ceiling |
| `lr` | 0.025 | learning rate |
| `punish_mult` | −0.5 | needed so AND/XOR can suppress wrong firing |
| `action_threshold` | 0.05 | rate→action cutoff |
| `operand_rate` | 0.5 | input firing prob when an operand is on |

**`logic_dendritic`** (dCaAP neuron, gate XOR)

| field | value | role |
|---|---|---|
| `neuron` | `dendritic` | non-monotonic band-pass (dCaAP) |
| `dcap_lo, dcap_hi` | 3.0, 6.0 | the dendritic-spike band |
| `input_gain` | 1.0 | places one-input drive near the band |
| `max_weight` | 0.45 | caps the weight inside the XOR window |
| `lr` | 0.04 | learning rate |
| `punish_mult` | **0.0** | reward-only — defeats the band-entry-order trap |
| `action_threshold` | 0.1 | rate→action cutoff |

---

## 9. How to run

```bash
# validation (prints the curriculum table + 9 checks)
python -m snn2.logic_gates

# the figure (4-panel: curves, truth tables, firing map, contrast bar)
python -c "from snn2.logic_gates import make_figure; make_figure('logic_gates.png')"

# a single gate, programmatically
python - << 'PY'
from snn2.logic_gates import train_gate
curve, final, W = train_gate("logic_dendritic", "XOR", n_seeds=8)
print("final per seed:", final.round(2))
PY
```

---

## 10. Extending it

- **Other gates.** Pass any `target_map` from `GATES` (or your own length-4 list)
  to `train_gate`. Excitatory-only inputs cannot do gates that fire at `00`
  (NAND/NOR) without adding a tonic bias group.
- **Learn the band.** Here the dCaAP band is fixed and the synapses learn; a
  richer demo would make `dcap_lo/hi` plastic or learn a bias.
- **Make AND learnable.** Add a learnable bias input (a tonic group), giving the
  neuron a learnable effective threshold — the missing degree of freedom that a
  single excitatory point neuron lacks for AND.
- **Multi-gate neuron.** With a band where `hi ≥ 2·lo`, a *single* fixed-band
  dendrite can realize OR, XOR, or AND purely by the learned weight scale (small
  scale → both levels in band = OR; medium → one in band = XOR; large → only the
  doubled level in band = AND). A nice follow-on experiment.

---

## 11. Reference

Gidon, A., Zolnik, T. A., Fidzinski, P., Bolduan, F., Papoutsi, A., Poirazi, P.,
Holtkamp, M., Vida, I., & Larkum, M. E. (2020). Dendritic action potentials and
computation in human layer 2/3 cortical neurons. *Science*, 367(6473), 83–87.
