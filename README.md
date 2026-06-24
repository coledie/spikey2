# snn2

A redesign of [Spikey](https://github.com/SpikeyCNS/spikey) around two goals:

- **(a) fast *and* provably correct** — the RL-STDP rule is a pure function pinned
  to a reference oracle; speed comes from batching, trust comes from validation.
- **(b) light and LLM-friendly** — an experiment is a three-line dict,
  not a class hierarchy.

It keeps Spikey's faithful semantics (LIF + reward-modulated STDP, rate-coded
inputs, the random-state task) but changes *how you express and run* experiments.

## Install

```bash
pip install -e .            # core (numpy only)
pip install -e ".[scale]"   # + ray[tune] for multi-core / async search
```

## 60-second tour

```python
import snn2

# one experiment — the whole spec is three meaningful keys
m = snn2.run({"preset": "izhi_randstate", "lr": 0.1, "stdp_window": 100})
print(m["final_reward"], m["mean_out_rate"])

# a sweep is just data
specs = snn2.sweep({"preset": "izhi_randstate"},
                   {"lr": [0.0, 0.05, 0.1, 0.3], "stdp_window": [50, 100, 200]})
results = snn2.schedule(specs)          # 12 runs, one batched bucket, deduped
best = max(results.values(), key=lambda r: r["final_reward"])
```

Validate the learning rule any time:

```bash
python -m snn2.validate
```

## Why it's shaped this way

### An experiment is a spec, not a class

Spikey expresses an experiment as a `network_template(RLNetwork)` subclass with
`parts` + `keys` dicts, plus scattered subclasses and a hand-written `train_func`.
Here it's a flat dict. That single change is what makes everything else possible:
a dict is data an LLM can *emit* (not author), it's hashable (so results dedupe and
resume), and it's diffable (so a sweep is a cartesian product of deltas).

Presets carry the defaults; your spec carries only the deltas. The 18-key config
from the original notebook collapses to:

```python
{"preset": "izhi_randstate", "lr": 0.1, "stdp_window": 100}
```

The model genuinely has many degrees of freedom — you don't lose any. You just
stop *writing* the ones you don't care about. The fully-resolved config (preset
expanded + defaults filled + derived keys) is what actually runs and what gets
hashed, so reproducibility stays honest even though the input was tiny. See
`snn2.expand(spec)`.

### Parts are pure functions in a registry

A neuron/encoder/readout/reward model is a function registered under a string.
Adding one is a single `@snn2.register(...)` — no inheritance:

```python
@snn2.register("neuron", "izhikevich")
def izhikevich(state, I, p, rng):
    a, b, c, d = p["izhi_a"], p["izhi_b"], p["izhi_c"], p["izhi_d"]
    v, u = state["v"], state["u"]
    for _ in range(2):                       # two 0.5 ms Euler sub-steps
        v = v + 0.5 * (0.04*v*v + 5*v + 140 - u + I)
    u = u + a * (b*v - u)
    fired = v >= 30.0
    state["v"] = np.where(fired, c, v)
    state["u"] = np.where(fired, u + d, u)
    return state, fired
```

> Note: the original repo's "Izhikevich" is a *paper-replication notebook* that
> actually runs the generic LIF neuron. `snn2` ships the real a/b/c/d
> recovery-variable model as a first-class part.

### The engine batches over experiments, not just neurons

Spikey vectorizes over neurons with NumPy. `snn2` adds a leading **experiment
axis `B`**: weights `[B, S, N]`, potentials `[B, N]`, traces `[B]`. One
`run_bucket` advances *every* experiment with the same array ops. The only Python
loops are over **time** — never over experiments or neurons.

Logically-staggered and uneven episodes share one loop via an `active` mask `[B]`:
a lane that has reached its own `len_episode` stops accruing reward and weight
updates while its neighbors keep going. So you can hand each lane a different
episode budget and still run them together.

Specs that differ in *shape* (different `n_neurons`/`n_inputs`) can't share a
tensor, so they're auto-bucketed by shape and each bucket runs batched. Padding
would waste the most compute on the biggest nets; bucketing avoids that.

### The learning rule is validated, not just shape-checked

`snn2/stdp.py` defines the reward-modulated LTP update three ways:

- `stdp_delta_ref` — an obviously-correct quadruple loop (the **oracle**),
- `stdp_delta` — vectorized single-experiment,
- `stdp_delta_batched` — vectorized over `B` (the hot path).

`snn2/validate.py` asserts the fast paths equal the oracle byte-for-byte on
random inputs, plus analytic properties: pre-before-post potentiates,
post-only/zero-trace produce no change, recency credit is monotonic. This is the
concrete meaning of *"RLSTDP has to work perfectly and be validatable."*

### Scheduling: Ray on top, batching underneath

Two scheduling problems, two tools:

| level | what it decides | tool |
|------|-----------------|------|
| across workers (coarse) | which specs run where, refill capacity, early-stop losers | **Ray Tune** (async ASHA) |
| within a worker (fine) | many lanes in lockstep on one core | **the batched engine** |

```python
results  = snn2.schedule_ray(specs)               # one process per shape-bucket
analysis = snn2.tune_run(search_space, 500)        # staggered async trials + early-stop
```

Ray gives real multi-core parallelism because each worker is a separate **process
with its own interpreter** — the CPython GIL is per-interpreter, so N workers =
N GILs. And inside a worker the heavy NumPy ops release the GIL anyway. The thing
to avoid isn't the GIL, it's a Python loop over experiments/neurons; batching
keeps those on array axes so the interpreter time stays negligible. Ray is
entirely optional: the core runs on NumPy alone.

## Layout

```
snn2/                   the package (pip install -e .)
  spec.py       presets, expand()->resolved, content-addressed hashing; GATES
  registry.py   string -> pure-function parts (neurons, inputs, ..., engines)
  parts.py      lif, izhikevich, dendritic(dCaAP), ratemap, threshold/population, ...
  stdp.py       stdp_delta (+ _batched) and stdp_delta_ref oracle
  engines/      engine variants, each registered by name
    batched.py    run_bucket for STATEFUL games (reset/step) -- engine="batched"
    trial.py      run_bucket for TRIAL games (cue sequences) -- engine="trial"
  api.py        run / sweep / schedule / schedule_ray / tune_run (dispatches by engine)
  validate.py   golden test vs oracle + analytic property tests
  florian.py    Florian (2007) MSTDPET, validated + figure
docs/
  usage.md         how an LLM drives this by writing specs only
  dcaap.md / dcaap-findings.md   dCaAP background + research synthesis
examples/
  izh/izhikevich.py        Izhikevich (2003) random-state task (batched engine)
  florian/florian_fig1.py  Florian (2007) figure
  izh/conditioning.py      instrumental GO/NO-GO, grouping validation (trial engine)
  logicgates/base/logic_gates.py  R-STDP gate curriculum + dendritic XOR (trial engine)
```

The engine is selected by the spec's `engine` key (default `"batched"`); it is
looked up in the registry exactly like any other part, so adding a third engine
is one `@register("engine", "name")` in `snn2/engines/`.


## Learning logic gates with R-STDP, and dendritic XOR (Gidon et al. 2020)

A single output neuron is shown the four cases of two binary operands and rewarded
when its spiking matches a target gate. Trained from random weights, an ordinary
**point neuron** learns **OR** instantly, **stalls on AND** (~0.75), and is
**stuck at chance on XOR** (~0.5) — XOR is not linearly separable. Swapping *only*
the neuron's activation for a non-monotonic **dCaAP** unit (fires for
threshold-level input, *suppressed* for stronger input, per Gidon et al. 2020)
lets a single neuron **learn XOR to ~1.0**.

```bash
python -m examples.logicgates.base.logic_gates   # curriculum table + 9 validation checks
python examples/logicgates/base/logic_gates.py   # also renders logic_gates.png
```

![logic gates and dendritic XOR](logic_gates.png)

- **Curves (top-left).** The point neuron degrades with difficulty (OR 1.0 → AND
  0.75 → XOR 0.5); the dendritic neuron climbs from ~0.83 to ~1.0 on XOR.
- **Truth tables (top-right).** Red cells are the failures: AND's `11`,
  monotonic-XOR's `01/10`. The dendritic XOR row is exactly `[0,1,1,0]`.
- **Firing map (bottom-left).** *Why* the dendrite works: there is a window of
  synaptic scale where one input lands in the dCaAP band and two inputs overshoot
  it. The red two-input curve owning the low-weight region is the
  "band-entry-order trap" (see docs).
- **Contrast (bottom-right).** Same task, same rule: point neuron 0.49, dendrite
  1.00.

Full write-up — encoding, learning rule, every result explained, the two pitfalls
(band-entry-order collapse and the dead-start trap), config reference, and how to
extend — is in **`docs/logic_gates.md`**.

## Why reward-modulated STDP works: Farries & Fairhall (2007)

The gate demos *use* reward-modulated STDP; this one shows **why it learns at
all**. Farries & Fairhall (2007) proved that a reward signal modulating STDP makes
the trial-averaged weight change climb the gradient of **expected reward** — the
synaptic eligibility trace acts as the policy-gradient score function — *provided
the reward is centered on a baseline*. We reproduce that and validate it against an
exact oracle: the finite-difference gradient of `E[reward]`.

```bash
python -m snn2.farries          # alignment cosines + 5 validation checks
python examples/farries.py      # also renders farries_fig.png
```

![Farries & Fairhall: R-STDP is gradient ascent on reward](farries_fig.png)

- **Centered R-STDP** update lies exactly on the reward-gradient diagonal
  (cosine 1.00); the **uncentered** update points the *opposite* way (cosine −1.00)
  — the unsupervised Hebbian bias drives an over-firing neuron to fire *more*.
- Centered learning climbs `E[R]`; a mis-set baseline collapses it. Performance
  peaks sharply at the true baseline.

This is the theoretical backbone under the conditioning and logic-gate demos. Full
derivation and method in **`docs/farries.md`**.

## Where the dopamine signal comes from: Chorley & Seth (2011)

Florian and Farries & Fairhall *consume* a dopamine/reward signal. Chorley & Seth
model **where it comes from** — a dual-pathway circuit (fast SEN→INT→DA excitation
racing a slow PFC→STR→DA inhibition, both under DA-modulated plasticity) in which
the canonical dopamine shift from reward (US) to reward-predicting cue (CS) emerges.

```bash
python -m snn2.chorley_seth         # 6 validation checks + summary
python examples/chorley_seth.py     # also renders chorley_seth_fig.png
```

![Chorley & Seth: dual-pathway DA reward-prediction](chorley_seth_fig.png)

- The **CS response emerges** (DA learns to fire to the cue) via DA-STDP and the
  distal-reward eligibility mechanism — peri-event DA goes from a single US peak to
  both a CS and a US peak.
- The **inhibitory pathway** develops anticipatory STR firing near the US.

This is a faithful *reduced-scale* reproduction: the cue-response shift is robust;
full *cancellation* of the US response (the sharply-timed inhibitory volley) needs
the paper's larger polychronous substrate and is partial here — documented honestly
in **`docs/chorley_seth.md`**.

## Validating that grouping is correct (instrumental conditioning)

The whole speed story rests on **grouping**: many experiments share one batched
run on a leading axis `B`. That is only legitimate if the lanes stay independent
— lane *i*'s neurons, weights, and reward must never touch lane *j*. Instrumental
(operant) conditioning is the ideal stress test, because if grouping leaked, an
agent could not learn a contingency that differs from its batch-mates.

**The task — GO / NO-GO.** Each trial presents one of two cues. A single output
neuron must fire for the GO cue and stay silent for the NO-GO cue; reward is +1
when the action matches the cue's target and −1 otherwise. The two cues reach the
neuron through *separate* input groups, so reward-modulated STDP tunes each on its
own — GO weights ratchet up from rewarded exploratory firing, NO-GO weights
ratchet down from punished firing — and the agent learns the contingency.

```bash
python examples/izh/conditioning.py     # validate + render grouping_conditioning.png
```

![grouping validated via instrumental conditioning](grouping_conditioning.png)

The three panels are exactly the validation:

1. **Instrumental conditioning (left).** 16 agents trained with R-STDP climb from
   chance (0.5) to 100% accuracy; the green band is the 25–75th percentile across
   agents. The matched `lr=0` control (grey, dashed) stays at chance — so the rise
   is learning, not drift. 16/16 seeds converge.
2. **Grouping independence (middle).** Each point is one spec's reward when run
   **alone** (x) versus **grouped** in a single batched bucket (y). Every point
   lies exactly on `y = x` — the per-lane weights and reward are **bit-identical**
   (max difference `0e+00`). Lanes are seeded from their spec, not their batch
   position, so a result never depends on who shares the batch (also verified by a
   shuffle-invariance check).
3. **Cross-contingency independence (right).** Two agents with **opposite**
   contingencies (GO=cue0 vs GO=cue1) trained in the *same* batch each reach 1.00
   on their own task. Neighboring lanes with conflicting goals do not interfere —
   the strongest evidence the grouping is real.

`snn2.conditioning.validate()` asserts all of this (independence == 0, shuffle
invariance, both cross-task agents > 0.9, ≥14/16 seeds converge).

## Reproducing the Izhikevich (2003) firing patterns

`snn2/izhikevich.py` reproduces the famous figure where one simple `a,b,c,d`
model yields every cortical firing type. The traces are produced by the *same
registered `izhikevich` neuron the batched engine uses* (stepped at B=1, N=1),
so the figure validates the model the rest of snn2 relies on.

```bash
python examples/izhikevich_fig1.py   # validate + render izhikevich_firing_patterns.png
```

Validated by the behaviors that define each type, not just by plotting: every
regime fires under the step current; fast-spiking fires the most; regular-spiking
shows spike-frequency adaptation (growing inter-spike intervals); intrinsically-
bursting opens with a burst then slows; chattering shows recurring short-ISI
bursts while regular-spiking shows none.

> The repo's `izhikevich2007.ipynb` is a different thing — an RL *learning task*
> that runs the LIF neuron. That task is the `izhi_randstate` preset / the
> `examples/izh/izhikevich.py` sweep. This figure is the neuron-dynamics
> reproduction, using the true recovery-variable model the repo lacks.

## Reproducing Florian (2007) Figure 1

`snn2/florian.py` reproduces the MSTDPET demonstration — a single synapse j→i
driven with fixed pre/post spike trains and a reward that flips +1→−1 at 100 ms.
It implements the paper's exact eligibility-trace equations (Eqs 7, 8, 42–44),
not the window approximation, so it matches the reference curve directly.

```bash
python examples/florian/florian_fig1.py     # validate + render florian_fig1.png
```

It is validated, not just plotted: the module's weight curve is asserted equal,
byte-for-byte, to an independent inlined transcription of the equations
(w: 0.200 → 0.316, peak 0.416), plus the qualitative facts — pre-before-post
gives positive eligibility, weight potentiates under positive reward, and the
same eligibility reduces the weight once reward turns negative.
