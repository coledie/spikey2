# snn2

A redesign of [Spikey](https://github.com/SpikeyCNS/spikey) around three goals:

- **(a) many experiments at once** — thousands of logically-different, possibly
  staggered runs, scheduled together.
- **(b) fast *and* provably correct** — the RL-STDP rule is a pure function pinned
  to a reference oracle; speed comes from batching, trust comes from validation.
- **(c) light and LLM-/everyone-friendly** — an experiment is a three-line dict,
  not a class hierarchy. No Ray required to run one.

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
snn2/
  spec.py       presets, expand()->resolved, content-addressed hashing
  registry.py   string -> pure-function parts
  parts.py      lif, izhikevich, ratemap, threshold/population, fire_states, randstate
  stdp.py       stdp_delta (+ _batched) and stdp_delta_ref oracle
  engine.py     run_bucket: batched loop with active-mask
  api.py        run / sweep / schedule / schedule_ray / tune_run
  validate.py   golden test vs oracle + analytic property tests
  SKILL.md      how an LLM drives this by writing specs only
examples/
  izhikevich.py the repo's task, redesigned
```

## Reproducing Florian (2007) Figure 1

`snn2/florian.py` reproduces the MSTDPET demonstration — a single synapse j→i
driven with fixed pre/post spike trains and a reward that flips +1→−1 at 100 ms.
It implements the paper's exact eligibility-trace equations (Eqs 7, 8, 42–44),
not the window approximation, so it matches the reference curve directly.

```bash
python examples/florian_fig1.py     # validate + render florian_fig1.png
```

It is validated, not just plotted: the module's weight curve is asserted equal,
byte-for-byte, to an independent inlined transcription of the equations
(w: 0.200 → 0.316, peak 0.416), plus the qualitative facts — pre-before-post
gives positive eligibility, weight potentiates under positive reward, and the
same eligibility reduces the weight once reward turns negative.

## Honest limits

- Same-shape-only batching; differing sizes are bucketed (handled, but more
  buckets = less batch parallelism).
- The bundled games (`randstate`) are minimal; a real CartPole/Gymnasium adapter
  is a registered `game` part you'd add (5-tuple → 4-tuple shim).
- Engine is NumPy; the same pure functions are JAX-`vmap`/`jit`-ready for GPU,
  which is the intended fast path at large `B`.
