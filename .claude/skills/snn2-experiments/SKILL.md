---
name: snn2-experiments
description: >
  How to run, sweep, and extend spiking-neural-net experiments in the `snn2`
  framework -- a spec-driven, batched, validated SNN engine. Use this whenever
  the user wants to run an SNN/RL experiment in this repo, write or tune a spec,
  pick or add an engine (batched vs trial), add a neuron/input/readout/reward
  part, sweep hyperparameters, or understand why a run is silent/saturated.
  Reach for this skill for any task touching `snn2.run`, `snn2.sweep`,
  `snn2.schedule`, presets, the registry, or the `engines/` package.
---

# Running snn2 experiments

You drive `snn2` by writing **specs** (plain dicts) and reading **metrics**
(plain dicts). You never write a class, a training loop, or touch an engine's
hot loop. Everything is selected by string name through the registry.

## Install

The package lives at the repo root as `snn2/`. Install once, editable:

```bash
pip install -e .
```

Then `import snn2` works from anywhere.

## The only functions you call

```python
import snn2
snn2.run(spec) -> metrics                   # one experiment
snn2.sweep(base_spec, grid) -> [spec, ...]  # cartesian product of deltas
snn2.schedule([spec, ...]) -> {hash: metrics}   # many, batched, deduped
snn2.schedule_ray([spec, ...])              # same, multi-process (needs ray)
snn2.tune_run(search_space, num_samples=N)  # async hyperparam search (needs ray)
```

## What a spec is

A flat dict. Start from a preset and override only what you care about:

```python
{"preset": "izhi_randstate", "lr": 0.1, "stdp_window": 100}
```

`snn2.PRESETS` lists presets; `snn2.DEFAULTS` lists keys you may omit;
`snn2.expand(spec)` shows the fully-resolved config that will run and be hashed.

## Choosing an engine (registry-style)

The engine is just another registered part, named by the `engine` key. The
preset usually sets it for you; `snn2.names("engine")` lists what's available.

- `engine="batched"` (default) -- **stateful** games via a `(reset, step)` pair,
  driven by actions. Used by `izhi_randstate` (the random-state RL task).
- `engine="trial"` -- **trial-based** games that return a per-lane cue sequence
  `[B, n_steps]`; each step is one independent trial with per-trial eligibility.
  Used by `instrumental`, `logic_monotonic`, `logic_dendritic`.

Same-shape specs that name the **same engine** batch together; differing engines
or sizes go in separate buckets automatically.

## What metrics come back

`{"spec": <resolved>, "final_reward": float, "mean_out_rate": float,
"weight_norm": float}` (trial runs also add `weights`, and `trial_reward` /
`trial_correct` when `log_trials=True`).

`mean_out_rate` near 0 means the net is silent; near 1 means saturated -- either
extreme means no learning signal. Aim for a graded middle.

## Adding a part (only if a preset/part doesn't exist)

One registered pure function, then it's a string anywhere:

```python
@snn2.register("neuron", "my_model")
def my_model(state, I, p, rng):
    ...
    return state, fired   # fired: [B, N] bool
```

Part contracts:
- neuron `(state, I, p, rng) -> (state, fired[B,N])`
- input `(state_int[B], rng, p) -> spikes[B, n_inputs]`
- readout `(out_spikes[B, T, n_outputs], p) -> action[B]`
- reward `(state, action, state_next, p) -> reward[B]`
- batched game `(rng, p) -> (reset, step)`; trial game `(rngs, p, n_steps) -> cues[B, n_steps]`
- engine `(specs, seed=0, **kw) -> [metrics, ...]` -- register under kind `"engine"`
  in `snn2/engines/` and import it in `snn2/engines/__init__.py`.

## Rules of thumb

- Vary parameters via `sweep`/`tune_run`, never by mutating a preset in place.
- Give lanes different `len_episode` freely -- the active-mask handles staggered
  and uneven episodes in one batched run.
- Before trusting any result, run `python -m snn2.validate` (the learning rule
  must match its reference oracle and pass the sign/zero/recency properties).

## Where things live

- `snn2/` -- the package: `api.py`, `spec.py`, `parts.py`, `registry.py`,
  `stdp.py`, `validate.py`, `florian.py`, and `engines/` (`batched.py`, `trial.py`).
- `examples/` -- runnable scripts and study packages (izhikevich, florian,
  izh/conditioning, logicgates).
- `docs/` -- `usage.md` (this guide's source), `dcaap.md`, `dcaap-findings.md`.
