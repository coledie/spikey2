---
name: neural-model-validation
description: >
  How to implement a computational neuroscience model from a paper -- a spiking
  neuron (Izhikevich, LIF, Hodgkin-Huxley), a synaptic plasticity / learning rule
  (STDP, reward-modulated STDP / MSTDPET, Hebbian), or a small dynamical system --
  AND validate that it actually works rather than merely runs. Use this whenever
  the user asks to "reproduce", "implement", "port", or "replicate" a model or a
  paper figure (e.g. "repro Florian Fig 1", "implement the Izhikevich neuron",
  "code up STDP", "make the cart-pole dynamics"), whenever they want to be sure a
  model is correct/validatable, or whenever they are building an SNN/RL component
  and need confidence it matches the source. Reach for this skill even if the user
  doesn't say the word "validate" -- a reproduction that isn't validated is not a
  reproduction.
---

# Validating computational neuroscience models

A model that runs is not a model that works. The gap between "produces numbers"
and "produces the *right* numbers" is where reproductions silently fail. This
skill captures a validation ladder that closes that gap, distilled from building
and verifying an Izhikevich neuron and the Florian (2007) MSTDPET learning rule.

The core idea: **never trust a single implementation of an equation.** Pin it
against an independent reference and against the analytic facts that define the
phenomenon. Then, and only then, plot it.

## The validation ladder

Climb these in order. Each rung catches a class of bug the others can't.

1. **Source the exact equations** — from the paper or canonical code, never from
   memory. "I know Izhikevich" is how you ship `0.04v² + 5v + 140` as
   `0.04v² + 5v + 150`. Copy the numbered equations and parameter tables verbatim
   into a docstring.
2. **Implement as a pure function** — explicit parameters in, arrays out, no
   hidden state. Pure functions are the only kind you can test cheaply and batch
   later.
3. **Golden test against an independent oracle** — write a second, deliberately
   dumb implementation (an explicit loop) and assert the two agree byte-for-byte
   on random inputs. This catches vectorization/indexing bugs.
4. **Analytic property tests** — assert the qualitative facts that *define* the
   phenomenon (sign of an STDP weight change, spike-frequency adaptation, zero
   input → zero change). This catches bugs the golden test misses when both
   implementations share the same mistake.
5. **Reproduce the canonical figure + capture numeric ground truth** — run the
   reference once, record key numbers (final weight, peak, spike counts), then
   assert your model hits them. Eyeball the figure against the paper.
6. **Tune nuisance parameters into the responsive regime** — drive strength,
   integration step, threshold. Find the band empirically and *document why* it's
   there. A model outside its regime looks broken even when the equations are right.

For full, annotated implementations of both worked examples, read
`references/worked_examples.md`. For a copy-paste starting point, use
`scripts/validate_template.py`.

## Rung 1 — Source the exact equations

Pull the real equations and parameters into the code as documentation. For
Izhikevich (2003):

```
v' = 0.04 v^2 + 5 v + 140 - u + I
u' = a (b v - u)
if v >= 30 mV:  v <- c,  u <- u + d
```

For Florian (2007) MSTDPET, cite the equation numbers:

```
P+  <- P+ * exp(-dt/tau_+) + A+ * f_pre        (Eq 43)
P-  <- P- * exp(-dt/tau_-) + A- * f_post       (Eq 44)
zeta = P+ * f_post + P- * f_pre                 (Eq 42)
z   <- z  * exp(-dt/tau_z) + zeta               (Eq 8)
w   <- w  + gamma0 * r * z                      (Eq 7)
```

If the source is a repo, read the actual implementation — not the README's prose.
Repos frequently differ from the paper in small ways (an off-by-one in a reward
schedule, a half-step integration scheme). Reproduce the behavior you can verify,
and note any deviation in a comment.

## Rung 2 — Pure function

Write the dynamics as `f(state, inputs, params) -> (state, outputs)`. Keep RNG
out where possible; where randomness is needed, take an explicit generator so
runs are reproducible. Reproducibility is a precondition for validation: you
cannot golden-test against an oracle if the same inputs give different outputs.

## Rung 3 — Golden test against an oracle

The single highest-value check. Write the obvious, slow version and assert
equality:

```python
def stdp_delta(spike_log, trace, lr, window, n_inputs, polarity):
    "vectorized -- the hot path"
    ...

def stdp_delta_ref(spike_log, trace, lr, window, n_inputs, polarity):
    "explicit quadruple loop -- obviously correct, never shipped hot"
    ...

# golden test, random inputs:
for _ in range(200):
    args = random_case()
    assert np.allclose(stdp_delta(*args), stdp_delta_ref(*args))
```

Rules that make this work:
- The oracle must be written **independently** from the fast version, ideally
  from the equations directly, so they don't share a bug. If you copy-paste and
  tweak, you copy the bug too.
- Test on **random** inputs, many trials — fixed inputs hide edge cases.
- Test the **batched** version against the oracle per-lane: stacking experiments
  on a leading axis is exactly where broadcasting bugs hide.

## Rung 4 — Analytic property tests

Encode the facts that define the phenomenon. These are your defense when the
oracle and the implementation share a mistake (they can't both be wrong about a
sign). Examples that actually caught or would catch bugs:

For an STDP rule:
- pre-before-post **potentiates** (positive Δw); post-before-pre **depresses**.
- zero reward / zero eligibility → **zero** weight change.
- no post-synaptic spike → no update.
- recency weighting is **monotonic** (a more recent co-fire earns ≥ credit).

For a spiking neuron, assert the behaviors that *name* each regime:
- every regime spikes under sufficient drive.
- fast-spiking fires the most; regular-spiking shows **adaptation** (inter-spike
  interval grows); bursting opens with a short-ISI burst; chattering shows
  recurring short-ISI bursts while regular-spiking shows none.

Write these as assertions with messages that read like the fact:
`assert isi[-1] > isi[0], "RS should show spike-frequency adaptation"`.

## Rung 5 — Figure + numeric ground truth

Run the reference once and **print the numbers you'll assert against** before
writing the model — e.g. "w: 0.200 → 0.316, peak 0.416" or per-regime spike
counts. Then your model has a target, not just a vibe. Reproduce the paper's
figure (multi-panel is fine) and compare visually. A figure is a weak test on its
own but a strong sanity check on top of rungs 3–4.

## Rung 6 — Regime tuning (the silent killer)

Most "the model is broken" moments are actually "the model is outside its
operating regime." Two nuisance parameters dominate:

- **Drive / input gain.** Izhikevich fires near +30 mV and needs real input
  current; an LIF-scale drive leaves it silent, too much saturates it. Sweep the
  gain, find where the firing rate is graded (≈0.2–0.5), and **record the band in
  a comment or preset** so nobody else rediscovers it. Example finding:
  "gain ≤4 silent, ≥6.5 saturated, 5.0 gives ~30% firing."
- **Integration step.** Use the paper's scheme. Izhikevich's canonical code takes
  two 0.5 ms Euler sub-steps for `v` and one update for `u` per ms; deviating
  changes the spike shape and timing.

Tune empirically with a tiny sweep, measuring an interpretable quantity (firing
rate, accuracy), and keep the model parameters fixed at paper values — only the
nuisance/coupling parameters are yours to set.

## Common pitfalls (all hit in practice)

- **The naming trap.** A file or task named after a model may not contain that
  model. The repo's `izhikevich2007.ipynb` is an RL *task* that runs an LIF
  neuron — the actual Izhikevich `a,b,c,d` model wasn't there. Verify what the
  code does, don't trust the label.
- **Model vs task confusion.** "Reproduce Izhikevich" can mean the neuron-dynamics
  figure *or* a learning task from a same-named paper. Ask or state which.
- **Credit assignment in reward-modulated rules.** A global scalar eligibility
  trace carried across trials misassigns reward to the *next* trial's activity →
  learning sits at chance. Use **per-trial (or per-synapse) eligibility** scaled
  by that trial's reward. This was the difference between 0.5 and 1.0 accuracy.
- **Oracle sharing a bug.** If the oracle is a lightly-edited copy of the fast
  path, the golden test passes while both are wrong. Write the oracle from the
  equations.
- **Reference off-by-ones.** When matching a repo's curve exactly, replicate its
  quirks (e.g. a reward flip that takes effect one step late) and document them,
  rather than "fixing" them and diverging from the target.
- **Saturation masquerading as failure.** Weights pinned at the clip ceiling or a
  neuron firing every step looks like learning but isn't. Always log the firing
  rate / weight norm and check it's in a graded middle.

## Output checklist

A validated model reproduction ships with:
- [ ] equations + parameters quoted from the source in a docstring
- [ ] pure function(s) for the dynamics
- [ ] an independent oracle + a golden test over random inputs (and batched)
- [ ] analytic property tests named after the facts they check
- [ ] numeric ground truth captured from the reference and asserted
- [ ] the canonical figure rendered and visually compared
- [ ] nuisance-parameter regime documented (with the band that works and why)
- [ ] a `validate()` entrypoint that runs all checks and prints pass/fail
