# Farries & Fairhall (2007): reward-modulated STDP is gradient ascent on reward

A reproduction of the central theoretical result of Farries, M. A. & Fairhall,
A. L. (2007), *“Reinforcement learning with modulated spike timing-dependent
synaptic plasticity,”* J. Neurophysiol. 98:3648–3665 — validated against an exact
independent oracle.

- Module: `snn2/farries.py`
- Example: `examples/farries.py`
- Run validation: `python -m snn2.farries`
- Figure: `python -c "from snn2.farries import make_figure; make_figure('farries_fig.png')"`

---

## 1. The result, in one paragraph

STDP on its own is unsupervised — it strengthens correlated synapses regardless of
outcome. Farries & Fairhall showed that if a **reward signal modulates** the STDP
update, the synapse’s spike-timing eligibility trace plays the role of the
**policy-gradient “score function”** `∂/∂w log P(output)`, so the *trial-averaged*
weight change climbs the gradient of **expected reward**:

```
⟨Δw_i⟩  =  Cov(R, e_i)  +  R̄ · ⟨e_i⟩
        ≈  ∂E[R]/∂w_i        — once the reward is CENTERED (R̄ subtracted)
```

The first term is the gradient (the useful, reward-sensitive part). The second is
an **unsupervised STDP bias** that does not point along the gradient. Subtracting a
reward baseline `R̄` removes it. So reward-modulated STDP performs reinforcement
learning **only when reward is referenced to its expected value** — the result, and
the now much-cited caveat, that this module reproduces and tests directly.

---

## 2. Why this is the right thing to reproduce (and how it’s validated)

The expected reward `E[R]` is a smooth function of the weights, so its gradient can
be measured **directly** by finite differences: nudge each weight `±ε`, re-estimate
`E[R]`, divide. That gives an **exact, independent oracle** for the direction the
learning rule *should* move — precisely the golden-test structure the snn2
validation methodology is built around (see the `neural-model-validation` skill).

`snn2/farries.py:validate()` then checks the rule against that oracle:

1. The **exact score-function** update aligns with the oracle gradient
   (cosine ≈ 1.0) — confirming the policy-gradient identity numerically.
2. The **realistic STDP** (Hebbian-coincidence) update, **centered**, also aligns
   with the gradient (cosine > 0.8) — this is the Farries–Fairhall claim.
3. **Centering matters:** at a weight where the gradient and the bias diverge, the
   centered update aligns (cosine ≈ +1) while the **uncentered** update points the
   *opposite* way (cosine ≈ −1).
4. **Centered learning climbs** `E[R]`; **mis-baselined learning does not**
   (it stalls or runs backward).

All checks pass.

---

## 3. The model

A minimal **escape-noise (stochastic) spiking neuron**, faithful to the paper’s
model class:

```
inputs       x_i(t) ~ Bernoulli(rate_i)            N input lines, fixed stimulus
membrane     u(t)   = Σ_i w_i x_i(t)
firing prob  ρ(t)   = sigmoid(gain · (u(t) − θ))   escape noise
output       y(t)   ~ Bernoulli(ρ(t)),  count n = Σ_t y(t)
reward       R(n)   = exp(−(n − n*)² / 2σ²)  ∈ [0, 1]
```

The neuron must learn weights that make its output spike count match a target `n*`
— a target-rate / “biofeedback” task of exactly the kind these rules were built to
explain.

### Two eligibility traces

| trace | formula | meaning |
|---|---|---|
| **score** | `e_i = Σ_t (y(t) − ρ(t)) x_i(t)` | exact `∂/∂w_i log P(y\|x)`; **mean zero** |
| **stdp**  | `e_i = Σ_t  y(t)        x_i(t)` | realistic Hebbian coincidence; **biased**, `⟨e_i⟩ = Σ_t ρ(t) x_i(t)` |

The score trace is what an ideal policy-gradient learner would use; the stdp trace
is what a synapse can actually measure from pre/post spikes. Their difference is
exactly the `ρ`-weighted “predicted-post” term — the bias that the reward baseline
has to cancel.

---

## 4. The math (why centering works)

By the REINFORCE / policy-gradient identity, for the mean-zero score trace and
**any** constant `b`,

```
∂E[R]/∂w_i = E[ R · e_i^score ] = E[ (R − b) · e_i^score ].
```

Writing the realistic trace as `e^stdp = e^score + bias`, the reward-modulated
update decomposes as

```
⟨(R − b) e^stdp⟩ = Cov(R, e^stdp) + (E[R] − b)·⟨e^stdp⟩.
```

- Choose `b = E[R]` (centered) → the second term **vanishes**, leaving
  `Cov(R, e^stdp) ≈ ∂E[R]/∂w` (the gradient). ✔
- Leave `b = 0` (uncentered) → the term `E[R]·⟨e^stdp⟩` survives. Since `⟨e^stdp⟩`
  is the always-positive Hebbian bias, it drags every synapse the same way,
  regardless of the gradient — and where the true gradient is *negative*
  (an over-firing neuron that should weaken), the uncentered update moves
  **backward**.

This is visible in the figure: at an over-driven weight the centered update sits on
the gradient diagonal (cosine +1), the uncentered update on the opposite diagonal
(cosine −1).

---

## 5. The figure

`farries_fig.png`, three panels:

- **Learning depends on the reward baseline.** Centered reward → `E[R]` climbs to
  ~0.8. Offset baseline (`b = E[R] − 0.6`) and no baseline (`b = 0`) → the
  unsupervised bias dominates, the neuron runs away, `E[R]` collapses to 0.
- **R-STDP update vs true reward gradient.** Per-synapse scatter against the
  finite-difference oracle. Centered R-STDP lies on the `y = x` line (cosine 1.00);
  uncentered lies on the wrong side (cosine −1.00).
- **Mis-setting the baseline destroys learning.** Final `E[R]` vs baseline offset
  `b − E[R]`: a sharp peak at zero (the true baseline). This echoes Frémaux et al.
  (2010), who showed exactly this fragility for the Farries–Fairhall R-STDP rule.

---

## 6. How this differs from the Florian reproduction

`snn2/florian.py` reproduces the **mechanics** of one reward-modulated synapse —
the eligibility-trace ODEs (MSTDPET) producing a specific weight trajectory, checked
against a numeric ground truth. `snn2/farries.py` reproduces the **theory of why it
works**: that the trial-averaged update is the gradient of expected reward, checked
against a finite-difference gradient oracle, plus the baseline condition that the
identity requires. Florian = “does the trace evolve correctly?”; Farries–Fairhall =
“does the rule climb the reward gradient, and when?”

---

## 7. Configuration

`snn2/farries.py:DEFAULTS`

| field | value | role |
|---|---|---|
| `n_inputs` | 6 | synapses (kept small so the FD-gradient oracle is cheap) |
| `n_steps` | 25 | time steps per trial |
| `gain` | 2.0 | escape-noise sharpness |
| `theta` | 1.2 | firing threshold (membrane offset) |
| `input_rate` | 0.4 | nominal input firing prob (actual rates drawn per line) |
| `n_target` | 8 | target output spike count |
| `reward_sigma` | 3.0 | width of the reward bump around `n_target` |

Key knobs in `validate()` / `learn()`: `elig` (`"e_stdp"` or `"e_score"`),
`baseline` (`"centered"`, `"none"`, `"offset"`), `offset`, `lr`, `n_trials`.

---

## 8. How to run / extend

```bash
python -m snn2.farries          # prints alignment cosines + 5 validation checks
python examples/farries.py      # also renders farries_fig.png
```

- **Discrimination task.** Replace the single-pattern target-count reward with two
  input patterns (reward firing for A, silence for B). The gradient then has mixed
  signs across synapses, making the baseline’s role even starker.
- **Reward predictor.** Instead of a scalar running `R̄`, learn a state-dependent
  baseline `b(stimulus)` — the “critic” that makes R-STDP work on tasks with
  several stimuli, as later actor–critic spiking models do.
- **Realistic STDP window.** Swap the instantaneous coincidence for a bi-phasic
  STDP kernel over `Δt = t_post − t_pre`; the alignment with the score function then
  depends on how well the kernel approximates the neuron’s sensitivity — the
  finer-grained version of Farries & Fairhall’s argument.

---

## 9. Reference

Farries, M. A., & Fairhall, A. L. (2007). Reinforcement learning with modulated
spike timing-dependent synaptic plasticity. *Journal of Neurophysiology*, 98(6),
3648–3665. doi:10.1152/jn.00364.2007

Related: Xie & Seung (2004); Pfister et al. (2006); Florian (2007); Izhikevich
(2007); Legenstein et al. (2008); Frémaux et al. (2010), which characterizes the
reward-baseline requirement tested in panel 3.
