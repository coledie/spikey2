# florian2 — a dCaAP postsynaptic neuron makes reward-modulated STDP self-limiting

Florian (2007) introduced **MSTDPET**: reward-modulated STDP with eligibility traces.
This extension runs it in **closed loop** — the postsynaptic neuron's firing depends
on the very weight being learned — and shows that a **dCaAP (band-pass) postsynaptic
neuron turns the eligibility trace into a homeostatic cap**: the weight settles at a
stable fixed point instead of running away.

## The idea

Drive = `w × (presynaptic activity)`, and the post neuron fires as a function of that
drive. Reward is held positive, so MSTDPET potentiates the synapse through the
pre→post pairing it creates.

- **Monotonic post:** more weight → more post firing → more pairing → more eligibility
  → more weight. Positive feedback — the weight **runs away** to its ceiling.
- **dCaAP post:** firing is band-pass. As the weight grows, drive climbs *into* the
  band (post fires, weight grows) and then *past* it (post goes silent → pairing
  vanishes → eligibility vanishes → growth stops). Negative feedback at the top edge —
  the weight settles at a **stable fixed point**.

This is the same non-monotonicity that lets one dCaAP neuron compute XOR, now
stabilizing its own reward-modulated learning: **the band is a homeostatic cap.**

## What's in `florian2.py`

- `closed_loop()` — the MSTDPET closed loop (Florian's `P+`, `P−`, eligibility `z`,
  reward-gated update), with a monotonic or dCaAP postsynaptic f-I.
- `phase_portrait()` — mean `dw/dt` as a function of (frozen) weight.
- `fixed_points()` — interior zero-crossings with stability.
- `validate()` / `make_figure()`.

Run it:
```bash
PYTHONPATH=. python -m snn2.florian2
```

## Results (validated)

| claim | result |
|---|---|
| monotonic post → weight runs away to ceiling | final w ≈ **1.97–1.99** (ceiling 2.0) |
| dCaAP post → a stable interior fixed point exists | **w\* ≈ 1.25** |
| dCaAP trajectories converge below the ceiling | all **< 1.3** |
| dCaAP trajectories converge together (4 different inits) | spread **< 0.03** |
| fixed-point drive sits at the dCaAP band | drive ≈ **6.25** (band top) |

## Figure — `florian2.png`

**Left:** the only thing changed between the two runs — the postsynaptic f-I curve
(monotonic vs band-pass). **Middle:** the phase portrait. The monotonic `dw/dt` stays
positive across the whole weight range (runaway); the dCaAP `dw/dt` crosses zero with
negative slope at `w*≈1.25` — a stable fixed point. **Right:** weight trajectories
from four initial weights. Same learning rule; the monotonic post pins every run at
the ceiling, the dCaAP post pulls every run to `w*≈1.26`.

## Honest notes
- The stable fixed point requires a small homeostatic weight-decay term to give a
  restoring force above the band; without it the dCaAP still self-limits (it stops
  growing once drive exceeds the band) but the equilibrium is a soft ceiling rather
  than a sharp attractor. The decay makes the attractor crisp and is standard
  synaptic homeostasis.
- The post f-I is a phenomenological band-pass (matching the two-compartment dCaAP
  neuron in the sibling folder), not a full biophysical compartment, to keep the
  closed-loop dynamics analyzable.
