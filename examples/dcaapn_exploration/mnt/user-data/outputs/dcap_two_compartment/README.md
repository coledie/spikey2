# Two-compartment dCaAP neuron — a non-monotonic f-I curve

A biophysically-flavored model of the **dendritic calcium action potential (dCaAP)**
from Gidon et al. (2020), *"Dendritic action potentials and computation in human
layer 2/3 cortical neurons"* (Science 367:83–87) — and its signature: a single neuron
with a **non-monotonic f-I curve**.

## The science

Gidon et al. found a dendritic Ca²⁺ spike whose **amplitude is maximal for
threshold-level input and *decreases* for stronger input** — a band-pass activation,
the opposite of a normal neuron. A single such neuron can compute XOR.

This module realizes it with **two compartments**:

- **dendrite** — produces a dCaAP whose amplitude is a non-monotonic ("hump") function
  of dendritic drive: ON above a low threshold, OFF again above a high one. This is the
  Gidon activation curve.
- **soma** — a regular-spiking **Izhikevich (2003)** neuron driven by the dCaAP
  current, producing the Na⁺ output spikes.

The consequence is a property **no single point neuron can produce**: the somatic
firing rate rises with input, **peaks at threshold-level drive, then falls** for strong
drive (the dCaAP is suppressed). A standard Izhikevich RS neuron has a monotonic,
saturating f-I curve.

## What's in `dcap_neuron.py`

- `dcap_activation()` — the non-monotonic dendritic activation.
- `simulate()` — the two-compartment neuron under constant drive.
- `fi_curve()` — the dCaAP f-I curve and a standard-RS comparison.
- `validate()` / `make_figure()`.

Run it:
```bash
PYTHONPATH=. python -m snn2.dcap_neuron
```

## Results (validated)

| claim | result |
|---|---|
| dCaAP f-I peaks at an **interior** drive | peak **41 Hz at drive = 6.5** |
| strong drive **suppresses** the dCaAP | **0 Hz** at strong drive |
| comparison RS neuron is **monotonic** | rises to ~39 Hz and saturates |
| dendritic activation is a **hump** | ✓ |

## Figure — `dcap_two_compartment.png`

**Left:** the dendritic dCaAP activation — amplitude vs drive, ON inside the band and
suppressed above it. **Middle:** somatic membrane traces at three drive levels —
sub-threshold (silent), threshold (regular dCaAP-driven spiking), and strong
(**suppressed**, silent again). **Right:** the f-I curve — the two-compartment dCaAP
(green) rises to a peak and falls back to zero, while a standard RS Izhikevich neuron
(grey, dashed) increases monotonically.

This neuron is the shared primitive behind the negative-patterning and florian2 demos:
a tunable "moderate-drive / exactly-one" detector.

## Honest notes
- The dendritic dCaAP is modeled phenomenologically (a low-pass-filtered band-pass
  activation injected into the soma), not with full Hodgkin–Huxley Ca²⁺ channel
  kinetics. It reproduces the *computational* signature — the non-monotonic f-I curve
  — rather than the detailed conductance waveforms.
- The soma is a faithful Izhikevich RS unit (two 0.5 ms sub-steps, true `a,b,c,d`).
