# dCaAP Negative Patterning — and a family of configural logic gates

A single non-monotonic (dCaAP) neuron solves **negative patterning**, a classic
animal-learning paradigm that elemental/linear models famously fail — and the same
band-pass non-linearity generalizes into a whole family of **"k-of-n" logic gates**
that no point neuron can compute.

## The science

**Negative patterning** (Rescorla; Pearce's configural theory): reinforce cue *A*
alone and cue *B* alone, but **not** the *AB* compound.

```
A  → reward      B  → reward      AB → no reward      (nothing → no reward)
```

The subject must respond to either single cue yet withhold to the compound. This is
exactly **XOR** over the two cues (`none=0, A=1, B=1, AB=0`). Elemental learners fail
it: if *A* and *B* each excite the response, their compound should excite it *more*
(**summation**) — the opposite of what's required. A configural unit that *suppresses*
the stronger compound drive can solve it. The dCaAP neuron is such a unit.

## What's in `dcap_negative_patterning.py`

- `negative_patterning()` — trains a dCaAP agent and a monotonic agent on the task
  with reward-modulated STDP, and reads out each one's response to `none/B/A/AB`.
- `kofn_firing()` — the dCaAP as a tunable **exactly-k-of-n** detector.
- `validate()` — six property checks (all pass).
- `make_figures()` — the two figures below.

Run it:
```bash
PYTHONPATH=. python -m snn2.dcap_negative_patterning   # prints the validation table
```

## Results (validated)

| claim | result |
|---|---|
| dCaAP learns negative patterning | accuracy **1.00** |
| monotonic agent fails it | accuracy **0.50** (chance) |
| dCaAP suppresses the compound | AB / single-cue = **0.02** |
| elemental learner summates | AB / single-cue = **1.73** |
| dCaAP can be an *exactly-1-of-3* detector | ✓ |
| dCaAP can be an *exactly-2-of-3* detector | ✓ |

## Figures

### `dcapNP_negative_patterning.png`
Left: the dCaAP learns negative patterning to ceiling while the monotonic learner is
stuck at chance. Middle: the **elemental** learner (which has learned the single-cue
associations) responds to *AB* *above* its single-cue response — summation, the wrong
answer. Right: the dCaAP responds strongly to *A* and *B* but **suppresses *AB*** —
correct configural responding.

### `dcapNP_kofn_gates.png`
A single dCaAP neuron is a tunable counting detector. **Two inputs:** the synaptic
weight scale selects AND ("both") or XOR ("exactly one"). **Three inputs:** it can be
tuned into an *exactly-1-of-3* or *exactly-2-of-3* detector. **Right panel:** why a
point neuron can't — its drive grows monotonically with the number of active inputs,
so it can never fire for "exactly k" and stay silent for "more than k". The band picks
out one count.

## Honest notes
- Negative patterning is *formally* XOR; the contribution here is the conditioning
  framing and the explicit **summation vs suppression** read-out, plus the k-of-n
  generalization to 3 inputs.
- The k-of-n panels are a **capability map** (firing vs weight scale), computed
  directly; the 2-input gates are also learned end-to-end via R-STDP in the
  `snn2.logic_gates` module this builds on.
