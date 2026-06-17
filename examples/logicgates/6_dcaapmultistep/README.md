# Learning multi-step dCaAP logic end-to-end — from multi-input gates to an all-at-once adder

This is the open milestone closed: composed, multi-step dCaAP circuits learned **end-to-end
from a single circuit-level reward**, scaling up to an adder learned **all at once**. No gate
is hand-placed — every weight and band, and every inter-gate interface scale, is learned by
the oscillator-based global optimizer (extremum-seeking + reward-gated annealing + relaxation
restart, from `snn2.dcap_global_learn`, generalized here to an arbitrary parameter vector).

## Milestones (all validated in one run)

| milestone | circuit | params learned | result |
|---|---|---|---|
| **A** | 3-input AND, 3-input OR, 4-input AND | single dCaAP `(w, band)` | 1.00 |
| **B** | **3-input XOR (parity)** = `XOR(XOR(a,b), c)` | 2 gates + 2 interface scales (5) | 1.00 |
| **C** | **full adder** (5 gates) | 3 gate types + 2 interfaces (11) | 1.00 |
| **D** | **4-bit ripple adder, all at once** | 12 **shared** params | **256/256 exhaustive** |
| **D′** | same tile at **8-bit** | (reused, width-independent) | all sampled correct, incl. carry-propagate |

Milestone B is the first genuinely **multi-step** result: a 2-input XOR cannot be a single
dCaAP neuron (its true-set is non-contiguous), so 3-input parity *requires* a 2-layer circuit,
and both XOR gates' parameters **and** the interface scales are learned jointly from the
8-row parity truth table.

## What makes "learn the adder all at once" tractable

An N-bit ripple adder is **N identical full-adder tiles**, so we **share parameters across the
tile**: all XOR gates share one `(w, band)`, all ANDs share one, all ORs share one, plus three
interface scales (XOR-output, AND-output, carry). That collapses ~120 parameters (8-bit) to
**12**, independent of width. The 12-vector is optimized from the adder's sum-correctness reward
over a handful of sampled additions — and then **generalizes**:

- learned on 4-bit → **exhaustively correct on all 256 additions**;
- the *same 12 numbers* dropped into an 8-bit adder → every sampled addition correct, including
  all carry-propagate stress cases (255+255, etc.);
- learning **directly** at 8-bit also succeeds (training reward 1.0, all sampled additions
  correct).

So the tile is **width-independent**: learn it small, deploy at any width.

See `dcap_multistep.png`: left, the composed circuits (parity, full adder, 4-bit adder) climbing
to reward 1.0 from one reward signal; right, every milestone from single-neuron gates to the
8-bit adder.

## How this connects to the earlier results

- The optimizer is the **oscillator/control learner** that solved the AND gate from any starting
  position (`../dcapGlobalLearn/`). Here it's lifted from one gate to a whole circuit's parameter
  vector.
- The earlier **honest negative result** still stands: reward-modulated Hebbian STDP cannot train
  these gates (deadband + gradient sign-reversal; `../dcapRStdpRewards/`). This module learns from
  reward but via gradient-free oscillatory search, not STDP.
- The hand-verified adder (`../dcap8bitAdder/`, 65,536/65,536 boolean) is the correctness oracle
  the learned circuits are checked against.

## Honest scope

- **Learning is gradient-free reward optimization** (extremum-seeking / annealing / restart),
  not spike-timing plasticity. It optimizes a real spiking dCaAP circuit's parameters from a
  scalar reward.
- **The topology is given** (the full-adder wiring and which gate feeds which). What is learned is
  every continuous parameter: gate weights, dendritic bands, and interface scales. Learning the
  *wiring* itself is not attempted here.
- **Parameter sharing is a modelling choice** that makes all-at-once learning tractable; it
  exploits the adder's regularity. Learning 120 *independent* gate parameters from one global
  reward would be far harder and is not claimed.
- **8-bit correctness** is established by (i) 4-bit exhaustive 256/256, (ii) the width-independent
  shared tile, and (iii) large 8-bit samples incl. all carry-propagate cases — not a full 65,536
  spiking sweep (which is verified for the hand-built adder in `../dcap8bitAdder/`).

## Run

```bash
cd snn2 && PYTHONPATH=. python -m snn2.dcap_multistep     # all milestones + checks
# or:
python run.py
```
