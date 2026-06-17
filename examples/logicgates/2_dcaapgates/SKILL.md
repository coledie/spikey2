---
name: dcap-logic-circuits
description: Build, compose, visualize, and stabilize spiking logic circuits made of dCaAP (band-pass / non-monotonic) neurons. Use when constructing single-neuron logic gates (OR, AND, XOR, NOT, NAND, NOR) from dendritic-calcium-spike neurons, wiring them into multi-step ("multiphase") circuits like XNOR, half-adders, or adders, diagnosing why a deep spiking-gate cascade loses its signal, or planning how to train multi-layer spiking logic end-to-end with reward-modulated STDP. Triggers include dCaAP, Gidon neuron, non-monotonic/band-pass neuron logic, spiking logic gates, cascade/noise-margin stability of SNN gates, and inverting gates from excitatory-only neurons.
---

# dCaAP logic circuits

A toolkit + methodology for making **logic gates out of single non-monotonic (dCaAP)
neurons**, **composing** them into multi-step circuits, and keeping deep cascades
**stable**. The companion code is `dcap_logic_framework.py` (standalone: numpy +
matplotlib).

## The one idea everything rests on

A dCaAP neuron fires only when its dendritic drive lands inside a band `[lo, hi]`. With
equal weights, drive depends on the number of active inputs `k`:

```
drive(k) = baseline + u*k          u = gain * w * group * rate   (the "drive unit")
```

A gate is just a **choice of where the band sits** on the ladder `{baseline, +u, +2u, ...}`:

| gate | fires on | how |
|------|----------|-----|
| XOR  | k = 1            | band straddles `u` (k=2 overshoots above) |
| AND  | k = 2            | band straddles `2u` (k=1 below) |
| OR   | k ≥ 1            | band covers `u..2u` |
| NOT  | k = 0 (1 input)  | **baseline in band**; input pushes drive *above* |
| NOR  | k = 0            | baseline in band; any input pushes above |
| NAND | k < 2            | band covers `baseline..+u`; k=2 pushes above |

The inverting gates (NOT/NAND/NOR) are the headline: **adding excitatory input makes
the neuron fire less** — negation by dendritic suppression, with no inhibitory synapse
and no second neuron. A monotonic point neuron with excitatory-only inputs cannot do
this. **XNOR cannot be done by one neuron** (it must fire on k=0 and k=2 but not k=1 —
two non-adjacent levels a single band can't isolate); it requires composition.

## Workflow

1. **Build a gate.** Pick `n_in`, then use `design_band(levels, fire_on)` to place the
   band on the drive ladder. If `design_band` raises "non-contiguous", the function is
   not single-neuron realizable — compose it.
2. **Validate the truth table.** `gate_truth(cfg)` returns firing rates and the binary
   table. Compare to the target. A correct gate has a clear gap between firing (>~0.5)
   and silent (<~0.1) rates — that gap is the **noise margin** and it matters downstream.
3. **Compose.** Wire gates with `Circuit`: an input line is `('in', i)` (primary bit) or
   `('node', j)` (an upstream gate's output *rate*). Evaluate with `.truth()`. Gate
   outputs are rates, not clean bits, so the next gate's band acts as a **logic-level
   restorer** that re-sharpens the signal each layer.
4. **Match the interface.** This is where cascades live or die. Use
   `match_interface(cfg, r_false, r_true, target)` to set the downstream input weight so
   the upstream "true" rate lands on the correct side of the band (`'suppress'` for an
   inverting gate, `'excite'` otherwise). An unmatched interface loses margin every layer.
5. **Check stability.** `cascade_margin(cfg, depth, w_if)` returns the noise margin at
   each depth. A matched interface holds the margin open arbitrarily deep; a mismatched
   one collapses it to zero within a few layers. Plot with `plot_stability`.
6. **Visualize.** `plot_gate_ladder` (the drive-ladder + band per gate) and
   `plot_stability` (margin vs depth) are the two diagnostic plots.

## Stability — what actually makes deep circuits work

The band is a **signal restorer**, exactly like a threshold restores logic levels in a
digital circuit. Three levers keep a multiphase circuit stable:

- **Interface matching** (`match_interface`): the upstream rate range must map onto the
  band; this is the dominant factor. Mismatch → margin collapse with depth.
- **Homeostatic self-limiting** of the dCaAP band caps runaway drive (see the `florian2`
  result): a band-pass neuron's plasticity has a stable interior fixed point, so learned
  weights don't drift out of the band. Use this when *training* (not just configuring).
- **Reward-only learning + weight caps** avoid the "band-entry-order trap": with a
  band-pass neuron, pure-LTP can push the wrong pattern's drive through the band first
  and lock a bad solution; `punish_mult = 0` plus a cap inside the target window fixes it.

## Validation ladder (use for every new gate/circuit)

1. drive ladder is correct: `drive(k) = baseline + u*k` lands where intended;
2. truth table matches target with a clear firing/silent gap (noise margin > ~0.5);
3. composed-circuit truth table matches the boolean function;
4. cascade margin stays open at the target depth (matched) and you can *show* it
   collapsing when the interface is detuned (a negative control);
5. results hold across random seeds.

## Files

- `dcap_logic_framework.py` — the standalone toolkit (gate library, `Circuit`,
  `design_band`, `match_interface`, `cascade_margin`, plot helpers, `self_test()`).
- `TRAINING_PLAN.md` — how to take this from hand-configured gates to **end-to-end
  trained multi-step circuits** with reward-modulated STDP (milestones, credit
  assignment, stability constraints, metrics).

Run the self-test: `python dcap_logic_framework.py`.
