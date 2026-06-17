# dcapLogicGates — all logic gates on dCaAP neurons, and multi-step circuits

Every basic logic gate realized on a **single dCaAP (non-monotonic) neuron** — including
the inverting gates a point neuron can't do — then composed into **multi-step circuits**,
with a first answer to the question that decides whether deep circuits work: **cascade
stability**.

## 1. All six gates on one neuron — `dcap_logic_gates.py`

A dCaAP fires only when its drive lands in a band. With equal weights, drive depends on
the number of active inputs `k`: `drive(k) = baseline + u·k`. Choosing where the band
sits on the ladder `{baseline, +u, +2u}` selects the gate.

| gate | fires on | result |
|------|----------|--------|
| OR   | k ≥ 1 | `[0,1,1,1]` ✓ |
| AND  | k = 2 | `[0,0,0,1]` ✓ |
| XOR  | k = 1 | `[0,1,1,0]` ✓ |
| NOT  | k = 0 (1 input) | `[1,0]` ✓ |
| NAND | k < 2 | `[1,1,1,0]` ✓ |
| NOR  | k = 0 | `[1,0,0,0]` ✓ |

**The headline:** NOT/NAND/NOR work by putting a **tonic baseline in the band**, so adding
excitatory input pushes drive *out the top* — the neuron fires *less* as input grows.
That is logical negation by dendritic **suppression**, with no inhibitory synapse and no
second neuron. An excitatory point neuron cannot invert. *6/6 truth tables validated.*

`dcap_logic_gates.png` shows the drive ladder + band for each gate (green = fires); the
bottom row (the inverters) visibly negates via suppression.

> XNOR is deliberately absent — it must fire on k=0 and k=2 but not k=1, two
> non-adjacent levels a single band can't isolate. It needs composition.

## 2. Multi-step circuits + cascade stability — `dcap_circuits.py`

Gate outputs are firing **rates**, not clean bits (XOR fires ~0.74 true / ~0.01 false).
Wiring gate → gate feeds that rate into the next gate, whose band acts as a **logic-level
restorer**, re-sharpening the signal each layer.

- **XNOR = NOT(XOR)** → `[1,0,0,1]` ✓ (the gate one neuron can't do, in two layers)
- **Half-adder** → SUM = XOR `[0,1,1,0]`, CARRY = AND `[0,0,0,1]` ✓
- **Cascade stability** → in a chain of gates, a **matched interface** holds the noise
  margin ≈ 1.0 to depth 10; a **mismatched interface** collapses it to 0 by depth ~3.

*5/5 checks validated.* `dcap_circuits_compose_stability.png` shows both circuits' truth
tables and the matched-vs-mismatched margin-vs-depth curves.

**The takeaway:** the band is what makes deep spiking logic possible (level restoration),
and matching each gate's input weight to its upstream rate range is what keeps the cascade
stable. That matching is the trainable quantity in the end-to-end plan.

## Run

```bash
# from the synced snn2 package (sibling folder):
cd snn2 && PYTHONPATH=. python -m snn2.dcap_logic_gates
cd snn2 && PYTHONPATH=. python -m snn2.dcap_circuits
# or the runners here (need snn2 on PYTHONPATH):
python run_gates.py
python run_circuits.py
```

## See also

- `../dcap-logic-circuits.skill/` — the reusable framework (`design_band`,
  `match_interface`, `Circuit`, `cascade_margin`) and `TRAINING_PLAN.md` for taking this
  to **end-to-end trained** multi-step circuits.
- `../dcaap.md` — the broader dCaAP concept and the homeostatic-cap result that underpins
  cascade stability during training.

## Honest notes

- These gates are **configured** (bands/weights placed analytically) to prove
  realizability and give a known-good target. Training them from reward uses the R-STDP
  machinery already demonstrated for XOR in `snn2/logic_gates.py`; the end-to-end,
  multi-layer training is laid out in `TRAINING_PLAN.md`, not yet run.
- Composition uses a **rate interface** between layers; cascade stability is analyzed as a
  noise-margin-vs-depth property. The matched/mismatched contrast is a hand-set
  illustration of the quantity that training must learn (`match_interface`).
