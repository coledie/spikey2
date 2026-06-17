"""
dcap_circuits -- composing single-neuron dCaAP gates into multi-step ("multiphase")
logic circuits, and the question that decides whether deep circuits work: CASCADE
STABILITY.

A gate's output is a firing RATE, not a clean bit: an XOR fires at ~0.74 when "true"
and ~0.01 when "false". To wire gate -> gate we feed that rate into the next gate as
its input drive. The next gate's band then acts as a LOGIC-LEVEL RESTORER -- it fires
fully for in-band drive and is silent outside, re-sharpening the signal each layer,
exactly like a threshold restores levels in a digital circuit. The dCaAP band is what
makes a deep cascade possible; whether it stays stable depends on matching each gate's
input weight to the rate range its upstream gate actually produces (the noise margin).

WHAT'S HERE
  * eval_gate / Circuit : evaluate a DAG of dCaAP gates by propagating rates.
  * XNOR = NOT . XOR     : the gate a single neuron CANNOT do (k=0 and k=2, not k=1),
                           realized as a 2-layer circuit.
  * half_adder           : sum = XOR(a,b), carry = AND(a,b) -- the classic building block.
  * inverter chain       : a depth sweep showing the noise margin staying open (matched
                           interface) vs collapsing (mismatched) -- the stability result.

Run: python -m snn2.dcap_circuits
"""
from __future__ import annotations
import numpy as np
from .dcap_logic_gates import GATES, TRUTH, GROUP, RATE, GAIN, P


def eval_gate(cfg, line_rates, weights=None, trials=200, seed=0):
    """Firing rate of a dCaAP gate whose input lines carry the given rates. Each
    line is a group of GROUP neurons firing at its rate; `weights` (per line)
    default to the gate's scalar w."""
    rng = np.random.default_rng(seed)
    lo, hi = cfg["band"]; base = cfg["baseline"]
    w = cfg["w"]
    weights = [w] * len(line_rates) if weights is None else weights
    drive = np.full(P, base, dtype=float)
    for r, wl in zip(line_rates, weights):
        if r > 0:
            spk = (rng.random((P, GROUP)) < r).sum(1)
            drive = drive + GAIN * wl * spk
    return float(((drive >= lo) & (drive <= hi)).mean())


class Circuit:
    """A DAG of dCaAP gates. Each node: (name, cfg, inputs, weights). An input is
    ('in', i) for primary input bit i, or ('node', j) for the output of node j
    (which must appear earlier -- topological order)."""
    def __init__(self, n_inputs):
        self.n_inputs = n_inputs
        self.nodes = []     # list of dicts

    def add(self, name, cfg, inputs, weights=None):
        self.nodes.append(dict(name=name, cfg=cfg, inputs=inputs, weights=weights))
        return len(self.nodes) - 1

    def evaluate(self, bits, seed=0):
        """Return the firing rate of every node for a primary-input bit pattern."""
        rates = []
        for ni, node in enumerate(self.nodes):
            line_rates = []
            for src in node["inputs"]:
                if src[0] == "in":
                    line_rates.append(RATE if bits[src[1]] else 0.0)
                else:
                    line_rates.append(rates[src[1]])
            rates.append(eval_gate(node["cfg"], line_rates, node["weights"],
                                   seed=seed + ni))
        return rates

    def truth(self, thresh=0.2, seed=0):
        combos = [tuple((c >> (self.n_inputs - 1 - i)) & 1 for i in range(self.n_inputs))
                  for c in range(2 ** self.n_inputs)]
        out = {node["name"]: [] for node in self.nodes}
        for c in combos:
            r = self.evaluate(c, seed=seed)
            for node, rr in zip(self.nodes, r):
                out[node["name"]].append(rr)
        return combos, {k: np.array(v) for k, v in out.items()}


# --------------------------------------------------------------------------- #
# Example circuits
# --------------------------------------------------------------------------- #
def xnor_circuit():
    """XNOR = NOT(XOR(a,b)). The NOT input weight is matched to the XOR output rate."""
    c = Circuit(2)
    xor = c.add("XOR", GATES["XOR"], [("in", 0), ("in", 1)])
    # NOT fed by the XOR rate (true~0.74). Weight matched so 'true' pushes drive above
    # the band and 'false' (~0) leaves it at baseline.
    not_cfg = dict(GATES["NOT"])
    c.add("XNOR", not_cfg, [("node", xor)], weights=[0.40])
    return c


def half_adder():
    """sum = XOR(a,b), carry = AND(a,b)."""
    c = Circuit(2)
    c.add("SUM(XOR)", GATES["XOR"], [("in", 0), ("in", 1)])
    c.add("CARRY(AND)", GATES["AND"], [("in", 0), ("in", 1)])
    return c


def inverter_chain(depth, w_if, seed=0):
    """A chain of `depth` NOT gates. Returns, at each stage, the firing rate for a
    primary input of 0 and of 1. `w_if` is the inter-stage input weight (the
    interface match). Returns arrays r0[stage], r1[stage]."""
    cfg = dict(GATES["NOT"])
    r0_stage, r1_stage = [], []
    r0 = 0.0          # rate carried when primary input = 0
    r1 = RATE         # rate carried when primary input = 1
    for s in range(depth):
        r0 = eval_gate(cfg, [r0], weights=[w_if], seed=seed + s)
        r1 = eval_gate(cfg, [r1], weights=[w_if], seed=seed + 100 + s)
        r0_stage.append(r0); r1_stage.append(r1)
    return np.array(r0_stage), np.array(r1_stage)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate(verbose=True):
    log = print if verbose else (lambda *a, **k: None)
    log("dcap_circuits -- multi-step dCaAP logic and cascade stability\n")

    # XNOR = NOT . XOR
    cx = xnor_circuit()
    combos, out = cx.truth()
    xnor = (out["XNOR"] > 0.2).astype(int)
    xnor_ok = list(xnor) == [1, 0, 0, 1]
    log("XNOR = NOT(XOR):")
    log("  combos %s" % [c for c in combos])
    log("  XOR  rates %s" % np.round(out["XOR"], 2))
    log("  XNOR rates %s -> %s  (target [1,0,0,1])" % (np.round(out["XNOR"], 2), list(xnor)))

    # half-adder
    ha = half_adder()
    _, hout = ha.truth()
    s = (hout["SUM(XOR)"] > 0.2).astype(int)
    cy = (hout["CARRY(AND)"] > 0.2).astype(int)
    ha_ok = list(s) == [0, 1, 1, 0] and list(cy) == [0, 0, 0, 1]
    log("\nHalf-adder:")
    log("  SUM   (XOR) %s -> %s  (target [0,1,1,0])" % (np.round(hout["SUM(XOR)"], 2), list(s)))
    log("  CARRY (AND) %s -> %s  (target [0,0,0,1])" % (np.round(hout["CARRY(AND)"], 2), list(cy)))

    # cascade stability: matched vs mismatched interface, margin vs depth
    depth = 10
    r0_m, r1_m = inverter_chain(depth, w_if=0.30)     # matched
    r0_x, r1_x = inverter_chain(depth, w_if=0.22)     # mismatched (marginal -> collapses)
    margin_m = np.abs(r1_m - r0_m)
    margin_x = np.abs(r1_x - r0_x)
    log("\nInverter-chain noise margin (|rate(in=1) - rate(in=0)|) vs depth:")
    log("  matched   (w_if=0.30): start %.2f  end %.2f" % (margin_m[0], margin_m[-1]))
    log("  mismatched(w_if=0.12): start %.2f  end %.2f" % (margin_x[0], margin_x[-1]))

    checks = [
        ("XNOR = NOT(XOR) gives [1,0,0,1]", xnor_ok),
        ("half-adder SUM=XOR and CARRY=AND correct", ha_ok),
        ("matched cascade keeps the margin open at depth 10 (> 0.5)",
         margin_m[-1] > 0.5),
        ("mismatched cascade collapses the margin (< 0.25)", margin_x[-1] < 0.25),
        ("the band restores levels: matched margin does not shrink with depth",
         margin_m[-1] >= margin_m[0] - 0.15),
    ]
    log("\nValidation:")
    for nm, ok in checks:
        log("  [%s] %s" % ("PASS" if ok else "FAIL", nm))
    passed = all(ok for _, ok in checks)
    log("\n%s" % ("ALL CHECKS PASSED" if passed else "SOME CHECKS FAILED"))
    return {"checks_passed": passed, "xnor": out, "combos": combos, "ha": hout,
            "margin_m": margin_m, "margin_x": margin_x,
            "chain_m": (r0_m, r1_m), "chain_x": (r0_x, r1_x)}


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def make_figures(out_prefix="dcap_circuits"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = validate(verbose=False)

    # ---- Figure 1: the two circuits + their node truth tables ---------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    combos = res["combos"]; labels = ["".join(map(str, c)) for c in combos]

    ax = axes[0]
    x = np.arange(4)
    ax.bar(x - 0.2, res["xnor"]["XOR"], 0.4, label="XOR (layer 1)", color="#16a34a")
    ax.bar(x + 0.2, res["xnor"]["XNOR"], 0.4, label="XNOR (layer 2 = NOT)", color="#7c3aed")
    ax.axhline(0.2, color="#9ca3af", ls=":", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_title("XNOR = NOT(XOR)\nsingle neuron can't; two layers can", fontsize=11)
    ax.set_xlabel("input ab"); ax.set_ylabel("firing rate"); ax.legend(fontsize=8)

    ax = axes[1]
    ax.bar(x - 0.2, res["ha"]["SUM(XOR)"], 0.4, label="SUM = XOR", color="#16a34a")
    ax.bar(x + 0.2, res["ha"]["CARRY(AND)"], 0.4, label="CARRY = AND", color="#ea580c")
    ax.axhline(0.2, color="#9ca3af", ls=":", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_title("Half-adder\nSUM=XOR, CARRY=AND", fontsize=11)
    ax.set_xlabel("input ab"); ax.set_ylabel("firing rate"); ax.legend(fontsize=8)

    # ---- cascade stability ----
    ax = axes[2]
    d = np.arange(1, len(res["margin_m"]) + 1)
    ax.plot(d, res["margin_m"], "o-", color="#16a34a", lw=2, label="matched interface")
    ax.plot(d, res["margin_x"], "o-", color="#dc2626", lw=2, label="mismatched interface")
    ax.axhspan(0.5, 1.0, color="#16a34a", alpha=0.07)
    ax.axhspan(0.0, 0.25, color="#dc2626", alpha=0.07)
    ax.set_title("Cascade stability\n(noise margin vs circuit depth)", fontsize=11)
    ax.set_xlabel("depth (# gates in series)"); ax.set_ylabel("noise margin")
    ax.set_ylim(0, 1); ax.legend(fontsize=8.5)

    fig.suptitle("snn2 — composing dCaAP gates: XNOR & half-adder, and why the band "
                 "is what keeps a deep cascade stable", fontsize=12.5, y=1.02)
    fig.tight_layout()
    f1 = out_prefix + "_compose_stability.png"
    fig.savefig(f1, dpi=130, bbox_inches="tight"); plt.close(fig)
    return (f1,)


def main():
    print("Validating dcap_circuits:\n")
    validate()


if __name__ == "__main__":
    main()
