"""
dcap_logic_framework -- a small, standalone toolkit for building, composing,
visualizing, and STABILIZING spiking logic circuits out of dCaAP (band-pass) neurons.

Depends only on numpy (+ matplotlib for the plot helpers). Drop-in usable outside any
larger framework.

Core model
----------
A dCaAP gate is a single neuron that fires only when its dendritic drive lands inside
a band [lo, hi]. With equal weights the drive depends on the number of active inputs k:

    drive(k) = baseline + u * k          u = gain * w * group * rate   (the drive unit)

Choosing where the band sits on the ladder {baseline, baseline+u, baseline+2u, ...}
selects the gate. Inverting gates (NOT/NAND/NOR) put the baseline in-band so that
adding excitatory input pushes drive OUT of the band -> firing decreases with input ->
logical negation by suppression (impossible for an excitatory point neuron).

Two design tools (the reusable heart of the toolkit)
----------------------------------------------------
  design_band(levels, fire_on)   -> a band that fires on exactly the chosen drive
                                    levels, or raises if they are non-contiguous
                                    (meaning: not single-neuron realizable -> compose).
  match_interface(cfg, r_false, r_true) -> the downstream input weight that maps an
                                    upstream gate's (false_rate, true_rate) output onto
                                    the right side of the downstream band -> the thing
                                    that keeps a cascade's noise margin open.

See SKILL.md for the workflow and the validation ladder.
"""
from __future__ import annotations
import numpy as np

GROUP, RATE, GAIN, P_STEPS = 20, 0.5, 1.0, 40


# --------------------------------------------------------------------------- #
# Gate library
# --------------------------------------------------------------------------- #
def gate(n_in, band, baseline=0.0, w=0.5):
    return dict(n_in=n_in, band=tuple(band), baseline=float(baseline), w=float(w))

STD_GATES = {
    "OR":   gate(2, (3.0, 12.0), 0.0, 0.50),
    "AND":  gate(2, (4.0,  6.0), 0.0, 0.25),
    "XOR":  gate(2, (4.0,  6.0), 0.0, 0.50),
    "NOT":  gate(1, (3.0,  7.0), 5.0, 0.30),
    "NAND": gate(2, (3.0,  7.0), 4.0, 0.20),
    "NOR":  gate(2, (3.0,  5.0), 4.0, 0.20),
}
TRUTH = {"OR": [0,1,1,1], "AND": [0,0,0,1], "XOR": [0,1,1,0],
         "NOT": [1,0], "NAND": [1,1,1,0], "NOR": [1,0,0,0]}


def drive_unit(cfg):
    return GAIN * cfg["w"] * GROUP * RATE


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def eval_gate(cfg, line_rates, weights=None, trials=200, seed=0):
    """Firing rate of a gate whose input lines carry the given rates (each line a
    group of GROUP neurons). weights default to the gate's scalar w."""
    rng = np.random.default_rng(seed)
    lo, hi = cfg["band"]; base = cfg["baseline"]
    weights = [cfg["w"]] * len(line_rates) if weights is None else weights
    acc = []
    for _ in range(trials):
        drive = np.full(P_STEPS, base, dtype=float)
        for r, wl in zip(line_rates, weights):
            if r > 0:
                drive += GAIN * wl * (rng.random((P_STEPS, GROUP)) < r).sum(1)
        acc.append(((drive >= lo) & (drive <= hi)).mean())
    return float(np.mean(acc))


def gate_truth(cfg, thresh=0.2):
    combos = [(0,), (1,)] if cfg["n_in"] == 1 else [(0,0),(0,1),(1,0),(1,1)]
    rates = np.array([eval_gate(cfg, [RATE if b else 0.0 for b in c], seed=i)
                      for i, c in enumerate(combos)])
    return rates, (rates > thresh).astype(int)


class Circuit:
    """A DAG of gates. Input spec: ('in', i) primary bit, or ('node', j) upstream out."""
    def __init__(self, n_inputs): self.n_inputs = n_inputs; self.nodes = []
    def add(self, name, cfg, inputs, weights=None):
        self.nodes.append(dict(name=name, cfg=cfg, inputs=inputs, weights=weights))
        return len(self.nodes) - 1
    def evaluate(self, bits, seed=0):
        rates = []
        for ni, nd in enumerate(self.nodes):
            lr = [(RATE if bits[s[1]] else 0.0) if s[0] == "in" else rates[s[1]]
                  for s in nd["inputs"]]
            rates.append(eval_gate(nd["cfg"], lr, nd["weights"], seed=seed + ni))
        return rates
    def truth(self, thresh=0.2, seed=0):
        combos = [tuple((c >> (self.n_inputs-1-i)) & 1 for i in range(self.n_inputs))
                  for c in range(2 ** self.n_inputs)]
        out = {nd["name"]: [] for nd in self.nodes}
        for c in combos:
            for nd, r in zip(self.nodes, self.evaluate(c, seed)):
                out[nd["name"]].append(r)
        return combos, {k: np.array(v) for k, v in out.items()}


# --------------------------------------------------------------------------- #
# Design tools
# --------------------------------------------------------------------------- #
def design_band(levels, fire_on, pad=0.5):
    """Given the sorted drive levels and a boolean mask of which should fire, return
    a band [lo, hi] that fires on exactly those levels. Raises ValueError if the
    firing levels are not contiguous (=> not single-neuron realizable; compose)."""
    levels = np.asarray(levels, float)
    idx = np.where(fire_on)[0]
    if len(idx) == 0:
        raise ValueError("no firing level requested")
    if np.any(np.diff(idx) != 1):
        raise ValueError("firing levels are non-contiguous -> needs composition "
                         "(e.g. XNOR = NOT . XOR)")
    lo = levels[idx[0]] - pad
    hi = levels[idx[-1]] + pad
    return (float(lo), float(hi))


def match_interface(cfg, r_false, r_true, target="suppress"):
    """Downstream input weight so an upstream output (r_false, r_true) lands on the
    right side of this gate's band. target='suppress' (inverting gate): r_true should
    push drive ABOVE hi while r_false stays near baseline. target='excite': r_true
    should land IN-band while r_false stays below lo."""
    lo, hi = cfg["band"]; base = cfg["baseline"]
    # expected added drive per unit weight from a rate r: GAIN * r * GROUP
    d_true = GAIN * r_true * GROUP
    d_false = GAIN * r_false * GROUP
    if target == "suppress":
        # want base + w*d_true > hi  and  base + w*d_false within band
        w = (hi - base + 1.0) / max(d_true, 1e-9)
    else:
        mid = 0.5 * (lo + hi)
        w = (mid - base) / max(d_true, 1e-9)
    return float(w)


def cascade_margin(cfg, depth, w_if, seed=0):
    """Noise margin |rate(in=1) - rate(in=0)| at each stage of a chain of this gate."""
    r0, r1, m = 0.0, RATE, []
    out0, out1 = [], []
    for s in range(depth):
        r0 = eval_gate(cfg, [r0], [w_if], seed=seed + s)
        r1 = eval_gate(cfg, [r1], [w_if], seed=seed + 100 + s)
        out0.append(r0); out1.append(r1); m.append(abs(r1 - r0))
    return np.array(out0), np.array(out1), np.array(m)


# --------------------------------------------------------------------------- #
# Visualization
# --------------------------------------------------------------------------- #
def plot_gate_ladder(ax, name, cfg):
    lo, hi = cfg["band"]; base = cfg["baseline"]; u = drive_unit(cfg); n = cfg["n_in"]
    ks = np.arange(n + 1); drives = base + u * ks
    ax.axhspan(lo, hi, color="#16a34a", alpha=0.13)
    for k, d in zip(ks, drives):
        fires = lo <= d <= hi
        ax.scatter([k], [d], s=160, zorder=5,
                   color="#16a34a" if fires else "#d1d5db",
                   edgecolor="#111827", lw=1.1)
    ax.plot(ks, drives, color="#6b7280", lw=1)
    ax.set_title(name, fontsize=11); ax.set_xlabel("# active inputs"); ax.set_xticks(ks)
    ax.set_ylabel("drive")


def plot_stability(ax, margins: dict):
    for label, (m, col) in margins.items():
        ax.plot(np.arange(1, len(m) + 1), m, "o-", color=col, lw=2, label=label)
    ax.axhspan(0.5, 1.0, color="#16a34a", alpha=0.07)
    ax.axhspan(0.0, 0.25, color="#dc2626", alpha=0.07)
    ax.set_xlabel("depth"); ax.set_ylabel("noise margin"); ax.set_ylim(0, 1)
    ax.legend(fontsize=8.5)


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def self_test():
    ok = True
    for name, cfg in STD_GATES.items():
        _, tt = gate_truth(cfg)
        good = list(tt) == TRUTH[name]
        ok &= good
        print("  %-5s -> %s  %s" % (name, list(tt), "OK" if good else "MISMATCH"))
    # design_band should refuse XNOR (fire on levels 0 and 2, not 1)
    try:
        design_band([0, 1, 2], [True, False, True]); print("  design_band XNOR: NO RAISE (bug)"); ok = False
    except ValueError:
        print("  design_band correctly refuses XNOR (non-contiguous) -> compose")
    print("\n%s" % ("ALL OK" if ok else "FAILURES"))
    return ok


if __name__ == "__main__":
    self_test()
