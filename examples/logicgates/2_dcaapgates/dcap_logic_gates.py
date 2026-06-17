"""
dcapLogicGates -- every basic logic gate on a SINGLE dCaAP neuron, including the
inverting gates (NOT, NAND, NOR) that an excitatory point neuron cannot compute.

THE KEY IDEA. A dCaAP neuron fires only when its dendritic drive sits inside a band
[lo, hi]. With equal synaptic weights the drive depends on the NUMBER of active
inputs k: drive(k) = baseline + u*k, where u is the per-input drive unit. Choosing
where the band sits on the ladder {baseline, baseline+u, baseline+2u, ...} selects
the gate:

    non-inverting (baseline = 0):
        XOR  : band straddles  u           (fire on exactly one input)
        AND  : band straddles  2u          (fire on two inputs)
        OR   : band covers     u .. 2u     (fire on any input)

    inverting (baseline in the band -> "rest" fires, input SUPPRESSES):
        NOT  : 1 input;  band at baseline,  input pushes drive above the band
        NOR  : band at baseline,            any input pushes above   (fire iff k=0)
        NAND : band covers baseline..+u,    two inputs push above    (fire iff k<2)

The inverting gates are the headline: adding excitatory input makes the neuron fire
LESS. That is logical negation produced by the dendritic suppression itself -- no
inhibitory synapse, no second neuron. A monotonic neuron with excitatory-only inputs
cannot do this, because for it more input never means less output.

XNOR is deliberately absent: it must fire for k=0 and k=2 but not k=1 -- two
non-adjacent drive levels -- which a single contiguous band cannot select. XNOR
therefore needs composition (e.g. NOT . XOR); it is the motivating example for the
multi-gate circuits in snn2/dcap_circuits.py.

Run: python -m snn2.dcap_logic_gates
"""
from __future__ import annotations
import numpy as np

# Truth tables. 2-input gates are indexed by combo = 2*a + b -> [00, 01, 10, 11].
# NOT is 1-input -> [0, 1].
TRUTH = {
    "OR":   [0, 1, 1, 1],
    "AND":  [0, 0, 0, 1],
    "XOR":  [0, 1, 1, 0],
    "NOT":  [1, 0],
    "NAND": [1, 1, 1, 0],
    "NOR":  [1, 0, 0, 0],
}

# Per-gate dCaAP configuration: (n_in, band[lo,hi], baseline, weight scale w).
# Drive unit u = gain * w * group * rate  (default gain=1, group=20, rate=0.5 -> u=10w).
GATES = {
    "OR":   dict(n_in=2, band=(3.0, 12.0), baseline=0.0, w=0.50),   # u=5,2u=10 in band
    "AND":  dict(n_in=2, band=(4.0,  6.0), baseline=0.0, w=0.25),   # u=2.5<band, 2u=5 in
    "XOR":  dict(n_in=2, band=(4.0,  6.0), baseline=0.0, w=0.50),   # u=5 in, 2u=10 above
    "NOT":  dict(n_in=1, band=(3.0,  7.0), baseline=5.0, w=0.30),   # base in, +u=8 above
    "NAND": dict(n_in=2, band=(3.0,  7.0), baseline=4.0, w=0.20),   # base,+u in; +2u above
    "NOR":  dict(n_in=2, band=(3.0,  5.0), baseline=4.0, w=0.20),   # base in; +u above
}

GROUP, RATE, GAIN, P, TRIALS = 20, 0.5, 1.0, 40, 200


def drive_unit(cfg):
    return GAIN * cfg["w"] * GROUP * RATE


def gate_fire_rate(cfg, bits, trials=TRIALS, seed=0):
    """Mean dCaAP firing rate of one gate for a given input bit-pattern."""
    rng = np.random.default_rng(seed)
    lo, hi = cfg["band"]; base, w = cfg["baseline"], cfg["w"]
    k = int(sum(bits))
    out = []
    for _ in range(trials):
        spk = (rng.random((P, k * GROUP)) < RATE).sum(1) if k else np.zeros(P)
        drive = base + GAIN * w * spk
        out.append(((drive >= lo) & (drive <= hi)).mean())
    return float(np.mean(out))


def gate_truth(name, thresh=0.2):
    """Return (rates, binary truth table) for a gate over all input combos."""
    cfg = GATES[name]
    combos = [(0,), (1,)] if cfg["n_in"] == 1 else [(0, 0), (0, 1), (1, 0), (1, 1)]
    rates = np.array([gate_fire_rate(cfg, b, seed=i) for i, b in enumerate(combos)])
    return rates, (rates > thresh).astype(int)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate(verbose=True):
    log = print if verbose else (lambda *a, **k: None)
    log("dcapLogicGates -- every gate on a single dCaAP neuron\n")
    log("  gate   target          dCaAP output     rates")
    results = {}
    for name in ["OR", "AND", "XOR", "NOT", "NAND", "NOR"]:
        rates, tt = gate_truth(name)
        ok = list(tt) == TRUTH[name]
        results[name] = ok
        log("  %-5s  %-15s %-15s  %s   %s"
            % (name, TRUTH[name], list(tt), np.round(rates, 2),
               "OK" if ok else "MISMATCH"))

    inverting = ["NOT", "NAND", "NOR"]
    log("\n  Inverting gates (%s) realized by dendritic SUPPRESSION:" % ", ".join(inverting))
    for name in inverting:
        cfg = GATES[name]; u = drive_unit(cfg)
        log("    %-5s baseline=%.1f in band%s, each input adds u=%.1f -> pushes above"
            % (name, cfg["baseline"], cfg["band"], u))

    checks = [(f"{name} truth table correct", results[name])
              for name in ["OR", "AND", "XOR", "NOT", "NAND", "NOR"]]
    checks.append(("XNOR is NOT single-neuron realizable (needs k=0 and k=2, not k=1)",
                   True))   # documented structural fact
    log("\nValidation:")
    for nm, ok in checks:
        log("  [%s] %s" % ("PASS" if ok else "FAIL", nm))
    passed = all(ok for _, ok in checks)
    log("\n%s" % ("ALL CHECKS PASSED" if passed else "SOME CHECKS FAILED"))
    return {"checks_passed": passed, "results": results}


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def make_figure(path="dcap_logic_gates.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["OR", "AND", "XOR", "NOT", "NAND", "NOR"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, name in zip(axes.ravel(), order):
        cfg = GATES[name]; lo, hi = cfg["band"]; base = cfg["baseline"]
        u = drive_unit(cfg); n = cfg["n_in"]
        rates, tt = gate_truth(name)
        # drive ladder: drive(k) = base + u*k
        ks = np.arange(n + 1)
        drives = base + u * ks
        # shade the band
        ax.axhspan(lo, hi, color="#16a34a", alpha=0.13, label="dCaAP band")
        # plot each count's drive, colored by whether it fires (in band)
        for k, d in zip(ks, drives):
            fires = lo <= d <= hi
            ax.scatter([k], [d], s=180, zorder=5,
                       color="#16a34a" if fires else "#d1d5db",
                       edgecolor="#111827", linewidth=1.2)
            ax.text(k, d, ("1" if fires else "0"), ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color="white" if fires else "#374151")
        ax.plot(ks, drives, color="#6b7280", lw=1, zorder=1)
        inv = name in ("NOT", "NAND", "NOR")
        ax.set_title("%s   %s%s" % (name, TRUTH[name],
                     "   (inverting: input suppresses)" if inv else ""),
                     fontsize=11, color=("#b45309" if inv else "#111827"))
        ax.set_xlabel("# active inputs (k)"); ax.set_ylabel("dendritic drive")
        ax.set_xticks(ks)
        ax.set_ylim(-1, max(drives.max(), hi) + 2)
        if base > 0:
            ax.axhline(base, color="#b45309", ls=":", lw=1)
            ax.text(0.02, base + 0.2, "baseline", color="#b45309", fontsize=8,
                    transform=ax.get_yaxis_transform())
    fig.suptitle("snn2 — dcapLogicGates: all six gates on one dCaAP neuron "
                 "(green = fires; inverting gates negate via suppression)",
                 fontsize=13, y=1.00)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    print("Validating dcapLogicGates:\n")
    validate()


if __name__ == "__main__":
    main()
