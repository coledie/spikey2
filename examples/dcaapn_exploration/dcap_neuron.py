"""
A two-compartment dCaAP neuron: a dendritic calcium spike driving a somatic
Izhikevich spike generator -- the biophysically-flavored version of Gidon et al.
(2020), "Dendritic action potentials and computation in human layer 2/3 cortical
neurons" (Science 367:83-87).

THE SIGNATURE. Gidon et al. found a dendritic Ca2+ action potential (dCaAP) whose
amplitude is MAXIMAL for threshold-level input and DECREASES for stronger input --
a non-monotonic, band-pass activation, the opposite of a normal neuron. A single
such neuron computes XOR. Here we model it with two compartments:

  dendrite : produces a dCaAP whose amplitude is a non-monotonic ("hump") function
             of dendritic drive -- on above a low threshold, OFF again above a high
             one. This is the Gidon activation curve.
  soma     : a regular-spiking Izhikevich (2003) neuron driven by the dCaAP current,
             producing the Na+ output spikes.

THE RESULT a single point neuron cannot produce: a NON-MONOTONIC f-I curve. The
somatic firing rate rises with input, peaks at threshold-level drive, then FALLS
for strong drive (the dCaAP is suppressed). A standard Izhikevich RS neuron, by
contrast, has a monotonic, saturating f-I curve.

This is the shared primitive behind the dCaAP logic-gate, negative-patterning, and
florian2 demos: a tunable "moderate-drive / exactly-one" detector.

Run: python -m snn2.dcap_neuron
"""
from __future__ import annotations
import numpy as np


P = dict(
    # Izhikevich RS soma
    a=0.02, b=0.2, c=-65.0, d=8.0,
    # dendritic dCaAP activation (non-monotonic in dendritic drive I_d)
    dcap_lo=4.0, dcap_hi=11.0, k_lo=0.8, k_hi=1.2,   # band edges / sharpness
    g_dcap=22.0,                # dCaAP -> soma coupling (mA at full activation)
    tau_ca=6.0,                 # dCaAP (Ca gate) time constant, ms
    leak=0.06,                  # small monotonic dendritic leak to soma
)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def dcap_activation(I_d, p=P):
    """Gidon dCaAP amplitude vs dendritic drive: a hump. ON above dcap_lo, OFF
    again above dcap_hi -> maximal at threshold, suppressed for strong input."""
    on = _sigmoid((I_d - p["dcap_lo"]) / p["k_lo"])
    off = _sigmoid((p["dcap_hi"] - I_d) / p["k_hi"])
    return on * off


def simulate(I_d, p=P, T=600, dt=1.0, seed=0):
    """Run the two-compartment neuron under a constant dendritic drive I_d for T ms.
    Returns somatic spike count and the membrane / dCaAP traces."""
    rng = np.random.default_rng(seed)
    a, b, c, d = p["a"], p["b"], p["c"], p["d"]
    v, u = c, b * c
    m = 0.0                                   # dCaAP Ca-gate
    m_inf = dcap_activation(I_d, p)
    dec = np.exp(-dt / p["tau_ca"])
    vt, mt = np.zeros(T), np.zeros(T)
    spikes = 0
    for t in range(T):
        m = m * dec + (1 - dec) * m_inf       # dendritic dCaAP relaxes to its amplitude
        I_soma = p["g_dcap"] * m + p["leak"] * I_d + rng.normal(0, 0.6)
        for _ in range(2):                    # 2 x 0.5 ms Euler substeps
            v = v + 0.5 * (0.04 * v * v + 5 * v + 140 - u + I_soma)
        u = u + a * (b * v - u)
        if v >= 30.0:
            v, u = c, u + d
            spikes += 1
        vt[t], mt[t] = min(v, 30.0), m
    return {"spikes": spikes, "rate": spikes / (T * dt / 1000.0), "v": vt, "m": mt}


def izhikevich_rs(I, p=P, T=600, dt=1.0, seed=0):
    """A plain RS Izhikevich neuron for the monotonic-f-I comparison."""
    rng = np.random.default_rng(seed)
    a, b, c, d = p["a"], p["b"], p["c"], p["d"]
    v, u = c, b * c
    spikes = 0
    for t in range(T):
        Ii = I + rng.normal(0, 0.6)
        for _ in range(2):
            v = v + 0.5 * (0.04 * v * v + 5 * v + 140 - u + Ii)
        u = u + a * (b * v - u)
        if v >= 30.0:
            v, u = c, u + d
            spikes += 1
    return spikes / (T * dt / 1000.0)


def fi_curve(drives, p=P, T=800):
    dcap = np.array([simulate(I, p, T)["rate"] for I in drives])
    mono = np.array([izhikevich_rs(I, p, T) for I in drives])
    return dcap, mono


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate(verbose=True):
    log = print if verbose else (lambda *a, **k: None)
    drives = np.linspace(0, 22, 45)
    dcap, mono = fi_curve(drives)

    peak_i = int(np.argmax(dcap))
    peak_drive = drives[peak_i]
    high_rate = dcap[drives >= 18].mean()
    peak_rate = dcap[peak_i]

    log("Two-compartment dCaAP neuron -- non-monotonic f-I curve\n")
    log("  dCaAP f-I  : peak %.0f Hz at drive=%.1f, then %.0f Hz at strong drive"
        % (peak_rate, peak_drive, high_rate))
    log("  RS f-I     : %.0f Hz at strong drive (monotonic, saturating)"
        % mono[drives >= 18].mean())

    checks = [
        ("dCaAP f-I peaks at INTERIOR drive (not the maximum)",
         2 < peak_i < len(drives) - 4),
        ("dCaAP fires at threshold-level drive (peak > 10 Hz)", peak_rate > 10),
        ("strong drive SUPPRESSES the dCaAP (high < 0.4*peak)",
         high_rate < 0.4 * peak_rate),
        ("the comparison RS neuron is monotonic (strong >= mid)",
         mono[drives >= 18].mean() >= mono[(drives > 6) & (drives < 12)].mean() - 1),
        ("activation curve is a hump (max in interior)",
         2 < int(np.argmax(dcap_activation(drives))) < len(drives) - 4),
    ]
    log("\nValidation:")
    for name, ok in checks:
        log("  [%s] %s" % ("PASS" if ok else "FAIL", name))
    passed = all(ok for _, ok in checks)
    log("\n%s" % ("ALL CHECKS PASSED" if passed else "SOME CHECKS FAILED"))
    return {"checks_passed": passed, "drives": drives, "dcap": dcap, "mono": mono,
            "peak_drive": peak_drive, "peak_rate": peak_rate}


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def make_figure(path="dcap_two_compartment.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    drives = np.linspace(0, 22, 45)
    dcap, mono = fi_curve(drives)
    act = dcap_activation(drives)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))

    # (A) the dendritic dCaAP activation curve (Gidon signature)
    ax = axes[0]
    ax.plot(drives, act, color="#16a34a", lw=2.5)
    ax.fill_between(drives, 0, act, color="#16a34a", alpha=0.12)
    ax.axvspan(P["dcap_lo"], P["dcap_hi"], color="#16a34a", alpha=0.06)
    ax.set_title("Dendritic dCaAP activation\n(amplitude vs drive — non-monotonic)",
                 fontsize=11)
    ax.set_xlabel("dendritic drive  I_d  (mA)")
    ax.set_ylabel("dCaAP amplitude")
    ax.text(P["dcap_lo"] + 0.3, 0.05, "ON", color="#15803d", fontsize=9)
    ax.text(P["dcap_hi"] + 0.3, 0.05, "suppressed", color="#b91c1c", fontsize=9)

    # (B) somatic membrane traces at three drives: sub / threshold / strong
    ax = axes[1]
    labels = [("sub-threshold", 2.0, "#9ca3af"),
              ("threshold (dCaAP)", 7.0, "#16a34a"),
              ("strong (suppressed)", 18.0, "#b91c1c")]
    for i, (lab, I, col) in enumerate(labels):
        v = simulate(I, P, T=300)["v"]
        ax.plot(np.arange(300), v + i * 95, color=col, lw=0.9, label="%s (I=%.0f)" % (lab, I))
    ax.set_title("Somatic membrane potential\n(only threshold drive spikes)", fontsize=11)
    ax.set_xlabel("time (ms)"); ax.set_yticks([])
    ax.legend(fontsize=8, loc="upper right")

    # (C) the non-monotonic f-I curve vs a monotonic RS neuron
    ax = axes[2]
    ax.plot(drives, dcap, color="#16a34a", lw=2.5, label="two-compartment dCaAP")
    ax.plot(drives, mono, color="#6b7280", lw=2, ls="--", label="standard RS (Izhikevich)")
    pk = int(np.argmax(dcap))
    ax.scatter([drives[pk]], [dcap[pk]], color="#16a34a", zorder=5, s=40)
    ax.annotate("peak at\nthreshold drive", (drives[pk], dcap[pk]),
                (drives[pk] + 1.5, dcap[pk] - 12), fontsize=8.5, color="#15803d")
    ax.set_title("f-I curve: dCaAP is non-monotonic", fontsize=11)
    ax.set_xlabel("input drive  (mA)"); ax.set_ylabel("firing rate (Hz)")
    ax.legend(fontsize=9, loc="upper left")

    fig.suptitle("snn2 — two-compartment dCaAP neuron (Gidon et al. 2020): "
                 "a single neuron with a non-monotonic f-I curve", fontsize=12.5, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    print("Validating two-compartment dCaAP neuron:\n")
    validate()


if __name__ == "__main__":
    main()
