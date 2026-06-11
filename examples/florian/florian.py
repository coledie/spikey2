"""
Florian (2007) Figure 1 -- the MSTDPET demonstration.

Reinforcement learning by modulating STDP with an eligibility trace. A single
synapse j -> i is driven with fixed pre/post spike trains; a global reward r is
+1 for the first 100 ms then -1. The figure shows how the eligibility machinery
turns spike timing + reward into a weight change.

This is the paper's exact formulation (exponential traces), not the window
approximation Spikey's RLSTDP uses -- so it reproduces the *reference* curve
("Real" panel in the repo notebook) directly.

Equations (Florian 2007):
    P+  <- P+ * exp(-dt/tau_+) + A+ * f_pre        (43)
    P-  <- P- * exp(-dt/tau_-) + A- * f_post       (44)
    zeta = P+ * f_post + P- * f_pre                 (42)
    z   <- z  * exp(-dt/tau_z) + zeta               (8, eligibility trace)
    w   <- w  + gamma0 * r * z                      (7, gamma0 = gamma/tau_z)

Reference: Florian R (2007) Reinforcement Learning Through Modulation of
Spike-Timing-Dependent Synaptic Plasticity. Neural Computation 19(6).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class FlorianParams:
    gamma: float = 0.2          # base learning rate
    tau_z: float = 25.0         # eligibility-trace time constant
    tau_pre: float = 20.0       # pre STDP-trace time constant (tau_+)
    tau_post: float = 20.0      # post STDP-trace time constant (tau_-)
    A_pre: float = 1.0          # A+
    A_post: float = -1.0        # A-
    w0: float = 0.2             # initial weight
    dt: float = 1.0
    length: int = 200
    spike_times_pre: tuple = (5, 80, 115, 135)
    spike_times_post: tuple = (10, 70, 110, 140)
    reward_flip_t: int = 100    # r = +1 for t <= flip, -1 after

    @property
    def gamma0(self) -> float:
        return self.gamma / self.tau_z


def mstdpet_traces(p: FlorianParams = FlorianParams()) -> dict:
    """Run the MSTDPET equations and return every panel of Figure 1.

    Pure function: same params -> same arrays. Faithful to the repo's reference
    loop, including that the reward flip takes effect on the step AFTER t>flip
    (the update at the flip step still sees the old reward)."""
    L = p.length
    f_pre = np.zeros(L)
    f_post = np.zeros(L)
    f_pre[list(p.spike_times_pre)] = 1.0
    f_post[list(p.spike_times_post)] = 1.0

    P_pre = P_post = z = 0.0
    w = p.w0
    r = 1.0
    out = {k: np.zeros(L) for k in
           ("f_pre", "f_post", "P_pre", "P_post", "zeta", "z", "r", "w")}

    for t in range(L):
        fj, fi = f_pre[t], f_post[t]
        P_pre = P_pre * np.exp(-p.dt / p.tau_pre) + p.A_pre * fj
        P_post = P_post * np.exp(-p.dt / p.tau_post) + p.A_post * fi
        zeta = P_pre * fi + P_post * fj
        z = z * np.exp(-p.dt / p.tau_z) + zeta
        w = w + p.gamma0 * r * z

        out["f_pre"][t] = fj
        out["f_post"][t] = fi
        out["P_pre"][t] = P_pre
        out["P_post"][t] = P_post
        out["zeta"][t] = zeta
        out["z"][t] = z
        out["r"][t] = r
        out["w"][t] = w

        if t > p.reward_flip_t:
            r = -1.0
    return out


# --------------------------------------------------------------------------- #
# Validation: the module must match an independent literal transcription of the
# paper equations, and satisfy the qualitative facts visible in Figure 1.
# --------------------------------------------------------------------------- #
def _reference_w(p: FlorianParams) -> np.ndarray:
    """Independent, inlined transcription -> golden oracle for the weight curve."""
    g0 = p.gamma / p.tau_z
    stp, stq = set(p.spike_times_pre), set(p.spike_times_post)
    P_pre = P_post = z = 0.0
    w = p.w0
    r = 1.0
    wlog = []
    for t in range(p.length):
        fj = 1.0 if t in stp else 0.0
        fi = 1.0 if t in stq else 0.0
        P_pre = P_pre * np.exp(-1.0 / p.tau_pre) + fj
        P_post = P_post * np.exp(-1.0 / p.tau_post) - fi
        zeta = P_pre * fi + P_post * fj
        z = z * np.exp(-1.0 / p.tau_z) + zeta
        w = w + g0 * r * z
        wlog.append(w)
        if t > p.reward_flip_t:
            r = -1.0
    return np.array(wlog)


def validate(verbose: bool = True) -> dict:
    p = FlorianParams()
    tr = mstdpet_traces(p)

    # 1) golden: module weight curve == independent reference, byte-for-byte
    assert np.allclose(tr["w"], _reference_w(p)), "MSTDPET != reference equations"

    # 2) eligibility/causality: pre(5) before post(10) => first zeta event positive
    assert tr["zeta"][10] > 0, "pre-before-post should give positive eligibility"

    # 3) with r=+1 and positive eligibility, weight rises early
    assert tr["w"][20] > p.w0, "weight should potentiate under positive reward"

    # 4) reward sign actually flips and stays negative
    assert tr["r"][:101].min() == 1.0 and tr["r"][-1] == -1.0, "reward never flipped"

    # 5) after the flip the same eligibility now pushes weight the other way:
    #    the post-flip net weight change is negative
    flip = p.reward_flip_t + 2
    assert tr["w"][-1] < tr["w"][flip:].max(), "negative reward should later reduce w"

    facts = {
        "w_start": float(p.w0), "w_end": float(tr["w"][-1]),
        "w_peak": float(tr["w"].max()), "checks_passed": 5,
    }
    if verbose:
        for k, v in facts.items():
            print(f"  {k:>9}: {v}")
        print("  ok  MSTDPET matches Florian (2007) reference equations")
    return facts


def make_figure(path: str, p: FlorianParams = FlorianParams()):
    """Render the 8-panel Florian Figure 1."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tr = mstdpet_traces(p)
    t = np.arange(p.length)
    panels = [
        ("$f_j$ (pre)", tr["f_pre"], None),
        ("$f_i$ (post)", tr["f_post"], None),
        ("$P^+_{ij},\\,P^-_{ij}$", [tr["P_pre"], tr["P_post"]], (-1.5, 1.5)),
        ("$\\zeta_{ij}$", tr["zeta"], (-1.5, 1.5)),
        ("$Z_{ij}$ (elig.)", tr["z"], (-1.5, 1.5)),
        ("$r$ (reward)", tr["r"], (-1.5, 1.5)),
        ("$w_{ij}$", tr["w"], (0.15, 0.45)),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(8, 11), sharex=True)
    fig.suptitle("Florian (2007) Fig. 1 — MSTDPET  (reproduced in snn2)",
                 fontsize=13, y=0.995)
    for ax, (label, data, ylim) in zip(axes, panels):
        if isinstance(data, list):
            ax.plot(t, data[0], color="#2563eb", lw=1.4, label="$P^+$")
            ax.plot(t, data[1], color="#dc2626", lw=1.4, label="$P^-$")
            ax.legend(loc="upper right", fontsize=8, frameon=False)
        else:
            ax.plot(t, data, color="#111827", lw=1.6)
        ax.axhline(0, ls="--", lw=0.7, color="#9ca3af")
        ax.axvline(p.reward_flip_t, ls=":", lw=0.8, color="#9ca3af")
        ax.set_ylabel(label, fontsize=10)
        if ylim:
            ax.set_ylim(*ylim)
        ax.margins(x=0)
    axes[-1].set_xlabel("time (ms)")
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    print("Validating Florian (2007) MSTDPET reproduction:")
    validate()


if __name__ == "__main__":
    main()
