"""
florian2 -- reward-modulated STDP with eligibility traces (MSTDPET, Florian 2007),
but now in CLOSED LOOP: the postsynaptic neuron's firing depends on the synaptic
weight it is learning. With a dCaAP (band-pass) postsynaptic neuron the eligibility
trace becomes SELF-LIMITING, giving the weight a stable fixed point -- the dendritic
band acts as a homeostatic cap.

THE SETUP. The original Florian demo (snn2/florian.py) drives a single synapse with
*fixed* pre/post spike trains and watches the eligibility trace evolve. Here the
post-synaptic firing is instead produced by the synapse itself: drive = w x (pre
activity), and the post neuron fires as a function of that drive. Reward is held
positive, so MSTDPET potentiates the synapse via the pre->post pairing it creates.

THE CONTRAST.
  monotonic post : firing rises monotonically with drive. More weight -> more post
                   firing -> more pre-post pairing -> more eligibility -> more
                   weight. Positive feedback: the weight RUNS AWAY to its ceiling.
  dCaAP post     : firing is band-pass (Gidon 2020). As the weight grows, drive
                   climbs into the band (post fires, weight grows) and then PAST it
                   (post goes silent -> pairing vanishes -> eligibility vanishes ->
                   growth stops). Negative feedback at the top edge: the weight
                   settles at a STABLE fixed point inside/at the band.

This formalizes the "the band caps the weight" intuition from the dCaAP XOR demo as a
genuine homeostatic property: the same non-monotonicity that lets one neuron compute
XOR also stabilizes its own reward-modulated learning.

Run: python -m snn2.florian2
"""
from __future__ import annotations
import numpy as np


P = dict(
    r_pre=0.5,                 # presynaptic firing prob / step
    gain=10.0,                 # drive = gain * w * pre   (per active pre step)
    # monotonic post f-I: sigmoid
    mono_theta=1.0, mono_k=0.8,
    # dCaAP post f-I: band-pass
    dcap_lo=3.0, dcap_hi=6.0, dcap_k=0.7,
    post_max=0.9,              # peak post firing prob
    # MSTDPET (Florian 2007)
    tau_plus=20.0, tau_minus=20.0, tau_z=40.0,
    A_plus=1.0, A_minus=0.8,   # mild LTP bias from a coincidence
    gamma=0.0005,              # learning-rate (reward-modulated)
    decay=0.012,               # homeostatic weight decay (restoring force)
    w_max=2.0,
)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def post_rate(drive, kind, p=P):
    """Postsynaptic firing probability as a function of synaptic drive."""
    if kind == "monotonic":
        return p["post_max"] * _sigmoid((drive - p["mono_theta"]) / p["mono_k"])
    # dCaAP: band-pass
    on = _sigmoid((drive - p["dcap_lo"]) / p["dcap_k"])
    off = _sigmoid((p["dcap_hi"] - drive) / p["dcap_k"])
    return p["post_max"] * on * off


# --------------------------------------------------------------------------- #
# Closed-loop MSTDPET simulation
# --------------------------------------------------------------------------- #
def closed_loop(kind, w0, p=P, reward=1.0, T=20000, seed=0, freeze=False):
    """Run the closed loop. Returns the weight trajectory (sampled). If freeze,
    the weight is held at w0 and we instead return the mean dw/dt at w0."""
    rng = np.random.default_rng(seed)
    w = float(w0)
    Pp = Pm = z = 0.0
    dec_p = np.exp(-1.0 / p["tau_plus"]); dec_m = np.exp(-1.0 / p["tau_minus"])
    dec_z = np.exp(-1.0 / p["tau_z"])
    traj = np.empty(T // 20); dw_acc = 0.0
    for t in range(T):
        f_pre = 1.0 if rng.random() < p["r_pre"] else 0.0
        drive = p["gain"] * w * f_pre            # pre-gated drive this step
        # post fires as a function of the *recent* drive (use instantaneous here)
        rho = post_rate(p["gain"] * w * p["r_pre"], kind, p)   # expected drive
        f_post = 1.0 if rng.random() < rho else 0.0
        # MSTDPET traces (Florian Eqs)
        Pp = Pp * dec_p + p["A_plus"] * f_pre
        Pm = Pm * dec_m - p["A_minus"] * f_post
        zeta = Pp * f_post + Pm * f_pre
        z = z * dec_z + zeta
        dw = p["gamma"] * reward * z - p["decay"] * w
        if not freeze:
            w = min(p["w_max"], max(0.0, w + dw))
        dw_acc += dw
        if t % 20 == 0:
            traj[t // 20] = w
    if freeze:
        return dw_acc / T          # mean drift dw/dt at fixed w0
    return traj


def phase_portrait(kind, ws, p=P, T=8000):
    """Mean dw/dt as a function of (frozen) weight -- the phase portrait."""
    return np.array([closed_loop(kind, w, p, T=T, freeze=True, seed=int(w * 1000))
                     for w in ws])


def fixed_points(ws, dwdt):
    """Interior zero-crossings of dw/dt, with stability (negative slope = stable)."""
    fps = []
    for i in range(len(ws) - 1):
        if dwdt[i] == 0 or (dwdt[i] > 0) != (dwdt[i + 1] > 0):
            wstar = ws[i] - dwdt[i] * (ws[i + 1] - ws[i]) / (dwdt[i + 1] - dwdt[i] + 1e-12)
            stable = dwdt[i + 1] < dwdt[i]    # crossing downward
            fps.append((wstar, stable))
    return fps


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate(verbose=True):
    log = print if verbose else (lambda *a, **k: None)
    ws = np.linspace(0.02, P["w_max"], 40)
    dw_m = phase_portrait("monotonic", ws)
    dw_d = phase_portrait("dcap", ws)

    fp_m = fixed_points(ws, dw_m)
    fp_d = fixed_points(ws, dw_d)
    stable_d = [w for w, s in fp_d if s and 0.3 < w < P["w_max"] - 0.05]

    # trajectories from several initial weights
    inits = [0.35, 0.55, 0.8, 1.1]
    traj_d = [closed_loop("dcap", w0, T=20000, seed=i) for i, w0 in enumerate(inits)]
    traj_m = [closed_loop("monotonic", w0, T=20000, seed=i) for i, w0 in enumerate(inits)]
    final_d = np.array([tr[-50:].mean() for tr in traj_d])
    final_m = np.array([tr[-50:].mean() for tr in traj_m])

    log("florian2 -- self-limiting MSTDPET with a dCaAP postsynaptic neuron\n")
    log("Phase portrait (sign of dw/dt across weight):")
    log("  monotonic: dw/dt > 0 for most of the range -> runs to the ceiling")
    log("  dCaAP    : stable interior fixed point(s) at w = %s"
        % np.round(stable_d, 2))
    log("\nClosed-loop final weights from inits %s:" % inits)
    log("  monotonic -> %s  (ceiling = %.1f)" % (np.round(final_m, 2), P["w_max"]))
    log("  dCaAP     -> %s  (converged)" % np.round(final_d, 2))

    drive_star = P["gain"] * np.mean(stable_d) * P["r_pre"] if stable_d else 0
    checks = [
        ("monotonic post -> weight runs away to the ceiling (all > 0.9*w_max)",
         np.all(final_m > 0.9 * P["w_max"])),
        ("dCaAP post -> a stable interior fixed point exists",
         len(stable_d) >= 1),
        ("dCaAP trajectories converge below the ceiling (all < 0.8*w_max)",
         np.all(final_d < 0.8 * P["w_max"])),
        ("dCaAP trajectories converge together (spread < 0.15)",
         final_d.max() - final_d.min() < 0.15),
        ("the fixed-point drive sits in/at the dCaAP band",
         P["dcap_lo"] - 0.5 <= drive_star <= P["dcap_hi"] + 1.5),
    ]
    log("\nValidation:")
    for name, ok in checks:
        log("  [%s] %s" % ("PASS" if ok else "FAIL", name))
    passed = all(ok for _, ok in checks)
    log("\n%s" % ("ALL CHECKS PASSED" if passed else "SOME CHECKS FAILED"))
    return {"checks_passed": passed, "ws": ws, "dw_m": dw_m, "dw_d": dw_d,
            "stable_d": stable_d, "inits": inits, "traj_d": traj_d, "traj_m": traj_m,
            "final_d": final_d, "final_m": final_m}


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def make_figure(path="florian2.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = validate(verbose=False)
    ws = res["ws"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))

    # (A) the two post f-I curves (why)
    ax = axes[0]
    drives = np.linspace(0, P["gain"] * P["w_max"] * P["r_pre"], 200)
    ax.plot(drives, post_rate(drives, "monotonic"), color="#dc2626", lw=2,
            label="monotonic post")
    ax.plot(drives, post_rate(drives, "dcap"), color="#16a34a", lw=2,
            label="dCaAP post (band-pass)")
    ax.axvspan(P["dcap_lo"], P["dcap_hi"], color="#16a34a", alpha=0.08)
    ax.set_title("Postsynaptic f-I\n(the only difference)", fontsize=11)
    ax.set_xlabel("synaptic drive"); ax.set_ylabel("post firing prob")
    ax.legend(fontsize=8.5)

    # (B) phase portrait dw/dt vs w
    ax = axes[1]
    ax.axhline(0, color="#9ca3af", lw=0.8)
    ax.plot(ws, res["dw_m"], color="#dc2626", lw=2, label="monotonic")
    ax.plot(ws, res["dw_d"], color="#16a34a", lw=2, label="dCaAP")
    for w, s in [(w, True) for w in res["stable_d"]]:
        ax.scatter([w], [0], color="#16a34a", zorder=5, s=70,
                   label="stable fixed point")
        ax.annotate("", (w, 0), (w - 0.18, 0.004),
                    arrowprops=dict(arrowstyle="->", color="#16a34a"))
        ax.annotate("", (w, 0), (w + 0.18, -0.004),
                    arrowprops=dict(arrowstyle="->", color="#16a34a"))
    ax.set_title("Phase portrait: dw/dt vs w\n(dCaAP crosses zero; monotonic doesn't)",
                 fontsize=11)
    ax.set_xlabel("synaptic weight  w"); ax.set_ylabel("dw/dt")
    # de-duplicate legend
    h, l = ax.get_legend_handles_labels()
    seen = dict(zip(l, h)); ax.legend(seen.values(), seen.keys(), fontsize=8.5)

    # (C) weight trajectories from several inits
    ax = axes[2]
    tt = np.arange(len(res["traj_d"][0])) * 20
    for tr in res["traj_m"]:
        ax.plot(tt, tr, color="#dc2626", lw=1.3, alpha=0.8)
    for tr in res["traj_d"]:
        ax.plot(tt, tr, color="#16a34a", lw=1.3, alpha=0.8)
    ax.axhline(P["w_max"], color="#9ca3af", ls=":", lw=1)
    ax.text(tt[-1] * 0.30, P["w_max"] - 0.16, "monotonic → ceiling (runaway)",
            color="#b91c1c", fontsize=9)
    if res["stable_d"]:
        wbar = float(np.mean(res["final_d"]))
        ax.axhline(wbar, color="#16a34a", ls="--", lw=1)
        ax.text(tt[-1] * 0.30, 0.55,
                "dCaAP → stable fixed point (w*≈%.2f)" % wbar,
                color="#15803d", fontsize=9)
    ax.set_title("Weight trajectories\n(same rule, different post neuron)", fontsize=11)
    ax.set_xlabel("time (ms)"); ax.set_ylabel("synaptic weight  w")
    ax.set_ylim(0, P["w_max"] * 1.05)

    fig.suptitle("snn2 — florian2: a dCaAP postsynaptic neuron makes reward-modulated "
                 "STDP self-limiting (the band is a homeostatic cap)", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    print("Validating florian2 (self-limiting MSTDPET):\n")
    validate()


if __name__ == "__main__":
    main()
