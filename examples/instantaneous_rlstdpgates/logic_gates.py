"""
Logic gates with reward-modulated learning -- the INSTANTANEOUS reproduction.

This is the closed-form twin of `examples/logicgates/logic_gates.py`. Same
curriculum, same conclusions, but every trial is solved with algebra instead of
a 30-step spiking simulation:

  1. OR   (monotonic) -- linearly separable -> learned (~1.0).
  2. AND  (monotonic) -- coincidence detection a single rate neuron can't place
                         reliably -> plateaus (~0.75).
  3. XOR  (monotonic) -- not linearly separable -> stuck near chance (<0.8).
  4. XOR  (dendritic) -- the non-monotonic dCaAP band-pass neuron computes XOR
                         from one unit (Gidon et al. 2020) -> solved (~1.0).

The monotonic neuron is `transfer.lif_rate` (the exact f-I curve of the discrete
leaky integrator); the dendritic neuron is `transfer.band_rate` (the analytic
band-occupancy probability of a Gaussian drive). No `processing_time` loop runs
anywhere on the learning path.

Run: python -m examples2.logic_gates
"""
from __future__ import annotations
import numpy as np

from .engine import run_instant
from .spec import expand, GATES, _operand_rate_map
from . import transfer as T


def train_gate(preset: str, gate: str, n_seeds: int = 12, length: int | None = None):
    spec = {"preset": preset, "target_map": GATES[gate]}
    if length is not None:
        spec["len_episode"] = length
    p = expand(spec)
    m = run_instant(p, n_seeds=n_seeds, seed=0, log_trials=True)
    curves = m["trial_correct"]                          # [n_seeds, T]
    final = curves[:, -200:].mean(1)                     # per seed
    weights = m["weights"][:, :, 0]                       # [n_seeds, n_in]
    return curves.mean(0), final, weights


def curriculum(n_seeds: int = 12):
    stages = [
        ("OR  (monotonic)",  "logic_monotonic", "OR"),
        ("AND (monotonic)",  "logic_monotonic", "AND"),
        ("XOR (monotonic)",  "logic_monotonic", "XOR"),
        ("XOR (dendritic)",  "logic_dendritic", "XOR"),
    ]
    out = {}
    for label, preset, gate in stages:
        curve, final, W = train_gate(preset, gate, n_seeds)
        out[label] = {"curve": curve, "final": final, "weights": W,
                      "preset": preset, "gate": gate}
    return out


def dendritic_firing_map(wscales, gain=1.0, lo=3.0, hi=6.0, n_in=40, rate=0.5):
    """Closed-form per-combo dendritic firing rate vs synaptic scale -- the same
    XOR-window plot as the spiking demo, but computed with `band_rate` (no
    sampling). One input lands in the band; two inputs overshoot it."""
    rm = _operand_rate_map(n_in, rate)
    out = {lbl: [] for lbl in ("00", "01", "10", "11")}
    for w in wscales:
        W = np.full((n_in, 1), w)
        for combo, lbl in zip(range(4), ("00", "01", "10", "11")):
            mean, var = T.input_moments(rm[combo], W, gain)
            out[lbl].append(float(T.band_rate(mean, var, lo, hi)[0]))
    return {k: np.array(v) for k, v in out.items()}


def gate_truth(preset, gate, seed=0):
    """The trained network's action on each of the 4 combos -- evaluated in closed
    form from the learned weights (no spiking)."""
    p = expand({"preset": preset, "target_map": GATES[gate]})
    m = run_instant(p, n_seeds=1, seed=seed)
    W = m["weights"][0]                                  # [n_in, 1]
    rm = _operand_rate_map(p["n_inputs"], p.get("operand_rate", 0.5))
    gain = p["input_gain"]
    acts = []
    for combo in range(4):
        mean, var = T.input_moments(rm[combo], W, gain)
        if p["neuron"] == "dendritic":
            f = T.band_rate(mean, var, p["dcap_lo"], p["dcap_hi"])[0]
        else:
            f = T.lif_rate(mean, p["potential_decay"], p["firing_threshold"],
                           p["resting_mv"])[0]
        acts.append(int(f > p["action_threshold"]))
    return acts


def validate(n_seeds: int = 12, verbose: bool = True):
    res = curriculum(n_seeds)
    log = print if verbose else (lambda *a, **k: None)

    or_final = res["OR  (monotonic)"]["final"].mean()
    and_final = res["AND (monotonic)"]["final"].mean()
    xm_final = res["XOR (monotonic)"]["final"].mean()
    xd_curve = res["XOR (dendritic)"]["curve"]
    xd_final = res["XOR (dendritic)"]["final"]
    xd_start = xd_curve[:20].mean()
    xd_conv = (xd_final > 0.9).mean()

    log("Instantaneous curriculum (mean final accuracy over %d seeds):" % n_seeds)
    for k in res:
        log("  %-16s  start=%.2f  final=%.2f"
            % (k, res[k]["curve"][:20].mean(), res[k]["final"].mean()))

    checks = []
    checks.append(("dendritic XOR starts not knowing (start < 0.92)", xd_start < 0.92))
    checks.append(("monotonic OR is learned (final > 0.9)", or_final > 0.9))
    checks.append(("monotonic AND plateaus below mastery (0.6 < final < 0.85)",
                   0.6 < and_final < 0.85))
    checks.append(("monotonic XOR fails at the wall (final < 0.8)", xm_final < 0.8))
    checks.append(("dendritic XOR is solved (final > 0.9)", xd_final.mean() > 0.9))
    checks.append(("dendritic XOR converges for most seeds (>=80%)", xd_conv >= 0.8))
    checks.append(("dendrite beats point neuron on XOR by >0.3",
                   xd_final.mean() - xm_final > 0.3))
    checks.append(("dendritic XOR improves with training (end > start + 0.05)",
                   xd_curve[-200:].mean() > xd_start + 0.05))
    tt = gate_truth("logic_dendritic", "XOR", seed=0)
    checks.append(("trained dendritic truth table == XOR [0,1,1,0] (got %s)" % tt,
                   tt == GATES["XOR"]))

    log("\nValidation:")
    for name, ok in checks:
        log("  [%s] %s" % ("PASS" if ok else "FAIL", name))
    passed = all(ok for _, ok in checks)
    log("\n%s" % ("ALL CHECKS PASSED" if passed else "SOME CHECKS FAILED"))
    return {"checks_passed": passed, "results": res,
            "or": or_final, "and": and_final,
            "xor_mono": xm_final, "xor_dend": xd_final.mean(),
            "xor_dend_start": xd_start, "truth_table": tt}


def make_figure(path="logic_gates_instant.png", n_seeds: int = 12):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = curriculum(n_seeds)
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1.0], hspace=0.42, wspace=0.32)
    colors = {"OR  (monotonic)": "#2563eb", "AND (monotonic)": "#0891b2",
              "XOR (monotonic)": "#dc2626", "XOR (dendritic)": "#16a34a"}

    ax = fig.add_subplot(gs[0, :2])
    for k, d in res.items():
        c = d["curve"]
        sm = np.convolve(c, np.ones(40) / 40, mode="valid")
        ax.plot(sm, color=colors[k], lw=2, label="%s  ->%.2f" % (k, d["final"].mean()))
    ax.axhline(0.5, color="#9ca3af", lw=0.8, ls=":")
    ax.axhline(1.0, color="#d1d5db", lw=0.8, ls="--")
    ax.set_title("Instantaneous R-modulated curriculum: point neuron degrades, "
                 "dendrite rescues XOR", fontsize=12)
    ax.set_xlabel("trial"); ax.set_ylabel("accuracy (40-trial avg)")
    ax.set_ylim(0.2, 1.05); ax.legend(loc="lower right", fontsize=9, framealpha=0.95)

    ax = fig.add_subplot(gs[0, 2]); ax.axis("off")
    combos = ["00", "01", "10", "11"]
    rows = [("OR", "logic_monotonic"), ("AND", "logic_monotonic"),
            ("XOR", "logic_monotonic"), ("XOR\u2020", "logic_dendritic")]
    ax.set_title("Learned truth tables\n(green = matches target)", fontsize=11)
    ax.text(0.30, 0.92, "  ".join(combos), family="monospace", fontsize=11,
            transform=ax.transAxes)
    for r, (name, preset) in enumerate(rows):
        y = 0.78 - r * 0.18
        gate = name.replace("\u2020", "")
        learned = gate_truth(preset, gate, seed=0)
        target = GATES[gate]
        ax.text(0.0, y, "%-5s" % name, family="monospace", fontsize=11,
                transform=ax.transAxes)
        for i, (lv, tv) in enumerate(zip(learned, target)):
            col = "#16a34a" if lv == tv else "#dc2626"
            ax.text(0.30 + i * 0.165, y, str(lv), family="monospace", fontsize=12,
                    color=col, fontweight="bold", transform=ax.transAxes)
    ax.text(0.0, 0.06, "\u2020 dendritic (dCaAP) neuron", fontsize=8.5,
            color="#16a34a", transform=ax.transAxes)

    ax = fig.add_subplot(gs[1, :2])
    ws = np.linspace(0.05, 1.2, 60)
    fmap = dendritic_firing_map(ws, gain=1.0, lo=3.0, hi=6.0)
    ax.plot(ws, fmap["00"], color="#9ca3af", lw=2, label="00 (no input)")
    ax.plot(ws, (fmap["01"] + fmap["10"]) / 2, color="#16a34a", lw=2.5,
            label="one input (01/10)")
    ax.plot(ws, fmap["11"], color="#b91c1c", lw=2.5, label="two inputs (11)")
    one = (fmap["01"] + fmap["10"]) / 2
    win = ws[(one > 0.15) & (fmap["11"] < 0.10)]
    if len(win):
        ax.axvspan(win.min(), win.max(), color="#16a34a", alpha=0.10)
        ax.text((win.min() + win.max()) / 2, 0.9, "XOR window", ha="center",
                color="#15803d", fontsize=10, fontweight="bold")
    ax.set_title("Why the dendrite computes XOR (closed-form band probability)",
                 fontsize=11)
    ax.set_xlabel("synaptic weight scale"); ax.set_ylabel("dendritic firing rate")
    ax.legend(loc="upper right", fontsize=9); ax.set_ylim(0, 1.0)

    ax = fig.add_subplot(gs[1, 2])
    xm = res["XOR (monotonic)"]["final"].mean()
    xd = res["XOR (dendritic)"]["final"].mean()
    ax.bar(["point\nneuron", "dendritic\n(dCaAP)"], [xm, xd],
           color=["#dc2626", "#16a34a"])
    ax.axhline(0.5, color="#9ca3af", lw=0.8, ls=":")
    ax.set_title("XOR: same task,\nsame learning rule", fontsize=11)
    ax.set_ylabel("final accuracy"); ax.set_ylim(0, 1.1)
    for i, v in enumerate([xm, xd]):
        ax.text(i, v + 0.03, "%.2f" % v, ha="center", fontsize=11, fontweight="bold")

    fig.suptitle("examples2 \u2014 instantaneous logic gates and dendritic XOR "
                 "(closed-form, no spiking loop)", fontsize=13, y=0.99)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    print("Validating instantaneous logic-gate curriculum:\n")
    validate()


if __name__ == "__main__":
    main()
