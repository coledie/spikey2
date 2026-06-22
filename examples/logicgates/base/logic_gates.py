"""
Learning logic gates with reward-modulated STDP -- and why XOR needs a dendrite.

THE CURRICULUM (easy -> impossible -> rescued). One output neuron is shown the
four combinations of two binary operands a, b (encoded as two input groups) and
rewarded when its action matches a target gate's truth table. We train it, from
random initial weights, on a difficulty-ordered sequence:

  1. OR   (monotonic neuron) -- linearly separable, low threshold. Solved almost
                               immediately: firing on any input already *is* OR.
  2. AND  (monotonic neuron) -- linearly separable but needs coincidence
                               detection (fire on two inputs, not one). A single
                               rate-coded point neuron + reward-modulated STDP
                               cannot reliably place a threshold in that narrow
                               window; it learns to stay silent and plateaus
                               around 0.75 (right on 00/01/10, wrong on 11).
  3. XOR  (monotonic neuron) -- NOT linearly separable. A point neuron is stuck
                               at chance (~0.5): no single threshold on a
                               monotonic drive can fire for one input yet not for
                               two, because two inputs always drive *harder*.
  4. XOR  (dendritic neuron) -- SOLVED. Gidon et al. 2020 (Science) showed human
                               layer 2/3 pyramidal neurons fire dendritic Ca2+
                               action potentials (dCaAPs) whose amplitude is
                               MAXIMAL near threshold and DECREASES for stronger
                               input -- an anti-coincidence / band-pass response.
                               A single such neuron computes XOR: silent at zero
                               input (below the band), fires at one input (drive
                               in the band), suppressed at two inputs (drive
                               overshoots the band). R-STDP tunes the synaptic
                               scale so the one- vs two-input drive levels straddle
                               the band, and the neuron learns XOR (~1.0).

THE POINT. Difficulty rises and the point neuron degrades (1.0 -> 0.75 -> 0.5);
swapping ONLY the neuron's activation -- monotonic threshold -> non-monotonic
dCaAP band -- breaks the XOR wall. The synapses, encoding, reward, and learning
rule are identical across 3 and 4; the dendrite is what changes.

A learning subtlety worth knowing (it shaped the dendritic preset): pure LTP with
a band-pass neuron has a "band-entry-order trap." At low weight the *two-input*
drive is highest and sits in the band, so it fires and -- if punished -- crushes
every weight to zero before one-input drive can climb in. The fix is reward-only
learning (punish_mult = 0) plus a weight cap inside the XOR window, so reward
ratchets the weight up into the band and clipping stops the overshoot. Weights are
also feedforward-only (no body->body self-excitation).

Run: python -m snn2.logic_gates
"""
from __future__ import annotations
import numpy as np

from snn2.engines.trial import run_bucket
from snn2 import expand, GATES
from snn2.spec import _operand_rate_map


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_gate(preset: str, gate: str, n_seeds: int = 12, length: int | None = None):
    """Train `n_seeds` independent learners on one gate. Returns (mean_curve [T],
    final_acc_per_seed [n_seeds], final_weights_per_seed [n_seeds, n_in])."""
    spec = {"preset": preset, "target_map": GATES[gate]}
    if length is not None:
        spec["len_episode"] = length
    curves, weights = [], []
    for sd in range(n_seeds):
        m = run_bucket([expand(spec)], seed=sd, log_trials=True)[0]
        curves.append(m["trial_correct"])
        weights.append(m["weights"][:expand(spec)["n_inputs"], 0])
    curves = np.array(curves)
    final = curves[:, -200:].mean(1)
    return curves.mean(0), final, np.array(weights)


def curriculum(n_seeds: int = 12):
    """Run the full four-stage curriculum. Returns a dict keyed by stage label."""
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


# --------------------------------------------------------------------------- #
# Static analysis: what does the dendritic neuron do as a function of weight?
# --------------------------------------------------------------------------- #
def dendritic_firing_map(wscales, gain=1.0, lo=3.0, hi=6.0, P=30, n_in=40,
                         rate=0.5, trials=400, seed=0):
    """Per-combo dendritic firing rate vs synaptic scale. Shows the XOR window:
    a range of weights where one input fires and two inputs are suppressed."""
    rng = np.random.default_rng(seed)
    rm = _operand_rate_map(n_in, rate)
    out = {lbl: [] for lbl in ("00", "01", "10", "11")}
    for w in wscales:
        W = np.full((n_in, 1), w)
        for combo, lbl in zip(range(4), ("00", "01", "10", "11")):
            spk = (rng.random((trials * P, n_in)) < rm[combo]).astype(float)
            I = (spk @ W)[:, 0] * gain
            out[lbl].append(float(((I >= lo) & (I <= hi)).mean()))
    return {k: np.array(v) for k, v in out.items()}


def gate_truth(preset, gate, seed=0):
    """Trained network's action on each of the 4 combos (the learned truth table)."""
    spec = expand({"preset": preset, "target_map": GATES[gate]})
    m = run_bucket([spec], seed=seed, log_trials=True)[0]
    W = m["weights"][:spec["n_inputs"], 0]
    rm = _operand_rate_map(spec["n_inputs"], spec.get("operand_rate", 0.5))
    gain = spec["input_gain"]; P = spec["processing_time"]
    rng = np.random.default_rng(seed + 777)
    acts = []
    for combo in range(4):
        rates = rm[combo]; fires = []
        for _ in range(120):
            spk = (rng.random((P, spec["n_inputs"])) < rates).astype(float)
            I = (spk @ W) * gain
            if spec["neuron"] == "dendritic":
                fire = ((I >= spec["dcap_lo"]) & (I <= spec["dcap_hi"])).mean()
            else:                                   # lif: accumulate, threshold
                v = 0.0; n = 0
                for t in range(P):
                    v = v * (1 - spec["potential_decay"]) + I[t]
                    if v >= spec["firing_threshold"]:
                        n += 1; v = spec["resting_mv"]
                fire = n / P
            fires.append(fire > spec["action_threshold"])
        acts.append(int(np.mean(fires) > 0.5))
    return acts


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate(n_seeds: int = 12, verbose: bool = True):
    res = curriculum(n_seeds)
    log = print if verbose else (lambda *a, **k: None)

    or_final  = res["OR  (monotonic)"]["final"].mean()
    and_final = res["AND (monotonic)"]["final"].mean()
    xm_final  = res["XOR (monotonic)"]["final"].mean()
    xd_curve  = res["XOR (dendritic)"]["curve"]
    xd_final  = res["XOR (dendritic)"]["final"]
    xd_start  = xd_curve[:20].mean()
    xd_conv   = (xd_final > 0.9).mean()

    log("Curriculum results (mean final accuracy over %d seeds):" % n_seeds)
    for k in res:
        log("  %-16s  start=%.2f  final=%.2f"
            % (k, res[k]["curve"][:20].mean(), res[k]["final"].mean()))

    checks = []
    # 1. starts not knowing: the dendritic learner is below mastery at the start
    checks.append(("dendritic XOR starts not knowing (start < 0.92)",
                   xd_start < 0.92))
    # 2. linearly-separable OR is learned
    checks.append(("monotonic OR is learned (final > 0.9)", or_final > 0.9))
    # 3. AND hits the point-neuron ceiling (learned-ish but not solved)
    checks.append(("monotonic AND plateaus below mastery (0.6 < final < 0.85)",
                   0.6 < and_final < 0.85))
    # 4. monotonic XOR fails -- the linear-separability wall
    checks.append(("monotonic XOR fails at the wall (final < 0.8)",
                   xm_final < 0.8))
    # 5. dendritic XOR is solved, and beats the monotonic neuron decisively
    checks.append(("dendritic XOR is solved (final > 0.9)", xd_final.mean() > 0.9))
    checks.append(("dendritic XOR converges for most seeds (>=80%%)",
                   xd_conv >= 0.8))
    checks.append(("dendrite beats point neuron on XOR by >0.3",
                   xd_final.mean() - xm_final > 0.3))
    # 6. learning happened: dendritic XOR end > start
    checks.append(("dendritic XOR improves with training (end > start + 0.05)",
                   xd_curve[-200:].mean() > xd_start + 0.05))
    # 7. the trained dendritic neuron implements the XOR truth table exactly
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


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def make_figure(path="logic_gates.png", n_seeds: int = 12):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = curriculum(n_seeds)
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1.0], hspace=0.42, wspace=0.32)

    colors = {"OR  (monotonic)": "#2563eb", "AND (monotonic)": "#0891b2",
              "XOR (monotonic)": "#dc2626", "XOR (dendritic)": "#16a34a"}

    # (A) learning curves -- the curriculum
    ax = fig.add_subplot(gs[0, :2])
    for k, d in res.items():
        c = d["curve"]
        sm = np.convolve(c, np.ones(40) / 40, mode="valid")
        ax.plot(sm, color=colors[k], lw=2, label="%s  →%.2f" % (k, d["final"].mean()))
    ax.axhline(0.5, color="#9ca3af", lw=0.8, ls=":")
    ax.axhline(1.0, color="#d1d5db", lw=0.8, ls="--")
    ax.text(5, 0.51, "chance", color="#6b7280", fontsize=8)
    ax.set_title("R-STDP logic-gate curriculum: point neuron degrades, "
                 "dendrite rescues XOR", fontsize=12)
    ax.set_xlabel("trial"); ax.set_ylabel("accuracy (40-trial avg)")
    ax.set_ylim(0.2, 1.05); ax.legend(loc="lower right", fontsize=9, framealpha=0.95)

    # (B) truth tables -- learned vs target
    ax = fig.add_subplot(gs[0, 2]); ax.axis("off")
    combos = ["00", "01", "10", "11"]
    rows = [("OR",  "logic_monotonic", GATES["OR"]),
            ("AND", "logic_monotonic", GATES["AND"]),
            ("XOR", "logic_monotonic", GATES["XOR"]),
            ("XOR†", "logic_dendritic", GATES["XOR"])]
    ax.set_title("Learned truth tables\n(green = matches target)", fontsize=11)
    ax.text(0.30, 0.92, "  ".join(combos), family="monospace", fontsize=11,
            transform=ax.transAxes)
    for r, (name, preset, target) in enumerate(rows):
        y = 0.78 - r * 0.18
        learned = gate_truth(preset, name.replace("†", ""), seed=0) \
            if name != "XOR†" else gate_truth("logic_dendritic", "XOR", seed=0)
        ax.text(0.0, y, "%-5s" % name, family="monospace", fontsize=11,
                transform=ax.transAxes)
        for i, (lv, tv) in enumerate(zip(learned, target)):
            col = "#16a34a" if lv == tv else "#dc2626"
            ax.text(0.30 + i * 0.165, y, str(lv), family="monospace", fontsize=12,
                    color=col, fontweight="bold", transform=ax.transAxes)
    ax.text(0.0, 0.06, "† dendritic (dCaAP) neuron", fontsize=8.5,
            color="#16a34a", transform=ax.transAxes)

    # (C) the dendritic XOR window: firing vs synaptic scale
    ax = fig.add_subplot(gs[1, :2])
    ws = np.linspace(0.05, 1.2, 40)
    fmap = dendritic_firing_map(ws, gain=1.0, lo=3.0, hi=6.0)
    ax.plot(ws, fmap["00"], color="#9ca3af", lw=2, label="00 (no input)")
    ax.plot(ws, (fmap["01"] + fmap["10"]) / 2, color="#16a34a", lw=2.5,
            label="one input (01/10)")
    ax.plot(ws, fmap["11"], color="#b91c1c", lw=2.5, label="two inputs (11)")
    # XOR window: one-input fires, two-input suppressed
    one = (fmap["01"] + fmap["10"]) / 2
    win = ws[(one > 0.15) & (fmap["11"] < 0.10)]
    if len(win):
        ax.axvspan(win.min(), win.max(), color="#16a34a", alpha=0.10)
        ax.text((win.min() + win.max()) / 2, 0.9, "XOR window", ha="center",
                color="#15803d", fontsize=10, fontweight="bold")
    ax.set_title("Why the dendrite computes XOR: band-pass firing vs synaptic scale\n"
                 "(one input lands in the dCaAP band; two inputs overshoot it)",
                 fontsize=11)
    ax.set_xlabel("synaptic weight scale"); ax.set_ylabel("dendritic firing rate")
    ax.legend(loc="upper right", fontsize=9); ax.set_ylim(0, 1.0)

    # (D) monotonic vs dendritic on XOR -- the contrast bar
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

    fig.suptitle("snn2 — learning logic gates with reward-modulated STDP, "
                 "and dendritic XOR (Gidon et al. 2020)", fontsize=13, y=0.99)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    print("Validating logic-gate curriculum:\n")
    validate()


if __name__ == "__main__":
    main()
