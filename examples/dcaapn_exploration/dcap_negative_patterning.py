"""
dCaAP negative patterning -- and the family of configural ("k-of-n") logic gates a
single non-monotonic neuron can compute.

NEGATIVE PATTERNING is a classic animal-learning paradigm (Rescorla; Pearce's
configural theory): reinforce cue A alone and cue B alone, but NOT the AB compound.

    A -> reward     B -> reward     AB -> no reward     (nothing -> no reward)

The animal must respond to either single cue yet WITHHOLD to the compound. Elemental
/ linear associative models fail this: if A and B each excite the response, their
compound AB should excite it *more* (summation) -- the opposite of what's required.
Negative patterning is, formally, XOR over the two cues:

    none=0   A=1   B=1   AB=0          (= XOR(a, b))

So the dCaAP solves a named behavioral phenomenon. A monotonic (point-neuron) agent
shows SUMMATION -- it responds to AB at least as strongly as to A or B -- and cannot
learn negative patterning. A dCaAP agent, whose response is band-pass (suppressed by
the stronger compound drive), learns it.

MORE GATES: the dCaAP as a "k-of-n" detector. Because the dCaAP fires only when its
drive sits in a band, and drive grows with the number of active inputs, a single
dCaAP neuron is a tunable "exactly-k active" detector. For two inputs the weight
scale selects AND ("both"), XOR ("exactly one"), or OR ("at least one"); for three
inputs it can select "exactly one of three" or "exactly two of three" -- configural
counting gates that NO monotonic point neuron can compute, because for a monotonic
neuron more active inputs always means more drive.

Run: python -m snn2.dcap_negative_patterning
"""
from __future__ import annotations
import numpy as np

from .engine import run_bucket
from .spec import expand, GATES, _operand_rate_map
from .logic_gates import train_gate


# --------------------------------------------------------------------------- #
# Negative patterning = XOR, learned (dCaAP) vs failed (monotonic)
# --------------------------------------------------------------------------- #
def combo_response(preset, gate, seed=0, trials=160):
    """Mean firing rate of the trained neuron to each of the 4 input cases
    (none, B, A, AB). The graded response -- used to expose summation."""
    spec = expand({"preset": preset, "target_map": GATES[gate]})
    m = run_bucket([spec], seed=seed, log_trials=True)[0]
    W = m["weights"][:spec["n_inputs"], 0]
    rm = _operand_rate_map(spec["n_inputs"], spec.get("operand_rate", 0.5))
    gain, Pn = spec["input_gain"], spec["processing_time"]
    rng = np.random.default_rng(seed + 31)
    out = []
    for combo in range(4):
        rates = rm[combo]; r = []
        for _ in range(trials):
            spk = (rng.random((Pn, spec["n_inputs"])) < rates).astype(float)
            I = (spk @ W) * gain
            if spec["neuron"] == "dendritic":
                r.append(((I >= spec["dcap_lo"]) & (I <= spec["dcap_hi"])).mean())
            else:
                v = 0.0; n = 0
                for t in range(Pn):
                    v = v * (1 - spec["potential_decay"]) + I[t]
                    if v >= spec["firing_threshold"]:
                        n += 1; v = spec["resting_mv"]
                r.append(n / Pn)
        out.append(np.mean(r))
    return np.array(out)        # [none, B, A, AB]


def negative_patterning(n_seeds=10):
    """Train dCaAP and monotonic agents on negative patterning (= XOR). For the
    summation test we also read out the response of an *elemental* learner -- a
    monotonic neuron that has learned the single-cue associations (trained on OR,
    i.e. respond to A and to B). Such a learner necessarily summates on AB."""
    cur_d, _, _ = train_gate("logic_dendritic", "XOR", n_seeds)
    cur_m, _, _ = train_gate("logic_monotonic", "XOR", n_seeds)
    resp_d = combo_response("logic_dendritic", "XOR")    # dCaAP on NP: AB suppressed
    resp_m = combo_response("logic_monotonic", "OR")     # elemental: AB summates
    return dict(curve_dcap=cur_d, curve_mono=cur_m, resp_dcap=resp_d, resp_mono=resp_m)


# --------------------------------------------------------------------------- #
# The k-of-n detector family (static capability map)
# --------------------------------------------------------------------------- #
def kofn_firing(n_inputs_total, n_operands, wscales, lo, hi, gain=1.0,
                rate=0.5, P=30, trials=300, seed=0):
    """For an n-operand dCaAP neuron, firing rate as a function of (number of
    active operands k) and synaptic weight scale. Reveals which weight makes the
    neuron an 'exactly-k' detector."""
    rng = np.random.default_rng(seed)
    grp = n_inputs_total // n_operands
    # firing[k][wi] = P(dCaAP fires) when exactly k operands active, at weight w
    firing = np.zeros((n_operands + 1, len(wscales)))
    for wi, w in enumerate(wscales):
        for k in range(n_operands + 1):
            active = np.zeros(n_inputs_total)
            active[:k * grp] = rate                    # first k groups active
            spk = (rng.random((trials * P, n_inputs_total)) < active).astype(float)
            I = spk.sum(1) * w * gain
            firing[k, wi] = ((I >= lo) & (I <= hi)).mean()
    return firing


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate(n_seeds=10, verbose=True):
    log = print if verbose else (lambda *a, **k: None)
    np_ = negative_patterning(n_seeds)
    rd, rm = np_["resp_dcap"], np_["resp_mono"]      # [none, B, A, AB]
    fin_d = np_["curve_dcap"][-200:].mean()
    fin_m = np_["curve_mono"][-200:].mean()

    # summation index: AB response relative to the mean single-cue response
    single_d = (rd[1] + rd[2]) / 2; single_m = (rm[1] + rm[2]) / 2
    summ_d = rd[3] / max(single_d, 1e-9)
    summ_m = rm[3] / max(single_m, 1e-9)

    # k-of-n detectors exist: exactly-1-of-3 and exactly-2-of-3 windows
    ws = np.linspace(0.1, 1.0, 70)
    f3 = kofn_firing(60, 3, ws, lo=4.0, hi=5.5, gain=1.0)
    e1 = np.any((f3[1] > 0.3) & (f3[0] < 0.12) & (f3[2] < 0.12) & (f3[3] < 0.12))
    e2 = np.any((f3[2] > 0.3) & (f3[1] < 0.12) & (f3[3] < 0.12))

    log("dCaAP negative patterning and the k-of-n gate family\n")
    log("Negative patterning (A+, B+, AB-):")
    log("  dCaAP    final accuracy = %.2f   response[none,B,A,AB]=%s"
        % (fin_d, np.round(rd, 2)))
    log("  monotonic final accuracy = %.2f   response[none,B,A,AB]=%s"
        % (fin_m, np.round(rm, 2)))
    log("  summation (AB / single-cue):  dCaAP %.2f   monotonic %.2f" % (summ_d, summ_m))
    log("\nk-of-n detectors (single dCaAP neuron):")
    log("  exactly-1-of-3 window exists: %s" % e1)
    log("  exactly-2-of-3 window exists: %s" % e2)

    checks = [
        ("dCaAP learns negative patterning (acc > 0.9)", fin_d > 0.9),
        ("monotonic FAILS negative patterning (acc < 0.8)", fin_m < 0.8),
        ("dCaAP suppresses the compound: AB < single cue (summ < 0.6)",
         summ_d < 0.6),
        ("monotonic shows summation: AB >= single cue (summ >= 0.9)",
         summ_m >= 0.9),
        ("a single dCaAP can be an exactly-1-of-3 detector", bool(e1)),
        ("a single dCaAP can be an exactly-2-of-3 detector", bool(e2)),
    ]
    log("\nValidation:")
    for name, ok in checks:
        log("  [%s] %s" % ("PASS" if ok else "FAIL", name))
    passed = all(ok for _, ok in checks)
    log("\n%s" % ("ALL CHECKS PASSED" if passed else "SOME CHECKS FAILED"))
    return {"checks_passed": passed, "np": np_, "summ_dcap": summ_d,
            "summ_mono": summ_m, "fin_dcap": fin_d, "fin_mono": fin_m,
            "f3": f3, "ws": ws, "e1": bool(e1), "e2": bool(e2)}


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def make_figures(out_prefix="dcap_np"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = validate(verbose=False)
    np_ = res["np"]; rd, rm = np_["resp_dcap"], np_["resp_mono"]

    # ---- Figure 1: negative patterning -------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    sm = lambda a: np.convolve(a, np.ones(40) / 40, mode="valid")
    ax = axes[0]
    ax.plot(sm(np_["curve_dcap"]), color="#16a34a", lw=2,
            label="dCaAP  →%.2f" % res["fin_dcap"])
    ax.plot(sm(np_["curve_mono"]), color="#dc2626", lw=2,
            label="monotonic  →%.2f" % res["fin_mono"])
    ax.axhline(0.5, color="#9ca3af", ls=":", lw=0.8)
    ax.set_title("Learning negative patterning\n(A+, B+, AB−)", fontsize=11)
    ax.set_xlabel("trial"); ax.set_ylabel("accuracy"); ax.set_ylim(0.3, 1.05)
    ax.legend(fontsize=9, loc="lower right")

    labels = ["none", "B", "A", "AB"]
    for ax, resp, col, name in [(axes[1], rm, "#dc2626", "monotonic (elemental)"),
                                (axes[2], rd, "#16a34a", "dCaAP (configural)")]:
        bars = ax.bar(labels, resp, color=["#d1d5db", col, col, "#111827"])
        ax.set_title("%s\nresponse to each cue" % name, fontsize=11)
        ax.set_ylabel("response (firing rate)")
        ax.set_ylim(0, max(rm.max(), rd.max()) * 1.25)
        single = (resp[1] + resp[2]) / 2
        ax.axhline(single, color="#6b7280", ls="--", lw=1)
        note = "AB summates ↑" if resp[3] >= single else "AB suppressed ↓"
        ax.text(3, resp[3] + 0.01, note, ha="center", fontsize=8.5,
                color=("#b91c1c" if resp[3] >= single else "#15803d"))
    fig.suptitle("snn2 — dCaAP solves negative patterning, where an elemental "
                 "(monotonic) learner summates", fontsize=12.5, y=1.02)
    fig.tight_layout()
    f1 = out_prefix + "_negative_patterning.png"
    fig.savefig(f1, dpi=130, bbox_inches="tight"); plt.close(fig)

    # ---- Figure 2: the k-of-n gate family ----------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    ws = res["ws"]; f3 = res["f3"]
    # 2-input map: OR / XOR / AND regions
    ws2 = np.linspace(0.1, 1.6, 60)
    f2 = kofn_firing(40, 2, ws2, lo=3.0, hi=6.0, gain=1.0)
    ax = axes[0]
    cols2 = ["#9ca3af", "#16a34a", "#b91c1c"]
    for k, lab in zip(range(3), ["0 active", "1 active", "2 active"]):
        ax.plot(ws2, f2[k], color=cols2[k], lw=2, label=lab)
    ax.set_title("Two inputs: one dCaAP, three gates\n(weight selects AND / XOR / OR)",
                 fontsize=11)
    ax.set_xlabel("synaptic weight scale"); ax.set_ylabel("dCaAP firing rate")
    ax.legend(fontsize=8.5)
    # annotate regions
    for lab, cond in [("AND", (f2[2] > 0.3) & (f2[1] < 0.1)),
                      ("XOR", (f2[1] > 0.3) & (f2[2] < 0.1))]:
        idx = np.where(cond)[0]
        if len(idx):
            ax.axvspan(ws2[idx[0]], ws2[idx[-1]], color="#fde68a", alpha=0.25)
            ax.text(ws2[idx].mean(), 0.9, lab, ha="center", fontsize=9, fontweight="bold")

    # 3-input map: exactly-1 and exactly-2 detectors
    ax = axes[1]
    cols3 = ["#9ca3af", "#16a34a", "#7c3aed", "#b91c1c"]
    for k in range(4):
        ax.plot(ws, f3[k], color=cols3[k], lw=2, label="%d active" % k)
    ax.set_ylim(0, 0.85)
    ax.set_title("Three inputs: exactly-k detectors\n(impossible for a point neuron)",
                 fontsize=11)
    ax.set_xlabel("synaptic weight scale"); ax.set_ylabel("dCaAP firing rate")
    ax.legend(fontsize=8.5)
    for lab, k_on, others in [("exactly 1", 1, [0, 2, 3]), ("exactly 2", 2, [0, 1, 3])]:
        cond = (f3[k_on] > 0.3)
        for o in others:
            cond = cond & (f3[o] < 0.12)
        idx = np.where(cond)[0]
        if len(idx):
            ax.axvspan(ws[idx[0]], ws[idx[-1]], color="#fde68a", alpha=0.25)
            ax.text(ws[idx].mean(), 0.78, lab, ha="center", fontsize=8.5, fontweight="bold")

    # schematic: why a monotonic neuron can't (drive is monotonic in #active)
    ax = axes[2]
    kk = np.arange(4)
    ax.plot(kk, kk, "o-", color="#6b7280", lw=2, label="drive (monotonic)")
    ax.axhspan(0.7, 1.4, color="#16a34a", alpha=0.15)
    ax.text(3.0, 1.05, "dCaAP band", color="#15803d", fontsize=9, ha="right")
    ax.set_title("Why it needs the band\n(drive grows with #active; band picks one)",
                 fontsize=11)
    ax.set_xlabel("number of active inputs"); ax.set_ylabel("dendritic drive (a.u.)")
    ax.set_xticks(kk); ax.legend(fontsize=8.5, loc="upper left")

    fig.suptitle("snn2 — a single dCaAP neuron is a tunable 'k-of-n' detector "
                 "(more gates than a point neuron)", fontsize=12.5, y=1.02)
    fig.tight_layout()
    f2name = out_prefix + "_kofn_gates.png"
    fig.savefig(f2name, dpi=130, bbox_inches="tight"); plt.close(fig)
    return f1, f2name


def main():
    print("Validating dCaAP negative patterning + k-of-n gates:\n")
    validate()


if __name__ == "__main__":
    main()
