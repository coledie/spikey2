"""
The Izhikevich random-state task -- the INSTANTANEOUS reproduction.

Closed-form twin of `examples/izh` (and the repo's `izhikevich.py`). The original
advances a true a,b,c,d Izhikevich neuron through `processing_time` Euler steps
per trial and reward-modulated STDP over a window; here a trial is algebra:

    pre-rate -> (mean, var) of drive -> E[ Izhikevich f-I(I) ]  (noise-aware) ->
    realised rate -> reward-modulated rate-Hebbian weight update.

The neuron's transfer function is the type-I sqrt f-I curve of the regular-spiking
Izhikevich model, CALIBRATED once against the real neuron (`transfer.calibrate_izhi`)
and then evaluated with a noise-aware Gauss-Hermite expectation so the sparse
near-rheobase firing -- the regime this task lives in -- is captured without
stepping. The reward (`fire_states`) is delivered on a fixed subset of states, so
reward-modulated potentiation should make the neuron fire MORE for reward states
than for the rest: that learned discrimination is what we validate.

Run: python -m examples2.izhikevich
"""
from __future__ import annotations
import time
import numpy as np

from .engine import run_instant
from .spec import expand
from . import transfer as T


def _izhi_fire_fn(cal):
    return lambda I: T.izhi_rate(I, cal[0], cal[1])


def state_firing(p, W, cal):
    """Closed-form expected firing rate of each output for every state, given
    learned weights W [n_in, n_out]. Returns mean firing per state [n_states]."""
    rm = p["state_rate_map"]; gain = p["input_gain"]
    fire_fn = _izhi_fire_fn(cal)
    rates = []
    for s in range(p["n_states"]):
        mean, var = T.input_moments(rm[s], W, gain)
        rates.append(float(T.expected_rate(fire_fn, mean, var).mean()))
    return np.array(rates)


def discrimination(p, m):
    """Mean firing on reward states minus mean firing on non-reward states,
    averaged over seeds. Positive => the agent learned the reward contingency."""
    rstates = set(p["reward_fire_states"])
    W = m["weights"]; cal = m["calibration"]
    rw, nrw = [], []
    for b in range(W.shape[0]):
        f = state_firing(p, W[b], cal)
        for s in range(p["n_states"]):
            (rw if s in rstates else nrw).append(f[s])
    return float(np.mean(rw)), float(np.mean(nrw))


def fI_error(p, currents=None, steps=400):
    """Max |analytic f-I  -  simulated f-I| over a current sweep -- the direct
    check that the closed-form transfer function reproduces the stepped neuron."""
    cal = T.calibrate_izhi(p)
    if currents is None:
        currents = np.linspace(0.0, 20.0, 41)
    a, b, c, d = p["izhi_a"], p["izhi_b"], p["izhi_c"], p["izhi_d"]
    sim = []
    for I in currents:
        v = np.array([c], float); u = np.array([b * c], float); n = 0
        for _ in range(steps):
            v, u, fired = T._izhi_step(v, u, I, a, b, c, d); n += int(fired[0])
        sim.append(n / steps)
    ana = T.izhi_rate(currents, cal[0], cal[1])
    return float(np.max(np.abs(ana - np.array(sim)))), cal


def validate(n_seeds: int = 12, verbose: bool = True):
    log = print if verbose else (lambda *a, **k: None)
    p = expand({"preset": "izhi_randstate"})

    err, cal = fI_error(p)
    m = run_instant(p, n_seeds=n_seeds, seed=0)
    rw, nrw = discrimination(p, m)

    # lr=0 control: no learning -> no discrimination
    p0 = expand({"preset": "izhi_randstate", "lr": 0.0})
    m0 = run_instant(p0, n_seeds=n_seeds, seed=0)
    rw0, nrw0 = discrimination(p0, m0)

    log("Izhikevich random-state task (instantaneous), %d seeds:" % n_seeds)
    log("  calibrated f-I: rheobase=%.2f slope=%.4f  (max|analytic-sim|=%.4f)"
        % (cal[0], cal[1], err))
    log("  learned firing  reward-states=%.4f  non-reward=%.4f  (gap=%.4f)"
        % (rw, nrw, rw - nrw))
    log("  lr=0 control    reward-states=%.4f  non-reward=%.4f  (gap=%.4f)"
        % (rw0, nrw0, rw0 - nrw0))
    log("  final_reward=%.3f  mean_out_rate=%.4f  weight_norm=%.1f"
        % (m["final_reward"], m["mean_out_rate"], m["weight_norm"]))

    checks = []
    checks.append(("closed-form f-I matches the stepped neuron (max err < 0.05)",
                   err < 0.05))
    checks.append(("learning fires more for reward states (gap > 2x non-reward)",
                   rw > 2.0 * max(nrw, 1e-9)))
    checks.append(("learned discrimination beats lr=0 control by >3x",
                   (rw - nrw) > 3.0 * abs(rw0 - nrw0) + 1e-6))
    checks.append(("weights grew (weight_norm > control)",
                   m["weight_norm"] > m0["weight_norm"]))

    log("\nValidation:")
    for name, ok in checks:
        log("  [%s] %s" % ("PASS" if ok else "FAIL", name))
    passed = all(ok for _, ok in checks)
    log("\n%s" % ("ALL CHECKS PASSED" if passed else "SOME CHECKS FAILED"))
    return {"checks_passed": passed, "fI_error": err, "calibration": cal,
            "gap": rw - nrw, "gap_control": rw0 - nrw0,
            "reward_fire": rw, "nonreward_fire": nrw}


def speed_demo(n_seeds: int = 12):
    """Show how cheap a trial is when the inner time loop is gone."""
    p = expand({"preset": "izhi_randstate"})
    t = time.time()
    run_instant(p, n_seeds=n_seeds, seed=0)
    dt = time.time() - t
    trials = n_seeds * p["len_episode"]
    print("%d learner-trials in %.3fs  (%.0f trials/s, processing_time=%d "
          "micro-steps each ELIMINATED)" % (trials, dt, trials / max(dt, 1e-9),
                                            p["processing_time"]))


def make_figure(path="izhikevich_instant.png", n_seeds: int = 12):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = expand({"preset": "izhi_randstate"})
    cal = T.calibrate_izhi(p)
    currents = np.linspace(0.0, 20.0, 41)
    a, b, c, d = p["izhi_a"], p["izhi_b"], p["izhi_c"], p["izhi_d"]
    sim = []
    for I in currents:
        v = np.array([c], float); u = np.array([b * c], float); n = 0
        for _ in range(400):
            v, u, fired = T._izhi_step(v, u, I, a, b, c, d); n += int(fired[0])
        sim.append(n / 400)
    ana = T.izhi_rate(currents, cal[0], cal[1])

    m = run_instant(p, n_seeds=n_seeds, seed=0)
    rstates = set(p["reward_fire_states"])
    fire = np.mean([state_firing(p, m["weights"][bb], cal)
                    for bb in range(n_seeds)], axis=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    ax1.plot(currents, sim, "o", color="#94a3b8", ms=5, label="stepped Izhikevich")
    ax1.plot(currents, ana, "-", color="#dc2626", lw=2,
             label="closed-form f-I (Laplace)")
    ax1.axvline(cal[0], color="#9ca3af", ls=":", lw=1)
    ax1.set_title("Closed-form transfer function reproduces the spiking f-I curve")
    ax1.set_xlabel("constant input current I"); ax1.set_ylabel("firing rate (/step)")
    ax1.legend(loc="upper left", fontsize=9)

    cols = ["#16a34a" if s in rstates else "#cbd5e1" for s in range(p["n_states"])]
    ax2.bar(range(p["n_states"]), fire, color=cols)
    ax2.set_title("Learned firing per state\n(green = reward states)")
    ax2.set_xlabel("state"); ax2.set_ylabel("expected firing rate (/step)")
    ax2.set_xticks(range(p["n_states"]))

    fig.suptitle("examples2 \u2014 instantaneous Izhikevich random-state task "
                 "(no processing-time loop)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    print("Validating instantaneous Izhikevich random-state task:\n")
    validate()
    print()
    speed_demo()


if __name__ == "__main__":
    main()
