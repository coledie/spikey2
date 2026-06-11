"""
Farries & Fairhall (2007), "Reinforcement learning with modulated spike
timing-dependent synaptic plasticity" (J Neurophysiol 98:3648-3665) -- the
central theoretical result, reproduced and validated against an exact oracle.

THE CLAIM. A reward signal that *modulates* STDP turns plain Hebbian plasticity
into reinforcement learning: the trial-averaged weight change climbs the gradient
of EXPECTED REWARD. Concretely, with an escape-noise (stochastic) spiking neuron,
the synaptic eligibility trace produced by STDP approximates the policy-gradient
"score function" d/dw log P(output), so

      <Delta w_i>  =  Cov(R, e_i)  +  R_bar * <e_i>            (decomposition)
                   ~=  dE[R]/dw_i      when the reward is CENTERED (R_bar removed)

The first term is the gradient (the useful, reward-sensitive part); the second is
an unsupervised STDP bias that does NOT point along the gradient. Subtracting a
reward baseline R_bar removes the bias term. This is the result -- and the now
much-cited caveat -- that the rule only does gradient ascent when reward is
referenced to its expected value.

WHY THIS IS THE RIGHT THING TO REPRODUCE. The gradient of expected reward can be
measured directly by finite differences (perturb each weight, re-estimate E[R]).
That gives an *independent oracle* for the direction the learning rule should
move -- exactly the golden-test structure the snn2 validation methodology is built
on. We check that the centered reward-modulated update aligns with that oracle
gradient, that the uncentered update does not, and that centered learning actually
climbs E[R] while badly-baselined learning stalls or goes backward.

THE MODEL (a minimal escape-noise spiking neuron, faithful to the paper's class).
  - N input lines; on each trial input i emits a Bernoulli spike train x_i(t).
  - Membrane:    u(t)   = sum_i w_i x_i(t)
  - Escape noise:rho(t) = sigmoid(gain * (u(t) - theta))      (stochastic firing)
  - Output:      y(t) ~ Bernoulli(rho(t)),  count n = sum_t y(t)
  - Reward:      R(n)   = exp(-(n - n_target)^2 / (2 sigma^2))  in [0, 1]
The neuron must learn weights that make its output count match a target -- a
target-rate ("biofeedback") task of the kind these rules were built for.

TWO ELIGIBILITY TRACES.
  - score  e_i = sum_t (y(t) - rho(t)) x_i(t)   -- exact d/dw_i log P(y|x), mean 0
  - stdp   e_i = sum_t  y(t)          x_i(t)   -- realistic Hebbian coincidence,
                                                  biased: <e> = sum_t rho(t) x_i(t)
The score trace is what an ideal policy-gradient learner would use; the stdp trace
is what a synapse can actually measure. The paper's point is that the stdp trace
approximates the score trace -- *once the reward baseline removes its bias*.

Run: python -m snn2.farries
"""
from __future__ import annotations
import numpy as np


# --------------------------------------------------------------------------- #
# Model defaults
# --------------------------------------------------------------------------- #
DEFAULTS = dict(
    n_inputs=6, n_steps=25, gain=2.0, theta=1.2,
    input_rate=0.4, n_target=8, reward_sigma=3.0,
)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _input_rates(p, rng_seed=0):
    """Fixed per-trial input firing probabilities (the stimulus pattern)."""
    rng = np.random.default_rng(rng_seed)
    return rng.uniform(0.25, 0.55, p["n_inputs"])


# --------------------------------------------------------------------------- #
# One trial: spikes, eligibility traces, reward (vectorized over a batch)
# --------------------------------------------------------------------------- #
def run_trials(w, p, rates, n_trials, rng):
    """Simulate `n_trials` independent trials of the escape-noise neuron.

    Returns dict with reward R [K], and eligibility traces e_stdp, e_score
    [K, N]. Vectorized over the trial axis K."""
    N, T = p["n_inputs"], p["n_steps"]
    K = n_trials
    x = (rng.random((K, T, N)) < rates[None, None, :]).astype(np.float64)  # inputs
    u = x @ w                                            # [K, T] membrane
    rho = _sigmoid(p["gain"] * (u - p["theta"]))         # [K, T] fire prob
    y = (rng.random((K, T)) < rho).astype(np.float64)    # [K, T] output spikes
    n = y.sum(1)                                         # [K] output count
    R = np.exp(-(n - p["n_target"]) ** 2 / (2 * p["reward_sigma"] ** 2))
    # eligibility traces: sum over time of (post-term) * pre
    e_stdp = np.einsum("kt,ktn->kn", y, x)               # Hebbian coincidence
    e_score = np.einsum("kt,ktn->kn", (y - rho), x)      # exact score function
    return {"R": R, "n": n, "e_stdp": e_stdp, "e_score": e_score, "rho": rho}


def expected_reward(w, p, rates, n_trials, rng):
    return run_trials(w, p, rates, n_trials, rng)["R"].mean()


# --------------------------------------------------------------------------- #
# The oracle: finite-difference gradient of expected reward
# --------------------------------------------------------------------------- #
def fd_gradient(w, p, rates, n_trials=20000, eps=0.02, seed=0):
    """dE[R]/dw_i by central finite differences. The independent oracle for the
    direction the learning rule should move. Common random numbers (same seed per
    +/- pair) sharply reduce variance."""
    N = p["n_inputs"]
    g = np.zeros(N)
    for i in range(N):
        wp = w.copy(); wp[i] += eps
        wm = w.copy(); wm[i] -= eps
        Rp = expected_reward(wp, p, rates, n_trials, np.random.default_rng(1000 + i))
        Rm = expected_reward(wm, p, rates, n_trials, np.random.default_rng(1000 + i))
        g[i] = (Rp - Rm) / (2 * eps)
    return g


# --------------------------------------------------------------------------- #
# The learning rule's expected update, by eligibility and baseline
# --------------------------------------------------------------------------- #
def mean_update(w, p, rates, elig="e_stdp", baseline="centered",
                n_trials=20000, seed=7):
    """Trial-averaged reward-modulated update <(R - b) * e_i> for the chosen
    eligibility trace and baseline. baseline: 'centered' uses b=<R>; 'none' uses
    b=0; a float uses that constant offset."""
    out = run_trials(w, p, rates, n_trials, np.random.default_rng(seed))
    R, e = out["R"], out[elig]
    if baseline == "centered":
        b = R.mean()
    elif baseline == "none":
        b = 0.0
    else:
        b = float(baseline)
    return ((R - b)[:, None] * e).mean(0)


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


# --------------------------------------------------------------------------- #
# Learning: online reward-modulated STDP with a running reward baseline
# --------------------------------------------------------------------------- #
def learn(p, rates, elig="e_stdp", baseline="centered", offset=0.0,
          lr=0.04, n_trials=4000, batch=20, w0=None, seed=0):
    """Train the neuron. baseline='centered' subtracts a running estimate of E[R]
    (the gradient-tracking rule); baseline='offset' subtracts that estimate PLUS a
    constant `offset` (a deliberately mis-set baseline, to show degradation).
    Returns the E[R] trajectory (per batch) and final weights."""
    rng = np.random.default_rng(seed)
    N = p["n_inputs"]
    w = (rng.uniform(0.1, 0.5, N) if w0 is None else w0.copy())
    Rbar = 0.5
    traj = []
    for _ in range(n_trials // batch):
        out = run_trials(w, p, rates, batch, rng)
        R, e = out["R"], out[elig]
        Rbar = 0.95 * Rbar + 0.05 * R.mean()
        b = Rbar + (offset if baseline == "offset" else 0.0)
        if baseline == "none":
            b = 0.0
        w = w + lr * ((R - b)[:, None] * e).mean(0)
        np.clip(w, 0.0, 3.0, out=w)
        traj.append(R.mean())
    return np.array(traj), w


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate(verbose=True):
    log = print if verbose else (lambda *a, **k: None)
    p = dict(DEFAULTS)
    rates = _input_rates(p)
    rng = np.random.default_rng(0)
    # Test alignment where the baseline matters: an OVER-driven neuron (output
    # above target), so the reward gradient wants to LOWER weights while the
    # unsupervised Hebbian bias <e> still points up -- the uncentered update then
    # moves the wrong way. (At an under-driven weight both point up and centering
    # would look unnecessary.)
    w = rng.uniform(0.45, 0.75, p["n_inputs"])

    grad = fd_gradient(w, p, rates)                              # oracle
    u_score_c = mean_update(w, p, rates, "e_score", "centered")  # ideal PG
    u_stdp_c  = mean_update(w, p, rates, "e_stdp",  "centered")  # R-STDP centered
    u_stdp_0  = mean_update(w, p, rates, "e_stdp",  "none")      # R-STDP uncentered

    cos_score = cosine(u_score_c, grad)
    cos_stdp  = cosine(u_stdp_c,  grad)
    cos_unc   = cosine(u_stdp_0,  grad)

    # Learning: centered vs deliberately offset baseline (offset ~ reward SD)
    tr_c, _ = learn(p, rates, baseline="centered", seed=1)
    tr_o, _ = learn(p, rates, baseline="offset", offset=-0.6, seed=1)
    gain_c = tr_c[-10:].mean() - tr_c[:10].mean()
    gain_o = tr_o[-10:].mean() - tr_o[:10].mean()

    log("Farries & Fairhall (2007) -- R-STDP is gradient ascent on E[reward]\n")
    log("Alignment with the finite-difference reward gradient (cosine):")
    log("  exact score-function eligibility, centered : %.3f" % cos_score)
    log("  realistic STDP eligibility,       centered : %.3f" % cos_stdp)
    log("  realistic STDP eligibility,       UNcentered: %.3f" % cos_unc)
    log("\nLearning (change in E[R], start -> end):")
    log("  centered reward baseline : %+.3f" % gain_c)
    log("  offset  reward baseline  : %+.3f" % gain_o)

    checks = [
        ("score-function update aligns with gradient (cos > 0.9)", cos_score > 0.9),
        ("centered R-STDP update aligns with gradient (cos > 0.8)", cos_stdp > 0.8),
        ("centering helps alignment (centered cos > uncentered + 0.2)",
         cos_stdp > cos_unc + 0.2),
        ("centered learning climbs E[R] (gain > 0.05)", gain_c > 0.05),
        ("offset-baseline learning does not climb (gain < centered - 0.04)",
         gain_o < gain_c - 0.04),
    ]
    log("\nValidation:")
    for name, ok in checks:
        log("  [%s] %s" % ("PASS" if ok else "FAIL", name))
    passed = all(ok for _, ok in checks)
    log("\n%s" % ("ALL CHECKS PASSED" if passed else "SOME CHECKS FAILED"))
    return {"checks_passed": passed, "cos_score": cos_score, "cos_stdp": cos_stdp,
            "cos_unc": cos_unc, "gain_centered": gain_c, "gain_offset": gain_o,
            "grad": grad, "u_stdp_c": u_stdp_c, "u_stdp_0": u_stdp_0}


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def make_figure(path="farries_fig.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = dict(DEFAULTS)
    rates = _input_rates(p)
    rng = np.random.default_rng(0)
    w = rng.uniform(0.45, 0.75, p["n_inputs"])   # over-driven: baseline matters
    grad = fd_gradient(w, p, rates)
    u_stdp_c = mean_update(w, p, rates, "e_stdp", "centered")
    u_stdp_0 = mean_update(w, p, rates, "e_stdp", "none")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))

    # (A) learning curves: centered vs offset baselines
    ax = axes[0]
    for off, col, lab in [(0.0, "#16a34a", "centered (b = E[R])"),
                          (-0.6, "#dc2626", "offset baseline (b = E[R] - 0.6)"),
                          (None, "#9ca3af", "no baseline (b = 0)")]:
        if off is None:
            tr, _ = learn(p, rates, baseline="none", seed=2)
        else:
            tr, _ = learn(p, rates, baseline="offset", offset=off, seed=2)
        sm = np.convolve(tr, np.ones(8) / 8, mode="valid")
        ax.plot(np.linspace(0, len(tr) * 20, len(sm)), sm, color=col, lw=2, label=lab)
    ax.set_title("Learning depends on the reward baseline", fontsize=11)
    ax.set_xlabel("trial"); ax.set_ylabel("expected reward  E[R]")
    ax.legend(fontsize=8.5, loc="lower right"); ax.set_ylim(0, 1.02)

    # (B) the policy-gradient result: update vs oracle gradient
    ax = axes[1]
    ax.axhline(0, color="#e5e7eb", lw=0.8); ax.axvline(0, color="#e5e7eb", lw=0.8)
    gs = grad / np.linalg.norm(grad)
    ax.scatter(gs, u_stdp_c / np.linalg.norm(u_stdp_c), color="#16a34a", s=70,
               zorder=3, label="centered R-STDP (cos %.2f)" % cosine(u_stdp_c, grad))
    ax.scatter(gs, u_stdp_0 / np.linalg.norm(u_stdp_0), color="#dc2626", s=70,
               marker="x", zorder=3,
               label="uncentered (cos %.2f)" % cosine(u_stdp_0, grad))
    lim = 0.8
    ax.plot([-lim, lim], [-lim, lim], color="#9ca3af", ls="--", lw=1, zorder=1)
    ax.set_title("R-STDP update vs true reward gradient\n(per-synapse, normalized)",
                 fontsize=11)
    ax.set_xlabel("oracle gradient  dE[R]/dw  (finite diff)")
    ax.set_ylabel("reward-modulated update  <(R-b) e>")
    ax.legend(fontsize=8.5, loc="upper left"); ax.set_aspect("equal")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)

    # (C) baseline offset sweep -- performance peaks at the true E[R]
    ax = axes[2]
    offsets = np.linspace(-0.8, 0.8, 13)
    finals = []
    for off in offsets:
        tr, _ = learn(p, rates, baseline="offset", offset=off, seed=3)
        finals.append(tr[-15:].mean())
    ax.plot(offsets, finals, "o-", color="#7c3aed", lw=2)
    ax.axvline(0, color="#16a34a", ls=":", lw=1.2)
    ax.text(0.02, ax.get_ylim()[0] + 0.03, "true baseline", color="#16a34a",
            fontsize=8.5, rotation=90, va="bottom")
    ax.set_title("Mis-setting the baseline destroys learning", fontsize=11)
    ax.set_xlabel("baseline offset  (b - E[R])")
    ax.set_ylabel("final  E[R]")

    fig.suptitle("snn2 — Farries & Fairhall (2007): reward-modulated STDP "
                 "performs gradient ascent on expected reward", fontsize=12.5, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    print("Validating Farries & Fairhall (2007) reproduction:\n")
    validate()


if __name__ == "__main__":
    main()
