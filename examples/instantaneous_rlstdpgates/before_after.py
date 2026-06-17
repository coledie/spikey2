"""
Before vs after training -- the reward task made visible.

The `izhi_randstate` experiment rewards the network for FIRING on a privileged
set of states (here {0, 3, 6, 9}) and for staying QUIET on the rest. Reward-
modulated Hebbian learning should therefore pull the output firing rate UP on
reward states and leave it low elsewhere -- turning a flat, indiscriminate net
into a state discriminator.

This script runs that experiment with the instantaneous engine (`examples2`),
captures the network at two moments,

    BEFORE : the random initial weights (what run_instant starts from),
    AFTER  : the weights once training has converged,

and renders a single figure so you can read the result at a glance:

  (1) Reward learning curve            -- reward climbs from chance to ceiling.
  (2) Per-state output firing rate      -- reward states rise above the quiet
      before vs after                      ones only AFTER training.
  (3) Discriminability                  -- gap between reward- and non-reward
                                           firing, before vs after.
  (4) Weight matrix before vs after     -- where the learned structure landed.

Everything is closed-form (no spiking loop on the learning path): the per-state
firing rates are read straight off the weights with the same transfer functions
the engine learns through, so the "before"/"after" panels are exact, not sampled.

Run: python examples2/before_after.py
"""
from __future__ import annotations
import numpy as np

from .spec import expand
from .engine import run_instant
from . import transfer as T


def state_firing(W, p, cal):
    """Mean output firing rate (per output neuron, averaged) for EACH state,
    read in closed form from weights `W` [n_in, n_out]. Returns [n_states]."""
    rm = p["state_rate_map"]                     # [n_states, n_in]
    gain = p["input_gain"]
    out = np.zeros(p["n_states"])
    for s in range(p["n_states"]):
        mean, var = T.input_moments(rm[s], W, gain)          # [n_out]
        f = T.expected_rate(lambda I: T.izhi_rate(I, cal[0], cal[1]), mean, var)
        out[s] = float(f.mean())
    return out


def before_after(preset="izhi_randstate", n_seeds=16, seed=0, length=400):
    """Train the reward task; return per-state firing + curves at BEFORE/AFTER."""
    p = expand({"preset": preset, "len_episode": length})
    cal = T.calibrate_izhi(p)

    # Reproduce run_instant's exact initial weights (same rng draw order) so the
    # "before" snapshot is the true starting point of the "after" run.
    n_in, n_out = p["n_inputs"], p["n_outputs"]
    rng = np.random.default_rng(seed)
    W0 = rng.uniform(0.0, 0.5, size=(n_seeds, n_in, n_out))

    m = run_instant(p, n_seeds=n_seeds, seed=seed, log_trials=True)
    Wf = m["weights"]                                        # [B, n_in, n_out]

    before = np.array([state_firing(W0[b], p, cal) for b in range(n_seeds)])
    after = np.array([state_firing(Wf[b], p, cal) for b in range(n_seeds)])

    reward_states = [int(s) for s in p["reward_fire_states"]]
    mask = np.zeros(p["n_states"], dtype=bool); mask[reward_states] = True

    return {
        "p": p, "cal": cal, "mask": mask,
        "before": before, "after": after,            # [B, n_states]
        "W0": W0.mean(0), "Wf": Wf.mean(0),           # mean weight matrix
        "reward_states": reward_states,
    }


def learning_curve(preset="izhi_randstate", n_seeds=16, seed=0,
                   lengths=(0, 25, 50, 100, 150, 200, 300, 400)):
    """Discriminability (reward-state firing minus non-reward firing) as a
    function of how many trials the network has trained for. length=0 is the
    untrained 'before'. This is the honest learning signal for this task: the
    raw per-trial reward is fixed by the random cue distribution, what actually
    improves is how selectively the net fires on the rewarded states."""
    p = expand({"preset": preset})
    cal = T.calibrate_izhi(p)
    n_in, n_out = p["n_inputs"], p["n_outputs"]
    mask = np.zeros(p["n_states"], dtype=bool)
    mask[[int(s) for s in p["reward_fire_states"]]] = True

    gaps = []
    for L in lengths:
        if L == 0:
            rng = np.random.default_rng(seed)
            W = rng.uniform(0.0, 0.5, size=(n_seeds, n_in, n_out))
        else:
            pl = expand({"preset": preset, "len_episode": int(L)})
            W = run_instant(pl, n_seeds=n_seeds, seed=seed)["weights"]
        fr = np.array([state_firing(W[b], p, cal) for b in range(n_seeds)])
        gaps.append(fr[:, mask].mean(1) - fr[:, ~mask].mean(1))
    return np.array(lengths), np.array(gaps)        # gaps: [n_lengths, n_seeds]


def make_figure(path="before_after.png", n_seeds=16, length=400):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = before_after(n_seeds=n_seeds, length=length)
    p, mask = d["p"], d["mask"]
    ns = p["n_states"]
    before, after = d["before"], d["after"]
    b_mu, a_mu = before.mean(0), after.mean(0)
    states = np.arange(ns)

    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], hspace=0.36, wspace=0.34)

    # (1) learning curve: discriminability vs training length ------------------
    ax = fig.add_subplot(gs[0, :2])
    lens, gaps = learning_curve(n_seeds=n_seeds)
    mu = gaps.mean(1); sd = gaps.std(1)
    ax.fill_between(lens, mu - sd, mu + sd, color="#86efac", alpha=0.45)
    ax.plot(lens, mu, color="#15803d", lw=2, marker="o",
            label="reward − non-reward firing (%d seeds)" % n_seeds)
    ax.axhline(0, color="#9ca3af", lw=0.8, ls=":")
    ax.scatter([lens[0]], [mu[0]], color="#6b7280", zorder=5, s=60)
    ax.annotate("before", (lens[0], mu[0]), textcoords="offset points",
                xytext=(8, 10), fontsize=9, color="#6b7280")
    ax.annotate("after", (lens[-1], mu[-1]), textcoords="offset points",
                xytext=(-30, -14), fontsize=9, color="#15803d")
    ax.set_title("Training: the net learns to fire selectively on reward states",
                 fontsize=12)
    ax.set_xlabel("trials trained")
    ax.set_ylabel("discriminability (firing-rate gap)")
    ax.legend(loc="lower right", fontsize=9); ax.margins(x=0.02)

    # (3) discriminability bar -------------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    gap_b = before[:, mask].mean(1) - before[:, ~mask].mean(1)
    gap_a = after[:, mask].mean(1) - after[:, ~mask].mean(1)
    ax.bar(["before", "after"], [gap_b.mean(), gap_a.mean()],
           yerr=[gap_b.std(), gap_a.std()], capsize=5,
           color=["#9ca3af", "#16a34a"])
    ax.axhline(0, color="#374151", lw=0.8)
    ax.set_title("Discriminability\n(reward \u2212 non-reward firing)", fontsize=11)
    ax.set_ylabel("firing-rate gap")

    # (2) per-state firing before vs after ------------------------------------
    ax = fig.add_subplot(gs[1, :2])
    w = 0.38
    colors_b = ["#d1d5db"] * ns
    colors_a = ["#16a34a" if mask[s] else "#dc2626" for s in range(ns)]
    ax.bar(states - w / 2, b_mu, w, yerr=before.std(0), capsize=2,
           color=colors_b, label="before")
    ax.bar(states + w / 2, a_mu, w, yerr=after.std(0), capsize=2,
           color=colors_a, label="after")
    for s in range(ns):
        if mask[s]:
            ax.axvspan(s - 0.5, s + 0.5, color="#dcfce7", alpha=0.5, zorder=0)
    ax.set_xticks(states)
    ax.set_title("Per-state output firing rate -- reward states (shaded) rise, "
                 "the rest stay quiet", fontsize=11)
    ax.set_xlabel("state  (shaded = rewarded: %s)" % list(d["reward_states"]))
    ax.set_ylabel("mean output firing rate")
    handles = [plt.Rectangle((0, 0), 1, 1, color="#d1d5db"),
               plt.Rectangle((0, 0), 1, 1, color="#16a34a"),
               plt.Rectangle((0, 0), 1, 1, color="#dc2626")]
    ax.legend(handles, ["before (untrained)", "after: reward state",
                        "after: non-reward state"], fontsize=9, loc="upper right")

    # (4) weight matrices before vs after -------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    # show mean input->output weight grouped by the input block that codes each
    # state (n_inputs / n_states inputs per state) -> [n_states, n_out] summary
    block = p["n_inputs"] // ns
    def block_view(W):
        return np.stack([W[s * block:(s + 1) * block].mean(0)
                         for s in range(ns)], axis=0)        # [n_states, n_out]
    bv = block_view(d["Wf"]) - block_view(d["W0"])
    im = ax.imshow(bv, aspect="auto", cmap="RdBu_r",
                   vmin=-np.abs(bv).max(), vmax=np.abs(bv).max())
    ax.set_title("Weight change (after \u2212 before)\nby input-state block",
                 fontsize=10.5)
    ax.set_xlabel("output neuron"); ax.set_ylabel("input state block")
    ax.set_yticks(states)
    for s in range(ns):
        if mask[s]:
            ax.add_patch(plt.Rectangle((-0.5, s - 0.5), bv.shape[1], 1,
                         fill=False, edgecolor="#16a34a", lw=2))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="\u0394 weight")

    fig.suptitle("izhi_randstate reward task: before vs after training "
                 "(reward = fire on states %s)" % list(d["reward_states"]),
                 fontsize=13, y=0.985)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    # console summary
    print("Before/after training (%d seeds, %d trials):" % (n_seeds, length))
    print("  reward-state firing : before=%.3f  after=%.3f"
          % (before[:, mask].mean(), after[:, mask].mean()))
    print("  non-reward firing   : before=%.3f  after=%.3f"
          % (before[:, ~mask].mean(), after[:, ~mask].mean()))
    print("  discriminability gap: before=%.3f  after=%.3f" % (gap_b.mean(), gap_a.mean()))
    print("  figure ->", path)
    return path


if __name__ == "__main__":
    make_figure("examples2/before_after.png")
