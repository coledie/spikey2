"""
Instrumental (operant) conditioning -- and the validation that batched
*grouping* is correct.

THE TASK (GO / NO-GO). Each trial presents one of two cues. One output neuron
must fire for the GO cue and stay silent for the NO-GO cue. Reward is +1 when
the action matches the cue's target, -1 otherwise. The two cues reach the neuron
through separate input groups, so reward-modulated STDP tunes each group on its
own: GO weights ratchet up (rewarded exploratory firing), NO-GO weights ratchet
down (punished firing). The agent learns the contingency.

WHY THIS VALIDATES GROUPING. The batched engine runs many agents on one leading
axis B. If lanes leaked into each other, an agent could not learn a contingency
that differs from its batch-mates. We check three things:

  1. independence  -- a spec run alone and the same spec run grouped give
                       BIT-IDENTICAL weights and reward (per-lane RNG seeded from
                       the spec, not the batch position).
  2. shuffle       -- a lane's result is invariant to where it sits in the batch.
  3. cross-task    -- two agents with OPPOSITE contingencies, trained in the same
                       batch, each reach ~100% on their own task.

Run: python -m snn2.conditioning
"""
from __future__ import annotations
import numpy as np

from .engine import run_bucket
from .spec import expand


# --------------------------------------------------------------------------- #
# Learning
# --------------------------------------------------------------------------- #
def learning_curves(n_agents: int = 16, length: int = 400):
    """Train n_agents on the GO/NO-GO task (and matched lr=0 controls)."""
    learn, ctrl = [], []
    for sd in range(n_agents):
        m = run_bucket([expand({"preset": "instrumental", "len_episode": length})],
                       seed=sd, log_trials=True)[0]
        c = run_bucket([expand({"preset": "instrumental", "len_episode": length,
                                "lr": 0.0})], seed=sd, log_trials=True)[0]
        learn.append(m["trial_correct"])
        ctrl.append(c["trial_correct"])
    return np.array(learn), np.array(ctrl)


# --------------------------------------------------------------------------- #
# Grouping validation
# --------------------------------------------------------------------------- #
def check_independence(seed: int = 7):
    specs = [{"preset": "instrumental", "lr": lr, "len_episode": 200}
             for lr in (0.0, 0.005, 0.01, 0.02, 0.03)]
    grouped = run_bucket([expand(s) for s in specs], seed=seed)
    solo = [run_bucket([expand(s)], seed=seed)[0] for s in specs]
    rwd = max(abs(g["final_reward"] - s["final_reward"])
              for g, s in zip(grouped, solo))
    wgt = max(float(np.abs(g["weights"] - s["weights"]).max())
              for g, s in zip(grouped, solo))
    return rwd, wgt, grouped, specs


def check_shuffle(grouped, specs, seed: int = 7):
    order = [3, 0, 4, 1, 2]
    shuf = run_bucket([expand(specs[i]) for i in order], seed=seed)
    return all(abs(shuf[k]["final_reward"] - grouped[order[k]]["final_reward"]) < 1e-12
               for k in range(len(order)))


def check_cross_task(seed: int = 5, length: int = 400):
    """Opposite contingencies in one batch; cross-evaluate the learned weights."""
    A = {"preset": "instrumental", "target_map": [1, 0], "len_episode": length}
    B = {"preset": "instrumental", "target_map": [0, 1], "len_episode": length}
    res = run_bucket([expand(A), expand(B)], seed=seed, log_trials=True)
    return res[0]["trial_correct"][-50:].mean(), res[1]["trial_correct"][-50:].mean()


def validate(verbose: bool = True) -> dict:
    rwd, wgt, grouped, specs = check_independence()
    assert rwd == 0.0 and wgt == 0.0, "grouped != solo: lanes are not independent!"
    shuffle_ok = check_shuffle(grouped, specs)
    assert shuffle_ok, "lane result depends on batch position"
    accA, accB = check_cross_task()
    assert accA > 0.9 and accB > 0.9, "opposite-contingency agents did not both learn"

    learn, ctrl = learning_curves(n_agents=16, length=300)
    conv = int((learn[:, -50:].mean(1) > 0.9).sum())
    assert conv >= 14, "instrumental learning not robust across seeds"

    facts = {
        "solo_vs_grouped_reward_diff": rwd,
        "solo_vs_grouped_weight_diff": wgt,
        "shuffle_invariant": shuffle_ok,
        "cross_task_accA": float(accA), "cross_task_accB": float(accB),
        "learn_acc_first20": float(learn[:, :20].mean()),
        "learn_acc_last50": float(learn[:, -50:].mean()),
        "ctrl_acc_last50": float(ctrl[:, -50:].mean()),
        "converged_seeds": f"{conv}/16",
        "checks_passed": 4,
    }
    if verbose:
        for k, v in facts.items():
            print(f"  {k:>28}: {v}")
        print("  ok  grouping is independent, position-invariant, and learns")
    return facts


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def make_figure(path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    learn, ctrl = learning_curves(n_agents=16, length=300)
    rwd, wgt, grouped, specs = check_independence()
    accA, accB = check_cross_task()

    def smooth(x, w=15):
        return np.convolve(x, np.ones(w) / w, "valid")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    # (1) learning curve: mean +/- IQR band, vs control
    ax = axes[0]
    lm = learn.mean(0); cm = ctrl.mean(0)
    t = np.arange(len(smooth(lm)))
    ax.plot(t, smooth(lm), color="#16a34a", lw=2, label="R-STDP (learn)")
    ax.fill_between(np.arange(learn.shape[1]),
                    np.percentile(learn, 25, axis=0),
                    np.percentile(learn, 75, axis=0),
                    color="#16a34a", alpha=0.15)
    ax.plot(np.arange(len(smooth(cm))), smooth(cm), color="#9ca3af", lw=2,
            ls="--", label="control (lr=0)")
    ax.axhline(0.5, color="#d1d5db", lw=0.8, ls=":")
    ax.set_title("Instrumental conditioning\n(GO/NO-GO, 16 agents)", fontsize=11)
    ax.set_xlabel("trial"); ax.set_ylabel("accuracy")
    ax.set_ylim(0.3, 1.05); ax.legend(fontsize=9, frameon=False)

    # (2) grouping independence: solo vs grouped scatter (should lie on y=x)
    ax = axes[1]
    solo = [run_bucket([expand(s)], seed=7)[0]["final_reward"] for s in specs]
    grp = [g["final_reward"] for g in grouped]
    ax.scatter(solo, grp, color="#2563eb", s=60, zorder=3)
    lim = [min(solo + grp) - 0.05, max(solo + grp) + 0.05]
    ax.plot(lim, lim, color="#9ca3af", ls="--", lw=1)
    ax.set_title("Grouping independence\n(max diff = %.0e)" % max(rwd, wgt),
                 fontsize=11)
    ax.set_xlabel("reward, run alone"); ax.set_ylabel("reward, run grouped")
    ax.set_xlim(*lim); ax.set_ylim(*lim)

    # (3) cross-contingency: opposite agents in one batch both learn
    ax = axes[2]
    ax.bar(["agent A\n(GO=cue0)", "agent B\n(GO=cue1)"], [accA, accB],
           color=["#7c3aed", "#db2777"])
    ax.axhline(0.5, color="#d1d5db", lw=0.8, ls=":")
    ax.set_title("Cross-contingency independence\n(opposite tasks, same batch)",
                 fontsize=11)
    ax.set_ylabel("final accuracy"); ax.set_ylim(0, 1.1)
    for i, v in enumerate([accA, accB]):
        ax.text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=10)

    fig.suptitle("snn2 — grouping validated via instrumental conditioning",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    print("Validating grouping via instrumental conditioning:")
    validate()


if __name__ == "__main__":
    main()
