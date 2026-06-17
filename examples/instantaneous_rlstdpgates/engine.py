"""
Instantaneous engine. The spiking `run_bucket` has TWO nested time loops per
trial (P micro-steps, each running STDP over a window); this `run_instant` has
NONE. A trial is pure algebra:

    pre-rate  ->  (mean, var) of drive   [input_moments]      # feed-forward sum
              ->  firing rate f          [closed-form f-I]    # the Laplace step
              ->  realised rate + noise  [one Gaussian draw]  # CLT exploration
              ->  action, reward
              ->  rate-based reward-modulated Hebbian weight update

The outer loop over trials remains -- that is the *learning* timescale, not the
membrane timescale -- and we still batch over independent seeds on a leading
axis B, exactly like the spiking engine batches over experiments.

LEARNING RULE (rate limit of reward-modulated STDP). The spiking rule sums, over
the window, recency-weighted pre-spikes times the post-spike, then scales by
reward. In expectation over the rate-coded window that collapses to the classic
pair-STDP rate limit

    dW_ij  =  lr * r * <pre_rate_i> * <post_rate_j>,

i.e. a reward-gated outer product of input and output rates. Weights are clipped
to [0, max_weight] and kept feed-forward, matching the spiking engine.
"""
from __future__ import annotations
import numpy as np

from . import transfer as T


def _firing_rate(neuron, mean, var, p, cal):
    if neuron == "lif":
        return T.lif_rate(mean, p["potential_decay"], p["firing_threshold"],
                          p["resting_mv"])
    if neuron == "dendritic":
        return T.band_rate(mean, var, p["dcap_lo"], p["dcap_hi"])
    if neuron == "izhikevich":
        # noise-aware expected rate: the mean drive can sit near rheobase, so the
        # firing comes from the upper tail of the input fluctuations.
        return T.expected_rate(lambda I: T.izhi_rate(I, cal[0], cal[1]), mean, var)
    raise ValueError("unknown neuron %r" % neuron)


def _reward(p, cue, action, target_map):
    if p["reward"] == "fire_states":
        hit = np.isin(cue, p["reward_fire_states"])
        return np.where(hit, p["reward_mult"], p["punish_mult"]).astype(np.float64)
    if p["reward"] == "match_action":
        B = len(cue)
        target = target_map[np.arange(B), cue]
        correct = action == target
        return np.where(correct, p["reward_mult"], p["punish_mult"]).astype(np.float64), target
    raise ValueError("unknown reward %r" % p["reward"])


def run_instant(p: dict, n_seeds: int = 12, seed: int = 0, log_trials: bool = False):
    """Run `n_seeds` independent instantaneous learners for one resolved spec `p`,
    batched on axis B = n_seeds. Returns one metrics dict aggregated over seeds,
    plus per-seed curves/weights when `log_trials`.
    """
    B = n_seeds
    n_in = p["n_inputs"]; n_out = p["n_outputs"]
    P = p["processing_time"]; gain = float(p["input_gain"])
    lr = float(p["lr"]); max_w = float(p["max_weight"])
    elig_gain = float(p["elig_gain"])
    n_steps = int(p["len_episode"])
    neuron = p["neuron"]
    rate_map = p["state_rate_map"]                       # [n_cells, n_in]
    rng = np.random.default_rng(seed)

    cal = (0.0, 0.0)
    if neuron == "izhikevich":
        cal = T.calibrate_izhi(p)

    # per-seed initial weights and cue sequences
    W = rng.uniform(0.0, 0.5, size=(B, n_in, n_out))
    if p["game"] == "randstate":
        cues = rng.integers(0, p["n_states"], size=(B, n_steps))
    else:
        cues = rng.integers(0, p["n_cues"], size=(B, n_steps))

    target_map = None
    if p["reward"] == "match_action":
        tm = np.asarray(p["target_map"])
        target_map = np.broadcast_to(tm, (B, len(tm)))

    reward_sum = np.zeros(B); out_rate_sum = np.zeros(B)
    trial_correct = np.zeros((B, n_steps)) if log_trials else None
    trial_reward = np.zeros((B, n_steps)) if log_trials else None

    for t in range(n_steps):
        cue = cues[:, t]                                 # [B]
        pre_rate = rate_map[cue]                          # [B, n_in]
        mean, var = T.input_moments(pre_rate, W, gain)    # [B, n_out]
        f = _firing_rate(neuron, mean, var, p, cal)       # [B, n_out]
        realized = T.realize(f, P, rng)                   # [B, n_out]
        out_mean = realized.mean(axis=1)                  # [B]
        action = (out_mean >= p["action_threshold"]).astype(int)

        if p["reward"] == "match_action":
            r, target = _reward(p, cue, action, target_map)
            correct = (action == target)
        else:
            r = _reward(p, cue, action, target_map)
            correct = None

        # reward-modulated rate Hebbian: dW = lr * elig_gain * r * pre (x) post.
        # elig_gain is the closed-form magnitude of the eligibility the spiking
        # engine accrues over its inner P-step / W-window loop (see spec.expand).
        dW = lr * elig_gain * r[:, None, None] * pre_rate[:, :, None] * realized[:, None, :]
        W = np.clip(W + dW, 0.0, max_w)

        reward_sum += r
        out_rate_sum += out_mean
        if log_trials:
            trial_reward[:, t] = r
            if correct is not None:
                trial_correct[:, t] = correct

    out = {
        "final_reward": float(reward_sum.mean() / max(1, n_steps)),
        "mean_out_rate": float(out_rate_sum.mean() / max(1, n_steps)),
        "weight_norm": float(np.mean([np.linalg.norm(W[b]) for b in range(B)])),
        "weights": W,
        "calibration": cal,
    }
    if log_trials:
        out["trial_correct"] = trial_correct
        out["trial_reward"] = trial_reward
    return out
