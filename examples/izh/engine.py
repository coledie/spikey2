"""
Batched engine. One leading axis B = many experiments at once. A single loop
over time advances every lane; an `active` mask freezes lanes that have hit
their (possibly different) episode length.

Per-lane reproducibility: each lane is seeded from (global_seed, spec_hash), so a
lane's randomness does NOT depend on which other lanes share its batch. That is
what makes grouping *validatable*: a spec run alone and the same spec run grouped
produce identical results, and shuffling a batch changes nothing per lane.

Games here are trial-based: a game returns a per-lane sequence of cue/stimulus
integers [B, n_steps] (each step is one trial), which keeps the hot loop fully
vectorized. Actions do not change the cue sequence (independent trials), as in
classical/instrumental conditioning experiments.
"""
from __future__ import annotations
import numpy as np

from . import parts as _parts          # registers the parts
from .registry import get
from .spec import spec_hash
from .stdp import stdp_delta_batched


def _lane_rngs(specs, seed):
    out = []
    for s in specs:
        ls = int(spec_hash(s), 16) % (2 ** 32)
        out.append(np.random.default_rng(np.random.SeedSequence([seed, ls])))
    return out


def run_bucket(specs: list[dict], seed: int = 0, log_trials: bool = False) -> list[dict]:
    """Run a bucket of same-shape specs, batched on axis B."""
    p0 = specs[0]
    B = len(specs)
    N = p0["n_neurons"]; n_in = p0["n_inputs"]; n_out = p0["n_outputs"]
    S = n_in + N
    W_T = p0["stdp_window"]; P = p0["processing_time"]

    rngs = _lane_rngs(specs, seed)

    lr = np.array([s["lr"] for s in specs])
    trace_decay = np.array([s["trace_decay"] for s in specs])
    max_w = np.array([s["max_weight"] for s in specs])
    gain = np.array([s["input_gain"] for s in specs])
    len_ep = np.array([s["len_episode"] for s in specs])
    n_steps = int(len_ep.max())

    p = dict(p0); p["_B"] = B
    if "target_map" in p0:                                   # per-lane contingency
        p["target_map"] = np.stack([np.asarray(s["target_map"]) for s in specs])
    p["reward_mult"] = np.array([s["reward_mult"] for s in specs])
    p["punish_mult"] = np.array([s["punish_mult"] for s in specs])

    neuron_model = p0["neuron"]
    nstate = _parts.init_neuron_state(neuron_model, B, N, p)
    neuron_step = get("neuron", neuron_model)
    input_rates = p0["state_rate_map"]                       # [n_states, n_inputs]
    readout = get("readout", p0["readout"])
    reward_fn = get("reward", p0["reward"])
    cues = get("game", p0["game"])(rngs, p, n_steps)         # [B, n_steps] int

    # weights + noise: per-lane streams, drawn before the hot loop
    W = np.zeros((B, S, N))
    for b in range(B):
        W[b, :n_in, :] = rngs[b].uniform(0, 0.5, size=(n_in, N))
    noise = np.stack([rngs[b].random((n_steps, P, n_in)) for b in range(B)])

    polarity = np.ones((B, S))
    trace = np.zeros(B)
    spike_log = np.zeros((B, W_T + P, S))
    reward_sum = np.zeros(B); out_rate = np.zeros(B)
    trial_reward = np.zeros((B, n_steps)) if log_trials else None
    trial_correct = np.zeros((B, n_steps)) if log_trials else None

    for t in range(n_steps):
        active = t < len_ep
        cue = cues[:, t]                                     # [B]
        rates = input_rates[cue]                             # [B, n_inputs]
        # independent trials: clear history and neuron state at trial start
        spike_log[:] = 0.0
        nstate = _parts.init_neuron_state(neuron_model, B, N, p)
        elig = np.zeros((B, S, N))                           # per-trial eligibility
        for i in range(P):
            in_spk = noise[:, t, i] < rates                  # [B, n_inputs] bool
            prev = spike_log[:, W_T + i - 1] if i > 0 else spike_log[:, W_T - 1]
            I = np.einsum("bs,bsn->bn", prev, W) * gain[:, None]
            nstate, body_fired = neuron_step(nstate, I, p, None)
            row = np.concatenate([in_spk.astype(np.float64),
                                  body_fired.astype(np.float64)], axis=1)
            spike_log[:, W_T + i] = row
            win = spike_log[:, i:i + W_T + 1]
            elig += stdp_delta_batched(win, np.ones(B), lr=1.0, window=W_T,
                                       n_inputs=n_in, polarity=polarity)

        out = spike_log[:, -P:, -n_out:]
        action = readout(out, p)                             # [B]
        r = reward_fn(cue, action, cue, p)                   # [B]
        # reward modulates THIS trial's eligibility -> correct credit assignment
        W = W + lr[:, None, None] * r[:, None, None] * elig * active[:, None, None]
        np.clip(W, 0, max_w[:, None, None], out=W)

        r = r * active
        reward_sum += r
        out_rate += out.mean(axis=(1, 2)) * active
        if log_trials:
            trial_reward[:, t] = r
            if "target_map" in p:
                tgt = p["target_map"][np.arange(B), cue]
                trial_correct[:, t] = (action == tgt) * active

    results = []
    for b in range(B):
        m = {"spec": specs[b],
             "final_reward": float(reward_sum[b] / max(1, len_ep[b])),
             "mean_out_rate": float(out_rate[b] / max(1, len_ep[b])),
             "weight_norm": float(np.linalg.norm(W[b])),
             "weights": W[b].copy()}
        if log_trials:
            m["trial_reward"] = trial_reward[b, :len_ep[b]].copy()
            m["trial_correct"] = trial_correct[b, :len_ep[b]].copy()
        results.append(m)
    return results
