"""
Batched engine ("batched"). One leading axis B = many experiments at once. A
single `step()` advances every lane; an `active` mask freezes lanes that have
hit their (possibly different) episode length, so heterogeneous and
logically-staggered runs share one loop. The only Python loops are over time.

Games here are STATEFUL: a game returns a (reset, step) pair and the engine
drives it with actions (reset/env_step), as in the original randstate task.
"""
from __future__ import annotations
import numpy as np

from .. import parts as _parts          # registers the parts
from ..registry import get, register
from ..stdp import stdp_delta_batched


@register("engine", "batched")
def run_bucket(resolved_specs: list[dict], seed: int = 0, **_kw) -> list[dict]:
    """Run a bucket of specs that SHARE network shape, batched on axis B.

    Returns one metrics dict per spec, in input order. Lanes with shorter
    `len_episode` stop contributing once finished (active mask), so the bucket
    need not be homogeneous in episode length or learning rate.
    """
    p0 = resolved_specs[0]
    B = len(resolved_specs)
    N = p0["n_neurons"]
    n_in = p0["n_inputs"]
    n_out = p0["n_outputs"]
    S = n_in + N
    W_T = p0["stdp_window"]
    P = p0["processing_time"]
    rng = np.random.default_rng(seed)

    # Per-lane scalars stacked into arrays.
    lr = np.array([s["lr"] for s in resolved_specs])
    trace_decay = np.array([s["trace_decay"] for s in resolved_specs])
    max_w = np.array([s["max_weight"] for s in resolved_specs])
    gain = np.array([s["input_gain"] for s in resolved_specs])
    len_ep = np.array([s["len_episode"] for s in resolved_specs])

    # Shared param view (shape-bucket => these match across the bucket).
    p = dict(p0)
    p["_B"] = B

    neuron_model = p0["neuron"]
    nstate = _parts.init_neuron_state(neuron_model, B, N, p)
    neuron_step = get("neuron", neuron_model)
    input_step = get("input", p0["input"])
    readout = get("readout", p0["readout"])
    reward_fn = get("reward", p0["reward"])
    reset, env_step = get("game", p0["game"])(rng, p)

    # Weights: feedforward inputs->body, zero body->body (matches the notebook).
    W = np.zeros((B, S, N))
    W[:, :n_in, :] = rng.uniform(0, 0.5, size=(B, n_in, N))
    polarity = np.ones((B, S))                       # all excitatory here
    trace = np.zeros(B)
    spike_log = np.zeros((B, W_T + P, S), dtype=np.float64)

    state = reset()
    n_steps = int(len_ep.max())
    reward_sum = np.zeros(B)
    out_rate = np.zeros(B)

    for t in range(n_steps):
        active = t < len_ep                          # [B] staggered/uneven episodes
        # roll window forward
        spike_log[:, :W_T] = spike_log[:, -W_T:]
        in_spk = input_step(state, rng, p)           # [B, n_in]

        for i in range(P):                           # processing window
            prev = spike_log[:, W_T + i - 1] if i > 0 else spike_log[:, W_T - 1]
            full_prev = prev.astype(np.float64)
            I = np.einsum("bs,bsn->bn", full_prev, W) * gain[:, None]
            nstate, body_fired = neuron_step(nstate, I, p, rng)
            row = np.concatenate([in_spk.astype(np.float64), body_fired.astype(np.float64)], axis=1)
            spike_log[:, W_T + i] = row
            # STDP every network step, gated by trace (reward-modulated)
            win = spike_log[:, i:i + W_T + 1]
            dW = stdp_delta_batched(win, trace, lr=1.0, window=W_T,
                                    n_inputs=n_in, polarity=polarity)
            dW *= lr[:, None, None]
            W = W + dW * active[:, None, None]
            np.clip(W, 0, max_w[:, None, None], out=W)
            trace = trace * (1.0 - trace_decay)

        out = spike_log[:, -P:, -n_out:]
        action = readout(out, p)
        state_next, _done = env_step(state, action)
        r = reward_fn(state, action, state_next, p) * active
        trace = trace + r
        reward_sum += r
        out_rate += out.mean(axis=(1, 2)) * active
        state = state_next

    return [
        {"spec": resolved_specs[b],
         "final_reward": float(reward_sum[b] / max(1, len_ep[b])),
         "mean_out_rate": float(out_rate[b] / max(1, len_ep[b])),
         "weight_norm": float(np.linalg.norm(W[b]))}
        for b in range(B)
    ]
