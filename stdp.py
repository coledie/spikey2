"""
The learning rule, isolated as pure functions so it can be validated.

`stdp_delta`      : fast, vectorized, batched over experiments. The hot path.
`stdp_delta_ref`  : an obviously-correct quadruple loop. The oracle. Never shipped hot.

Both compute the *reward-modulated LTP* update used by Spikey's RLSTDP/LTP synapse:
for a source that fired earlier in the window and a body neuron that fires now,
increase the connecting weight by a recency-weighted, reward-scaled amount.

    dts[i]    = ( sum_t recency(t) * pre_fired[t, i] ) * polarity[i] * lr/window * trace
    dW[i, j]  = dts[i]   for every body neuron j that fired this step

The two implementations MUST agree byte-for-byte on random inputs (golden test in
validate.py) and MUST satisfy the analytic sign/zero properties. That is what makes
the rule "truly validatable" rather than merely shape-checked.
"""
from __future__ import annotations
import numpy as np


def _recency(window: int, T: int) -> np.ndarray:
    """Recency weights for the T-1 history rows; more recent -> larger."""
    return np.arange(window - T, window - 1).astype(np.float64)  # [T-1]


def stdp_delta_ref(spike_log, trace, lr, window, n_inputs, polarity):
    """Reference oracle. spike_log: [T, S] bool, spike_log[-1] == this step.

    Returns dW: [S, N] where S = n_inputs + N (sources), N = body neurons.
    Deliberately written as explicit loops for auditability.
    """
    spike_log = np.asarray(spike_log, dtype=np.float64)
    T, S = spike_log.shape
    N = S - n_inputs
    rec = _recency(window, T)
    post = spike_log[-1]
    dW = np.zeros((S, N))
    for i in range(S):                       # source unit
        dts_i = 0.0
        for t in range(T - 1):               # history rows
            dts_i += rec[t] * spike_log[t, i]
        dts_i *= polarity[i] * lr / window * trace
        for j in range(N):                   # body target
            if post[n_inputs + j]:
                dW[i, j] += dts_i
    return dW


def stdp_delta(spike_log, trace, lr, window, n_inputs, polarity):
    """Vectorized single-experiment version. Same contract as the oracle."""
    spike_log = np.asarray(spike_log, dtype=np.float64)
    T, S = spike_log.shape
    N = S - n_inputs
    rec = _recency(window, T)[:, None]                       # [T-1, 1]
    dts = (rec * spike_log[:-1]).sum(0) * polarity * (lr / window * trace)  # [S]
    post_body = spike_log[-1, n_inputs:]                     # [N]
    return np.outer(dts, post_body)                          # [S, N]


def stdp_delta_batched(spike_log, trace, lr, window, n_inputs, polarity):
    """Batched over experiments. The leading axis B is the whole point.

    spike_log : [B, T, S] bool
    trace     : [B]
    polarity  : [B, S]   (+1 excitatory, -1 inhibitory)
    returns dW: [B, S, N]
    """
    spike_log = np.asarray(spike_log, dtype=np.float64)
    B, T, S = spike_log.shape
    N = S - n_inputs
    rec = _recency(window, T)[None, :, None]                 # [1, T-1, 1]
    dts = (rec * spike_log[:, :-1]).sum(1)                   # [B, S]
    dts = dts * polarity * (lr / window) * trace[:, None]    # [B, S]
    post_body = spike_log[:, -1, n_inputs:]                  # [B, N]
    return dts[:, :, None] * post_body[:, None, :]           # [B, S, N]
