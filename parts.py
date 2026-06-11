"""
Concrete parts, each a pure function, each registered under a string name.

Conventions
-----------
Every part is batched over experiments on the leading axis B. None of them
loop over experiments or neurons in Python -- those are array axes. The only
Python loops in the whole engine are over *time* (and over shape-buckets in
the scheduler). That is what keeps the interpreter (GIL or not) out of the way.

State dicts hold arrays of shape [B, N] (body) so a part can carry whatever
internal variables its dynamics need (Izhikevich needs a recovery var u; LIF
needs a refractory counter).
"""
from __future__ import annotations
import numpy as np
from .registry import register


# --------------------------------------------------------------------------- #
# Neurons  (state, I, p, rng) -> (state, fired[B,N] bool)
# --------------------------------------------------------------------------- #
@register("neuron", "lif")
def lif(state, I, p, rng):
    v = state["v"] * (1.0 - p["potential_decay"]) + I
    can_fire = state["refrac"] <= 0
    fired = (v >= p["firing_threshold"]) & can_fire
    state["v"] = np.where(fired, p["resting_mv"], v)
    state["refrac"] = np.where(fired, p["refractory_period"],
                               np.maximum(state["refrac"] - 1, 0))
    return state, fired


@register("neuron", "izhikevich")
def izhikevich(state, I, p, rng):
    """Simple model of spiking neurons (Izhikevich 2003): the real a,b,c,d
    recovery-variable dynamics, not the LIF stand-in the repo notebook uses.
    Two 0.5 ms Euler sub-steps for numerical stability, as in the paper."""
    a, b, c, d = p["izhi_a"], p["izhi_b"], p["izhi_c"], p["izhi_d"]
    v, u = state["v"], state["u"]
    for _ in range(2):
        v = v + 0.5 * (0.04 * v * v + 5 * v + 140 - u + I)
    u = u + a * (b * v - u)
    fired = v >= 30.0
    state["v"] = np.where(fired, c, v)
    state["u"] = np.where(fired, u + d, u)
    return state, fired


def init_neuron_state(model, B, N, p):
    if model == "izhikevich":
        b = p["izhi_b"]
        v = np.full((B, N), p["izhi_c"], dtype=np.float64)
        return {"v": v, "u": b * v}
    return {"v": np.full((B, N), p["resting_mv"], dtype=np.float64),
            "refrac": np.zeros((B, N), dtype=np.float64)}


# --------------------------------------------------------------------------- #
# Inputs  (state_int[B], rng, p) -> input_spikes[B, n_inputs] bool
# --------------------------------------------------------------------------- #
@register("input", "ratemap")
def ratemap(state_int, rng, p):
    rates = p["state_rate_map"][state_int]          # [B, n_inputs] fire prob
    return rng.random(rates.shape) < rates


# --------------------------------------------------------------------------- #
# Readouts  (out_spikes[B, T, n_outputs]) -> action[B]
# --------------------------------------------------------------------------- #
@register("readout", "threshold")
def threshold(out_spikes, p):
    return (out_spikes.mean(axis=(1, 2)) >= p["action_threshold"]).astype(int)


@register("readout", "population")
def population(out_spikes, p):
    rates = out_spikes.mean(axis=1)                 # [B, n_outputs]
    return rates.argmax(axis=1)


# --------------------------------------------------------------------------- #
# Rewards  (state[B], action[B], state_next[B], p) -> reward[B]
# --------------------------------------------------------------------------- #
@register("reward", "fire_states")
def fire_states(state, action, state_next, p):
    hit = np.isin(state, p["reward_fire_states"])
    return np.where(hit, p["reward_mult"], p["punish_mult"]).astype(np.float64)


# --------------------------------------------------------------------------- #
# Games  -- minimal RL envs, batched.
# --------------------------------------------------------------------------- #
@register("game", "randstate")
def randstate(rng, p):
    """Random-state task from examples/izhikevich2007.ipynb: each step the
    environment shows a uniformly random integer state in [0, n_states)."""
    B, n = p["_B"], p["n_states"]

    def reset():
        return rng.integers(0, n, size=B)

    def step(state, action):
        nxt = rng.integers(0, n, size=B)
        done = np.zeros(B, dtype=bool)              # length handled by scheduler
        return nxt, done

    return reset, step
