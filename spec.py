"""
Spec layer. An experiment is a flat dict. Presets carry the defaults; a spec
carries only the deltas. `expand` turns a 3-line spec into the fully-resolved
config that actually runs -- and the resolved config (not the sparse input) is
what gets hashed, so reproducibility stays honest even when the input is tiny.
"""
from __future__ import annotations
import hashlib
import json
import numpy as np

# Registry-level defaults: anything here may be omitted from a spec.
DEFAULTS = {
    "potential_decay": 0.05,
    "resting_mv": 0.0,
    "firing_threshold": 8.0,
    "refractory_period": 0,
    "action_threshold": 0.0,
    "trace_decay": 0.10,
    "max_weight": 2.0,
    "reward_mult": 1.0,
    "punish_mult": 0.0,
    # regular-spiking Izhikevich
    "izhi_a": 0.02, "izhi_b": 0.2, "izhi_c": -65.0, "izhi_d": 8.0,
}

# Whole-experiment presets. The LLM names one and overrides a few keys.
PRESETS = {
    "izhi_randstate": {
        "game": "randstate", "n_states": 10,
        "neuron": "izhikevich",
        "input": "ratemap",
        "synapse": "ltp",            # reward-modulated LTP (validated in stdp.py)
        "readout": "threshold",
        "reward": "fire_states", "reward_fire_states": [0, 3, 6, 9],
        "n_inputs": 100, "n_neurons": 50,
        "processing_time": 100, "stdp_window": 100,
        "lr": 0.1,
        # Izhikevich fires near +30 mV, so it needs real input current, not the
        # tiny LIF-scale drive; the preset owns that detail so the user never sees it.
        "input_gain": 5.0,
        "len_episode": 100,
    },
}


def _state_rate_map(n_states, n_inputs, rate=0.2):
    """state k -> its own contiguous block of input neurons fires at `rate`."""
    m = np.zeros((n_states, n_inputs))
    block = n_inputs // n_states
    for k in range(n_states):
        m[k, k * block:(k + 1) * block] = rate
    return m


def expand(spec: dict) -> dict:
    """sparse spec -> fully-resolved config (preset < defaults < spec)."""
    out = {}
    if "preset" in spec:
        out.update(PRESETS[spec["preset"]])
    for k, v in DEFAULTS.items():
        out.setdefault(k, v)
    out.update({k: v for k, v in spec.items() if k != "preset"})
    out.setdefault("input_gain", 1.0)

    # Derived -- never specified by hand.
    out.setdefault("n_outputs", out["n_neurons"])
    out["state_rate_map"] = _state_rate_map(out["n_states"], out["n_inputs"])
    if "preset" in spec:
        out["preset"] = spec["preset"]
    return out


def _jsonable(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def spec_hash(resolved: dict) -> str:
    """Content address of a resolved config -> dedup, resume, result keys."""
    payload = {k: _jsonable(v) for k, v in resolved.items() if not k.startswith("_")}
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]
