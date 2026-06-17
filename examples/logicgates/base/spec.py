"""
Spec layer (examples build). An experiment is a flat dict; a preset carries the
defaults and the spec carries only the deltas. `expand` resolves a sparse spec
into the full config that actually runs (and that gets hashed).

This is the *trial-based* build used by the conditioning and logic-gate
examples: games return a per-lane cue sequence and the engine scopes
eligibility to each trial. It is a superset of the izhi_randstate preset.
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
    "input_gain": 1.0,
    # regular-spiking Izhikevich
    "izhi_a": 0.02, "izhi_b": 0.2, "izhi_c": -65.0, "izhi_d": 8.0,
}

# Logic-gate truth tables, indexed by combo = 2*a + b.
GATES = {
    "OR":   [0, 1, 1, 1],
    "AND":  [0, 0, 0, 1],
    "XOR":  [0, 1, 1, 0],
    "NAND": [1, 1, 1, 0],
    "NOR":  [1, 0, 0, 0],
}

# Whole-experiment presets. Name one and override a few keys.
PRESETS = {
    "izhi_randstate": {
        "game": "randstate", "n_states": 10,
        "neuron": "izhikevich", "input": "ratemap",
        "synapse": "ltp", "readout": "threshold",
        "reward": "fire_states", "reward_fire_states": [0, 3, 6, 9],
        "n_inputs": 100, "n_neurons": 50,
        "processing_time": 100, "stdp_window": 100,
        "lr": 0.1, "input_gain": 5.0, "len_episode": 100,
        "encoding": "state", "state_rate": 0.2,
    },
    # GO / NO-GO instrumental conditioning: 2 cues -> 2 input groups, one output
    # neuron, binary action (fire / stay silent), reward when action matches the
    # cue's target. Linearly separable, so a monotonic point neuron solves it.
    "instrumental": {
        "game": "cue", "n_cues": 2,
        "neuron": "lif", "input": "ratemap",
        "synapse": "ltp", "readout": "threshold", "reward": "match_action",
        "n_inputs": 40, "n_neurons": 1,
        "processing_time": 30, "stdp_window": 20,
        "firing_threshold": 10.0, "action_threshold": 0.05,
        "input_gain": 0.8, "max_weight": 0.5,
        "lr": 0.03, "reward_mult": 1.0, "punish_mult": -0.5,
        "target_map": [1, 0], "len_episode": 400,
        "encoding": "cue", "cue_rate": 0.5,
    },
    # Logic-gate curriculum on a monotonic LIF point neuron (OR / AND / XOR).
    "logic_monotonic": {
        "game": "cue", "n_cues": 4,
        "neuron": "lif", "input": "ratemap",
        "synapse": "ltp", "readout": "threshold", "reward": "match_action",
        "n_inputs": 40, "n_neurons": 1,
        "processing_time": 30, "stdp_window": 20,
        "firing_threshold": 10.0, "action_threshold": 0.05,
        "input_gain": 0.8, "max_weight": 0.5,
        "lr": 0.025, "reward_mult": 1.0, "punish_mult": -0.5,
        "operand_rate": 0.5, "target_map": [0, 1, 1, 1],
        "len_episode": 1000, "encoding": "operand",
    },
    # Same task, dCaAP (band-pass / anti-coincidence) neuron -> solves XOR.
    "logic_dendritic": {
        "game": "cue", "n_cues": 4,
        "neuron": "dendritic", "input": "ratemap",
        "synapse": "ltp", "readout": "threshold", "reward": "match_action",
        "n_inputs": 40, "n_neurons": 1,
        "processing_time": 30, "stdp_window": 20,
        "dcap_lo": 3.0, "dcap_hi": 6.0, "action_threshold": 0.1,
        "input_gain": 1.0, "max_weight": 0.45,
        "lr": 0.04, "reward_mult": 1.0, "punish_mult": 0.0,
        "operand_rate": 0.5, "target_map": [0, 1, 1, 0],
        "len_episode": 1000, "encoding": "operand",
    },
}


def _state_rate_map(n_states, n_inputs, rate=0.2):
    """state k -> its own contiguous block of input neurons fires at `rate`."""
    m = np.zeros((n_states, n_inputs))
    block = n_inputs // n_states
    for k in range(n_states):
        m[k, k * block:(k + 1) * block] = rate
    return m


def _operand_rate_map(n_inputs, rate=0.5):
    """combo (2*a + b) -> [n_inputs] fire-prob. Two operand groups: group A is
    the first half (operand a), group B the second half (operand b). A group
    fires at `rate` when its operand is 1, else stays silent."""
    half = n_inputs // 2
    m = np.zeros((4, n_inputs))
    for combo in range(4):
        a, b = combo >> 1, combo & 1
        if a:
            m[combo, :half] = rate
        if b:
            m[combo, half:] = rate
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
    enc = out.get("encoding", "state")
    if enc == "operand":
        out["state_rate_map"] = _operand_rate_map(out["n_inputs"],
                                                  out.get("operand_rate", 0.5))
    elif enc == "cue":
        out["state_rate_map"] = _state_rate_map(out["n_cues"], out["n_inputs"],
                                                out.get("cue_rate", 0.5))
    else:
        out["state_rate_map"] = _state_rate_map(out["n_states"], out["n_inputs"],
                                                out.get("state_rate", 0.2))
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
