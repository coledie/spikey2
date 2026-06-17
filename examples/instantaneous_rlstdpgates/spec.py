"""
Spec layer for examples2 -- identical philosophy to the spiking build (a flat
dict, a preset carrying defaults, `expand` resolving the rest) but the presets
describe an *instantaneous* network: the neuron is a closed-form transfer
function, not a stepped simulation.

The numeric values (gains, thresholds, band edges, learning rates) are inherited
from the spiking presets in `examples/logicgates` and `examples/izh` so the two
builds are directly comparable.
"""
from __future__ import annotations
import numpy as np

# Logic-gate truth tables, indexed by combo = 2*a + b.
GATES = {
    "OR":   [0, 1, 1, 1],
    "AND":  [0, 0, 0, 1],
    "XOR":  [0, 1, 1, 0],
    "NAND": [1, 1, 1, 0],
    "NOR":  [1, 0, 0, 0],
}

DEFAULTS = {
    "potential_decay": 0.05,
    "resting_mv": 0.0,
    "firing_threshold": 8.0,
    "action_threshold": 0.0,
    "stdp_window": 20,
    "max_weight": 2.0,
    "reward_mult": 1.0,
    "punish_mult": 0.0,
    "input_gain": 1.0,
    "izhi_a": 0.02, "izhi_b": 0.2, "izhi_c": -65.0, "izhi_d": 8.0,
}

PRESETS = {
    # Random-state task, instantaneous Izhikevich (type-I f-I curve).
    "izhi_randstate": {
        "game": "randstate", "n_states": 10,
        "neuron": "izhikevich", "reward": "fire_states",
        "reward_fire_states": [0, 3, 6, 9], "readout": "threshold",
        "n_inputs": 100, "n_neurons": 50,
        "processing_time": 100, "stdp_window": 100,
        "lr": 0.1, "input_gain": 5.0, "len_episode": 100,
        "encoding": "state", "state_rate": 0.2,
        "action_threshold": 0.0,
    },
    # Logic-gate curriculum on a monotonic (LIF) point neuron.
    "logic_monotonic": {
        "game": "cue", "n_cues": 4,
        "neuron": "lif", "reward": "match_action", "readout": "threshold",
        "n_inputs": 40, "n_neurons": 1,
        "processing_time": 30, "stdp_window": 20,
        "firing_threshold": 10.0, "action_threshold": 0.05,
        "input_gain": 0.8, "max_weight": 0.5,
        "lr": 0.025, "reward_mult": 1.0, "punish_mult": -0.5,
        "operand_rate": 0.5, "target_map": [0, 1, 1, 1],
        "len_episode": 1000, "encoding": "operand",
    },
    # Same task, non-monotonic dCaAP (band-pass) neuron -> solves XOR.
    "logic_dendritic": {
        "game": "cue", "n_cues": 4,
        "neuron": "dendritic", "reward": "match_action", "readout": "threshold",
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
    m = np.zeros((n_states, n_inputs))
    block = n_inputs // n_states
    for k in range(n_states):
        m[k, k * block:(k + 1) * block] = rate
    return m


def _operand_rate_map(n_inputs, rate=0.5):
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
    out = {}
    if "preset" in spec:
        out.update(PRESETS[spec["preset"]])
    for k, v in DEFAULTS.items():
        out.setdefault(k, v)
    out.update({k: v for k, v in spec.items() if k != "preset"})
    out.setdefault("input_gain", 1.0)
    out.setdefault("n_outputs", out["n_neurons"])
    # elig_gain: closed-form magnitude of the accumulated recency-weighted
    # eligibility the spiking engine sums over its P-step, W-window inner loop
    # (mean-field value ~ P * window / 2). This is the single constant that lets
    # an O(1) rate update stand in for that O(P*window) accumulation.
    out.setdefault("elig_gain", 0.5 * out["processing_time"] * out["stdp_window"])
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
