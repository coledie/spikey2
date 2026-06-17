"""
Registry: the indirection that lets a spec name a model with a string.

A "part" is a pure function. Adding a neuron/encoder/reward model is one
`register(...)` call -- no class, no inheritance. The LLM/user never touches
this file; they just write the registered name in a spec.
"""
from __future__ import annotations

_REGISTRY: dict[str, dict[str, object]] = {
    "neuron": {}, "input": {}, "readout": {}, "reward": {}, "game": {},
}


def register(kind: str, name: str, fn=None):
    """register('neuron', 'izhikevich', fn)  or  @register('neuron','izhikevich')."""
    if kind not in _REGISTRY:
        raise KeyError(f"unknown part kind {kind!r}; one of {list(_REGISTRY)}")

    def _add(f):
        _REGISTRY[kind][name] = f
        return f

    return _add if fn is None else _add(fn)


def get(kind: str, name: str):
    try:
        return _REGISTRY[kind][name]
    except KeyError:
        raise KeyError(
            f"no {kind} named {name!r}; registered: {list(_REGISTRY[kind])}"
        )


def names(kind: str):
    return sorted(_REGISTRY[kind])
