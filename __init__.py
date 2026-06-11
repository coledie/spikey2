"""snn2 -- spec-driven, batched, validated spiking-net experiments."""
from .api import run, sweep, schedule, schedule_ray, tune_run
from .registry import register, get, names
from .spec import expand, spec_hash, PRESETS, DEFAULTS
from . import florian

__all__ = [
    "run", "sweep", "schedule", "schedule_ray", "tune_run",
    "register", "get", "names", "expand", "spec_hash", "PRESETS", "DEFAULTS",
    "florian",
]
