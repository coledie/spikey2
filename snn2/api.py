"""
The whole public surface. An LLM/user touches only these.

    run(spec)            -> metrics for one experiment
    sweep(base, grid)    -> list of specs (cartesian product of deltas)
    schedule(specs)      -> metrics for many, bucketed by shape, run batched
    schedule_ray(specs)  -> same, but each shape-bucket is a Ray task (multi-core)
    tune_run(space, ...) -> Ray Tune async search (ASHA): staggered + early-stop

Ray is optional. Import-safe without it; schedule_ray/tune_run raise a clear
message if Ray is missing.
"""
from __future__ import annotations
from collections import defaultdict
from itertools import product

from .spec import expand, spec_hash
from .registry import get
from . import engines  # noqa: F401  (import registers the engines)


def _engine_for(resolved: dict):
    """Look up the engine a resolved spec names (default set in spec.DEFAULTS)."""
    return get("engine", resolved["engine"])


def _bucket_key(r: dict):
    """Same engine + tensor shape => same bucket => can share one batched run."""
    return (r["engine"], r["neuron"], r["input"], r["readout"], r["game"],
            r["n_inputs"], r["n_neurons"], r["n_outputs"],
            r["processing_time"], r["stdp_window"], r["n_states"])


def run(spec: dict, seed: int = 0) -> dict:
    r = expand(spec)
    return _engine_for(r)([r], seed=seed)[0]


def sweep(base: dict, grid: dict) -> list[dict]:
    keys = list(grid)
    return [{**base, **dict(zip(keys, combo))}
            for combo in product(*(grid[k] for k in keys))]


def _bucketize(specs):
    buckets = defaultdict(list)
    for s in specs:
        r = expand(s)
        buckets[_bucket_key(r)].append(r)
    return buckets


def schedule(specs: list[dict], seed: int = 0) -> dict:
    """Local batched scheduler. Returns {hash: metrics}, deduped by content."""
    out = {}
    for resolved in _bucketize(specs).values():
        engine = _engine_for(resolved[0])
        for m in engine(resolved, seed=seed):
            out[spec_hash(m["spec"])] = m
    return out


# --------------------------------------------------------------------------- #
# Ray layer (optional). Coarse scheduling = Ray; fine batching = engine.
# --------------------------------------------------------------------------- #
def schedule_ray(specs: list[dict], seed: int = 0, num_cpus=None) -> dict:
    try:
        import ray
    except ImportError as e:
        raise ImportError("schedule_ray needs Ray: pip install 'ray[tune]'") from e
    if not ray.is_initialized():
        ray.init(num_cpus=num_cpus, ignore_reinit_error=True, log_to_driver=False)

    @ray.remote
    def _run(resolved, seed):
        engine = get("engine", resolved[0]["engine"])
        return engine(resolved, seed=seed)       # one Ray PROCESS per bucket -> own GIL

    futs = [_run.remote(b, seed) for b in _bucketize(specs).values()]
    out = {}
    for batch in ray.get(futs):
        for m in batch:
            out[spec_hash(m["spec"])] = m
    return out


def tune_run(search_space: dict, num_samples: int = 50, metric: str = "final_reward",
             mode: str = "max", max_concurrent: int | None = None):
    """Ray Tune with an async scheduler: trials start staggered and the loser
    trials are stopped early -- this IS the slot-pool/refill behavior, for free.
    Each trial runs one batched bucket inside its own Ray worker process."""
    try:
        from ray import tune
        from ray.tune.schedulers import ASHAScheduler
    except ImportError as e:
        raise ImportError("tune_run needs Ray Tune: pip install 'ray[tune]'") from e

    def trial(config):
        m = run(config)
        tune.report({metric: m[metric], "mean_out_rate": m["mean_out_rate"]})

    return tune.run(
        trial,
        config=search_space,
        num_samples=num_samples,
        scheduler=ASHAScheduler(metric=metric, mode=mode),
        max_concurrent_trials=max_concurrent,
        verbose=0,
    )
