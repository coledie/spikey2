"""Engine variants, each registered under a string name via the registry.

Importing this package registers every engine, so a spec can name one
(`engine="batched"` or `engine="trial"`) and the scheduler looks it up -- the
same indirection the registry already gives neurons, inputs, and rewards.
"""
from . import batched, trial  # noqa: F401  (import registers the engines)
