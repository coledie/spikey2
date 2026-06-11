"""
Copy-paste starting point for validating a model. Fill the three TODO regions:
the dynamics, the independent oracle, and the property tests. Then run:

    python validate_template.py

The structure -- pure function, independent oracle, golden test over random
inputs, analytic properties, ground-truth print -- is the whole point. Don't skip
the oracle; it is the rung that catches the most bugs.
"""
from __future__ import annotations
import numpy as np


# --------------------------------------------------------------------------- #
# 1. THE MODEL (pure function). Quote the source equations in the docstring.
# --------------------------------------------------------------------------- #
def model(x, params):
    """TODO: implement the dynamics here, vectorized / fast.

    Source equations (quote them):
        ...
    """
    raise NotImplementedError


# --------------------------------------------------------------------------- #
# 2. THE ORACLE (independent, obviously-correct, slow). Write from the equations,
#    NOT by copying `model` -- a shared bug would pass the golden test.
# --------------------------------------------------------------------------- #
def model_ref(x, params):
    """TODO: explicit-loop transcription of the same equations."""
    raise NotImplementedError


# --------------------------------------------------------------------------- #
# 3. RANDOM CASE GENERATOR for the golden test.
# --------------------------------------------------------------------------- #
def random_case(rng):
    """TODO: return (x, params) drawn randomly, covering edge cases."""
    raise NotImplementedError


# --------------------------------------------------------------------------- #
# Validation harness -- usually does not need editing.
# --------------------------------------------------------------------------- #
def test_golden(trials: int = 200):
    rng = np.random.default_rng(0)
    for _ in range(trials):
        x, params = random_case(rng)
        fast = model(x, params)
        ref = model_ref(x, params)
        assert np.allclose(fast, ref), "model != oracle (golden test failed)"


def test_properties():
    """TODO: assert the qualitative facts that DEFINE the phenomenon, each with a
    message that reads like the fact. Examples:

        assert delta_pre_before_post > 0, "pre-before-post should potentiate"
        assert model(zero_input, p) == 0,  "zero input -> zero change"
        assert isi[-1] > isi[0],           "should show adaptation"
    """
    raise NotImplementedError


def ground_truth():
    """TODO: run the reference once, print the numbers you will assert against
    (final value, peak, counts). Capture these BEFORE trusting the model."""
    raise NotImplementedError


def main():
    print("validating model:")
    test_golden();      print("  ok  golden: model matches independent oracle")
    test_properties();  print("  ok  properties: defining facts hold")
    print("  -- ground truth --")
    ground_truth()


if __name__ == "__main__":
    main()
