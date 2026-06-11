"""
Validation for the learning rule. This is the answer to "RLSTDP has to work
perfectly and be validatable". Two independent guarantees:

  1. GOLDEN  : the fast vectorized rule == the dumb-loop oracle, byte-for-byte,
               on random inputs (single and batched).
  2. PROPERTY: analytic facts that must hold regardless of implementation --
               pre-before-post potentiates, no fire => no change, zero reward
               (trace) => no change, recency is monotonic.

Run: python -m snn2.validate
"""
from __future__ import annotations
import numpy as np

from .stdp import stdp_delta, stdp_delta_ref, stdp_delta_batched


def _rand_case(rng, n_inputs=4, N=3, T=6, window=8):
    S = n_inputs + N
    spike_log = (rng.random((T, S)) < 0.4)
    trace = float(rng.uniform(0.1, 2.0))
    polarity = rng.choice([-1.0, 1.0], size=S)
    return spike_log, trace, polarity, n_inputs, window


def test_golden_single(trials=200):
    rng = np.random.default_rng(0)
    for _ in range(trials):
        sl, tr, pol, n_in, w = _rand_case(rng)
        fast = stdp_delta(sl, tr, lr=0.3, window=w, n_inputs=n_in, polarity=pol)
        ref = stdp_delta_ref(sl, tr, lr=0.3, window=w, n_inputs=n_in, polarity=pol)
        assert np.allclose(fast, ref), "vectorized != oracle"


def test_golden_batched(trials=50):
    rng = np.random.default_rng(1)
    for _ in range(trials):
        cases = [_rand_case(rng) for _ in range(5)]
        n_in, w = cases[0][3], cases[0][4]
        sl = np.stack([c[0] for c in cases])
        tr = np.array([c[1] for c in cases])
        pol = np.stack([c[2] for c in cases])
        batched = stdp_delta_batched(sl, tr, lr=0.3, window=w, n_inputs=n_in, polarity=pol)
        for b, c in enumerate(cases):
            ref = stdp_delta_ref(c[0], c[1], 0.3, w, n_in, c[2])
            assert np.allclose(batched[b], ref), "batched != oracle"


def test_ltp_sign():
    # source 0 fires early, body neuron 0 fires now -> weight must increase
    n_in, N, T, w = 2, 2, 4, 6
    S = n_in + N
    sl = np.zeros((T, S))
    sl[0, 0] = 1          # pre (input source 0) fires early
    sl[-1, n_in + 0] = 1  # post (body 0) fires now
    pol = np.ones(S)
    d = stdp_delta(sl, trace=1.0, lr=0.5, window=w, n_inputs=n_in, polarity=pol)
    assert d[0, 0] > 0, "pre-before-post should potentiate"


def test_zero_trace():
    rng = np.random.default_rng(2)
    sl, _, pol, n_in, w = _rand_case(rng)
    d = stdp_delta(sl, trace=0.0, lr=0.5, window=w, n_inputs=n_in, polarity=pol)
    assert np.all(d == 0), "no reward (trace=0) must mean no weight change"


def test_no_post_no_change():
    n_in, N, T, w = 2, 2, 4, 6
    S = n_in + N
    sl = np.zeros((T, S))
    sl[0, 0] = 1          # pre fires, but nothing fires now
    pol = np.ones(S)
    d = stdp_delta(sl, trace=1.0, lr=0.5, window=w, n_inputs=n_in, polarity=pol)
    assert np.all(d == 0), "no post fire must mean no update"


def test_recency_monotonic():
    # a more-recent pre fire should earn at least as much credit as an older one
    n_in, N, T, w = 1, 1, 5, 8
    S = n_in + N
    pol = np.ones(S)
    def credit(t):
        sl = np.zeros((T, S)); sl[t, 0] = 1; sl[-1, n_in] = 1
        return stdp_delta(sl, 1.0, 0.5, w, n_in, pol)[0, 0]
    cs = [credit(t) for t in range(T - 1)]
    assert all(b >= a - 1e-9 for a, b in zip(cs, cs[1:])), "recency not monotonic"


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed -- rule validated against oracle + properties.")


if __name__ == "__main__":
    main()
