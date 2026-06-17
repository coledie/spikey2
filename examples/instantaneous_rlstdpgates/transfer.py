"""
Instantaneous neuron models -- the Laplace / steady-state core.

WHY THIS EXISTS
---------------
The spiking engine in `examples/` advances every neuron through an inner loop of
`processing_time` (P) micro-steps per trial, re-injecting the previous step's
spikes and re-running reward-modulated STDP over a `stdp_window`. That is an
O(P * window) Monte-Carlo simulation *per trial* whose only purpose is to
estimate two numbers: how often the neuron fires, and how its inputs co-fire
with it. This module computes those two numbers in CLOSED FORM, so a trial costs
O(1) and the network behaves like an ordinary feed-forward layer.

THE LAPLACE IDEA
----------------
A leaky integrator is a first-order linear filter:

        tau * dv/dt = -v + I(t)          <=>        V(s) = H(s) I(s),   H(s) = 1/(tau s + 1)

The discrete leaky neuron used by the spiking engine,

        v_{t+1} = (1 - d) v_t + I,                                    (d = potential_decay)

is the same system sampled in time; its transfer function is
H(z) = 1 / (1 - (1 - d) z^{-1}). The DC gain (s -> 0, i.e. z -> 1) is H(0) = 1/d,
so a *constant* drive I settles at v* = I / d. Solving the recurrence from reset
(v = 0) gives the exact membrane trajectory and hence the exact time to the first
threshold crossing -- a closed-form firing rate (the "f-I curve"). No stepping.

THE CENTRAL-LIMIT IDEA
----------------------
The drive itself is a sum of many independent rate-coded (Bernoulli) inputs, so
per step it is approximately Gaussian with mean and variance we can write down
directly (`input_moments`). Propagating just those two moments through each
neuron's transfer function reproduces the trial-averaged firing rate that the
spiking loop estimates by brute force -- this is exactly what a rate model /
mean-field reduction of an SNN does.

CONTRACT
--------
Every function is vectorised and batched on a leading experiment axis B and a
trailing neuron axis N; the only Python loops anywhere in examples2 are over
*trials* (the learning timescale) and over experiment seeds -- never over the P
micro-steps that this module dissolves into algebra.
"""
from __future__ import annotations
import numpy as np


# --------------------------------------------------------------------------- #
# Normal CDF (Abramowitz & Stegun 7.1.26 erf approximation; pure numpy, ~1e-7).
# Keeps examples2 numpy-only -- no scipy dependency, just like the core repo.
# --------------------------------------------------------------------------- #
def _erf(x):
    x = np.asarray(x, dtype=np.float64)
    sign = np.sign(x)
    z = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * z)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                - 0.284496736) * t + 0.254829592) * t * np.exp(-z * z)
    return sign * y


def phi(x):
    """Standard normal CDF, Phi(x) = P(Z <= x)."""
    return 0.5 * (1.0 + _erf(np.asarray(x, dtype=np.float64) / np.sqrt(2.0)))


# --------------------------------------------------------------------------- #
# Moment propagation: many Bernoulli inputs -> (mean, var) of the summed drive.
# --------------------------------------------------------------------------- #
def input_moments(rates, W, gain):
    """First two moments of the per-step input current I = gain * sum_i s_i w_i,
    where input i fires independently with probability rates_i each step.

    rates : [..., S]        per-input fire probability (expected rate)
    W     : [..., S, N]     synaptic weights
    gain  : scalar or [...] input_gain
    returns mean [..., N], var [..., N]
    """
    rates = np.asarray(rates, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    g = np.asarray(gain, dtype=np.float64)
    mean = g[..., None] * np.einsum("...s,...sn->...n", rates, W)
    var = (g[..., None] ** 2) * np.einsum("...s,...sn->...n",
                                          rates * (1.0 - rates), W ** 2)
    return mean, var


# --------------------------------------------------------------------------- #
# Closed-form firing rates (spikes per step in [0, 1]).
# --------------------------------------------------------------------------- #
def lif_rate(mean, decay, threshold, resting=0.0):
    """Exact firing rate of the discrete leaky integrate-and-fire neuron
    v_{t+1} = (1-decay) v_t + I (reset to `resting` on crossing `threshold`),
    driven by a constant current `mean`.

    From v_t = (I/decay)(1 - (1-decay)^t) the neuron can fire only if its
    steady-state potential v* = I/decay reaches threshold, i.e. I > threshold*decay
    (the rheobase). The first crossing happens at

        t* = ln(1 - threshold*decay/I) / ln(1 - decay)

    and the rate is 1/t* spikes/step (capped at 1). This is the Laplace/DC-gain
    result H(0)=1/decay made exact for the sampled system -- it replaces the inner
    P-step integration loop with one evaluation.
    """
    I = np.asarray(mean, dtype=np.float64)
    rheobase = threshold * decay
    fireable = I > rheobase
    safe_I = np.where(fireable, I, np.inf)            # avoid divide-by-zero warnings
    arg = np.clip(1.0 - rheobase / safe_I, 1e-12, 1.0)
    tstar = np.log(arg) / np.log(1.0 - decay)
    rate = np.where(fireable, 1.0 / np.maximum(tstar, 1.0), 0.0)
    return np.clip(rate, 0.0, 1.0)


def band_rate(mean, var, lo, hi):
    """Firing rate of the non-monotonic dCaAP (dendritic) unit: it fires on a step
    iff the drive lands inside the band [lo, hi]. With the drive ~ N(mean, var)
    (central-limit over the rate-coded inputs), the expected per-step firing rate
    is the analytic band-occupancy probability

        P(lo <= I <= hi) = Phi((hi-mean)/sd) - Phi((lo-mean)/sd).

    This is the smooth, instantaneous version of the engine's memoryless band
    indicator: no input means drive ~ 0 (below band, silent), one input lands the
    drive in the band (fires), two inputs overshoot it (suppressed) -- XOR.
    """
    sd = np.sqrt(np.asarray(var, dtype=np.float64)) + 1e-9
    m = np.asarray(mean, dtype=np.float64)
    return np.clip(phi((hi - m) / sd) - phi((lo - m) / sd), 0.0, 1.0)


def izhi_rate(mean, rheobase, slope, fmax=1.0):
    """Type-I (square-root) f-I curve for the regular-spiking Izhikevich neuron.

    The subthreshold a,b,c,d model linearises to a second-order system whose
    spiking onset is a saddle-node-on-invariant-circle bifurcation, giving the
    classic continuous f ~ sqrt(I - I_rheobase) onset (Izhikevich 2007, ch.7).
    `rheobase` and `slope` are calibrated once against the actual neuron by
    `calibrate_izhi` -- after that, evaluating the rate is O(1) with no stepping.
    """
    I = np.asarray(mean, dtype=np.float64)
    over = np.maximum(0.0, I - rheobase)
    return np.clip(slope * np.sqrt(over), 0.0, fmax)


# Gauss-Hermite nodes/weights for E_{I~N(mu,var)}[g(I)] (the diffusion term).
_GH_X, _GH_W = np.polynomial.hermite.hermgauss(9)


def expected_rate(rate_fn, mean, var):
    """Noise-aware mean-field rate: E[ rate_fn(I) ] for I ~ N(mean, var), the
    per-step drive being approximately Gaussian by the CLT over many inputs.

    This is the diffusion correction to a bare f-I curve. It matters whenever the
    mean drive sits near the rheobase: the neuron still fires on the upper tail of
    the input fluctuations even though the *mean* is sub-threshold -- which is
    exactly the stochastic exploration the spiking engine got from sampled spikes.
    Computed by 9-node Gauss-Hermite quadrature, so it stays O(1) -- no stepping.
    """
    mean = np.asarray(mean, dtype=np.float64)
    sd = np.sqrt(np.maximum(np.asarray(var, dtype=np.float64), 0.0))
    grid = mean[..., None] + np.sqrt(2.0) * sd[..., None] * _GH_X
    vals = rate_fn(grid)
    return (vals * _GH_W).sum(-1) / np.sqrt(np.pi)


def realize(rate, P, rng):
    """Turn an analytic per-step firing rate into a trial-averaged *realised* rate
    with the sampling variance the spiking engine would have produced over P
    micro-steps -- but in one Gaussian draw, not P Bernoulli steps.

    The count of spikes in P steps is ~ Binomial(P, rate); by the CLT the mean
    rate is ~ N(rate, rate(1-rate)/P). This single draw supplies the exploration
    noise that reward-modulated learning needs (the spiking engine got it from
    stochastic spikes), while staying O(1) per trial.
    """
    rate = np.asarray(rate, dtype=np.float64)
    sd = np.sqrt(np.maximum(rate * (1.0 - rate), 0.0) / max(P, 1))
    return np.clip(rate + rng.normal(0.0, 1.0, size=rate.shape) * sd, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# One-time calibration of the Izhikevich f-I curve against the real neuron.
# This is the ONLY place a time loop appears in examples2, and it runs once at
# setup purely to fit two constants; the learning path never steps in time.
# --------------------------------------------------------------------------- #
def _izhi_step(v, u, I, a, b, c, d):
    for _ in range(2):
        v = v + 0.5 * (0.04 * v * v + 5 * v + 140 - u + I)
    u = u + a * (b * v - u)
    fired = v >= 30.0
    v = np.where(fired, c, v)
    u = np.where(fired, u + d, u)
    return v, u, fired


def calibrate_izhi(p, currents=None, steps=400):
    """Measure the real Izhikevich f-I curve over a current sweep and fit the
    type-I sqrt law f = slope * sqrt(I - rheobase). Returns (rheobase, slope).

    Run once; the fitted constants then drive `izhi_rate` instantaneously.
    """
    a, b, c, d = p["izhi_a"], p["izhi_b"], p["izhi_c"], p["izhi_d"]
    if currents is None:
        currents = np.linspace(0.0, 20.0, 41)
    rates = []
    for I in currents:
        v = np.array([c], dtype=np.float64)
        u = np.array([b * c], dtype=np.float64)
        n = 0
        for _ in range(steps):
            v, u, fired = _izhi_step(v, u, I, a, b, c, d)
            n += int(fired[0])
        rates.append(n / steps)
    rates = np.array(rates)
    firing = rates > 0
    if firing.sum() < 2:
        return float(currents[-1]), 0.0
    rheobase = float(currents[firing][0])
    over = currents[firing] - rheobase
    # least-squares slope through origin for f = slope*sqrt(over)
    x = np.sqrt(np.maximum(over, 0.0))
    y = rates[firing]
    slope = float((x @ y) / max(x @ x, 1e-12))
    return rheobase, slope
