# Worked examples

Two complete reproductions built with the validation ladder. Each shows the same
shape: exact equations → pure function → oracle/golden test → property tests →
ground truth → figure → regime notes.

## Contents
- [Florian (2007) MSTDPET — a learning rule](#florian-2007-mstdpet)
- [Izhikevich (2003) — a neuron model](#izhikevich-2003)
- [What each rung caught](#what-each-rung-caught)

---

## Florian (2007) MSTDPET

A single synapse j→i driven with fixed pre/post spike trains and a reward that
flips +1→−1 at 100 ms. Deterministic trace simulation, so "validate" = reproduce
the exact equations and match the reference curve.

### Equations (quoted from the paper)

```
P+  <- P+ * exp(-dt/tau_+) + A+ * f_pre        (Eq 43)
P-  <- P- * exp(-dt/tau_-) + A- * f_post       (Eq 44)
zeta = P+ * f_post + P- * f_pre                 (Eq 42)
z   <- z  * exp(-dt/tau_z) + zeta               (Eq 8, eligibility trace)
w   <- w  + gamma0 * r * z                      (Eq 7, gamma0 = gamma/tau_z)
```

### Pure function

```python
def mstdpet_traces(p):
    f_pre = np.zeros(p.length); f_post = np.zeros(p.length)
    f_pre[list(p.spike_times_pre)] = 1.0
    f_post[list(p.spike_times_post)] = 1.0
    P_pre = P_post = z = 0.0; w = p.w0; r = 1.0
    out = {k: np.zeros(p.length) for k in
           ("f_pre","f_post","P_pre","P_post","zeta","z","r","w")}
    for t in range(p.length):
        fj, fi = f_pre[t], f_post[t]
        P_pre  = P_pre  * np.exp(-p.dt/p.tau_pre)  + p.A_pre  * fj
        P_post = P_post * np.exp(-p.dt/p.tau_post) + p.A_post * fi
        zeta = P_pre * fi + P_post * fj
        z = z * np.exp(-p.dt/p.tau_z) + zeta
        w = w + p.gamma0 * r * z
        for k, v in zip(("f_pre","f_post","P_pre","P_post","zeta","z","r","w"),
                        (fj, fi, P_pre, P_post, zeta, z, r, w)):
            out[k][t] = v
        if t > p.reward_flip_t:        # repo quirk: flip takes effect next step
            r = -1.0
    return out
```

### Oracle + golden test

The oracle is an **independent** inlined transcription of the same equations
(not a copy of the function above), so a shared bug can't pass both:

```python
def _reference_w(p):
    g0 = p.gamma / p.tau_z
    stp, stq = set(p.spike_times_pre), set(p.spike_times_post)
    P_pre = P_post = z = 0.0; w = p.w0; r = 1.0; wlog = []
    for t in range(p.length):
        fj = 1.0 if t in stp else 0.0
        fi = 1.0 if t in stq else 0.0
        P_pre  = P_pre  * np.exp(-1.0/p.tau_pre)  + fj
        P_post = P_post * np.exp(-1.0/p.tau_post) - fi
        zeta = P_pre * fi + P_post * fj
        z = z * np.exp(-1.0/p.tau_z) + zeta
        w = w + g0 * r * z
        wlog.append(w)
        if t > p.reward_flip_t:
            r = -1.0
    return np.array(wlog)

assert np.allclose(mstdpet_traces(p)["w"], _reference_w(p))   # golden
```

### Property tests (the facts that define MSTDPET)

```python
tr = mstdpet_traces(p)
assert tr["zeta"][10] > 0          # pre(5) before post(10) -> positive eligibility
assert tr["w"][20] > p.w0          # positive reward potentiates early
assert tr["r"][-1] == -1.0         # reward actually flips
assert tr["w"][-1] < tr["w"][102:].max()   # negative reward later reduces w
```

### Ground truth (captured from the reference, then asserted)

`w: 0.200 → 0.316112, peak 0.416076`. If your model doesn't hit these, stop and
debug before plotting.

---

## Izhikevich (2003)

The famous demonstration that one `a,b,c,d` model produces every cortical firing
type. Here "validate" = assert the behaviors that *name* each regime, driven by
the same neuron function the rest of the system uses.

### Equations + parameter table (quoted)

```
v' = 0.04 v^2 + 5 v + 140 - u + I ;  u' = a (b v - u)
if v >= 30 mV:  v <- c,  u <- u + d
RS  a=.02 b=.20 c=-65 d=8   IB  a=.02 b=.20 c=-55 d=4   CH  a=.02 b=.20 c=-50 d=2
FS  a=.10 b=.20 c=-65 d=2   LTS a=.02 b=.25 c=-65 d=2
```

### Pure function (paper's two-half-step integration)

```python
def izhikevich(state, I, p, rng):
    a, b, c, d = p["izhi_a"], p["izhi_b"], p["izhi_c"], p["izhi_d"]
    v, u = state["v"], state["u"]
    for _ in range(2):                       # two 0.5 ms Euler sub-steps for v
        v = v + 0.5 * (0.04*v*v + 5*v + 140 - u + I)
    u = u + a * (b*v - u)                     # u once per ms
    fired = v >= 30.0
    state["v"] = np.where(fired, c, v)
    state["u"] = np.where(fired, u + d, u)
    return state, fired
```

### Property tests (validation IS the behavior)

There is no single "correct" trace to golden-test against, so the analytic
properties carry the weight:

```python
counts = {k: len(spikes(k)) for k in REGIMES}
assert all(c > 0 for c in counts.values())          # all regimes fire
assert counts["FS"] == max(counts.values())         # fast spiking is fastest
assert isi("RS")[-1] > isi("RS")[0]                  # RS adapts (ISI grows)
assert isi("IB")[0] < isi("IB")[-1]                  # IB bursts then slows
assert (isi("CH") <= 5).sum() >= 3                   # CH bursts repeatedly
assert (isi("RS") <= 5).sum() == 0                   # RS does not burst
```

### Ground truth + regime note

Spike counts over 600 ms with step I=10: RS 11, IB 16, CH 24, FS 36, LTS 24.
Regime band found by sweeping `input_gain`: ≤4 silent, ≥6.5 saturated, **5.0 gives
~30% firing** — recorded in the preset so it isn't rediscovered.

---

## What each rung caught

- **Golden test** caught broadcasting/indexing mistakes in the batched STDP
  delta — the per-lane batched result had to equal the single-experiment oracle.
- **Property tests** caught a credit-assignment design flaw: a global eligibility
  trace left instrumental-conditioning accuracy at chance (0.5); switching to
  per-trial eligibility fixed it (→1.0). No golden test would have found this —
  the equation ran fine, the *learning* didn't.
- **Ground truth** caught regime problems: an Izhikevich `out_rate` of 0.92 meant
  saturation (gain too hot), invisible without logging the firing rate.
- **The naming trap** was caught by reading code instead of labels: the repo's
  "izhikevich" notebook ran an LIF neuron, not the recovery-variable model.
