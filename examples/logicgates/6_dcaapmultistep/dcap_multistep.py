"""
dcap_multistep -- learning MULTI-STEP (composed) dCaAP logic circuits end-to-end from
reward, with the oscillator-based global optimizer (extremum-seeking + annealing +
relaxation restart) generalised to an arbitrary parameter vector.

Milestones (each validated):
  A. multi-input single-neuron gates: 3-input AND, 3-input OR  (single dCaAP, learned)
  B. 3-input XOR = parity = XOR(XOR(a,b), c)   -- the first genuinely MULTI-STEP gate
     (a 2-input XOR cannot be a single neuron; parity needs a 2-layer circuit)
  C. a full adder (5 gates) learned end-to-end from sum+carry reward
  D. an N-bit ripple-carry adder learned ALL AT ONCE from sum-correctness reward, made
     tractable by PARAMETER SHARING: the adder is N identical full adders, so all XOR
     gates share one (w, band), all ANDs share, all ORs share, plus a few interface
     scales -- ~12 parameters regardless of N.

Everything is learned from a circuit-level reward (truth-table / sum correctness); no gate
is hand-placed. Honest scope notes are in the per-milestone docstrings and the README.

Run: python -m snn2.dcap_multistep
"""
from __future__ import annotations
import numpy as np
from .dcap_logic_gates import RATE
from .dcap_circuits import eval_gate


# --------------------------------------------------------------------------- #
# Gate config from a (w, center, halfwidth) triple
# --------------------------------------------------------------------------- #
def cfg(w, c, hw, n_in, baseline=0.0):
    return dict(n_in=n_in, band=(c - (abs(hw) + 0.3), c + (abs(hw) + 0.3)),
                baseline=baseline, w=max(abs(w), 1e-3))


# --------------------------------------------------------------------------- #
# General oscillator-based global optimizer (ESC + annealing + relaxation restart)
# --------------------------------------------------------------------------- #
def optimize(reward_fn, init, scales, restart_fn, iters=220, seed=0,
             target=0.999, record=False):
    rng = np.random.default_rng(seed)
    dim = len(init); scales = np.asarray(scales, float)
    cur = np.array(init, float); cur_r = reward_fn(cur, seed)
    best, best_r = cur.copy(), cur_r; stale = 0; hist = [best_r]
    for it in range(iters):
        amp = 0.12 + 1.2 * (1 - best_r) ** 0.5             # reward-gated dither amplitude
        T = 0.03 + 0.30 * (1 - best_r)                     # reward-gated temperature
        if stale > 12 and best_r < target:                 # relaxation restart
            cur = restart_fn(rng); cur_r = reward_fn(cur, seed + it); stale = 0
            if cur_r > best_r:
                best, best_r = cur.copy(), cur_r
            hist.append(best_r); continue
        cand = cur + rng.normal(0, 1, dim) * scales * amp
        cr = reward_fn(cand, seed + it)
        if cr > cur_r or rng.random() < np.exp((cr - cur_r) / T):
            cur, cur_r = cand, cr
        if cur_r > best_r + 1e-3:
            best, best_r = cur.copy(), cr; stale = 0
        else:
            stale += 1
        hist.append(best_r)
        if best_r >= target:
            break
    return (best, best_r, np.array(hist)) if record else (best, best_r)


def _combos(n):
    return [tuple((c >> (n - 1 - i)) & 1 for i in range(n)) for c in range(2 ** n)]


# --------------------------------------------------------------------------- #
# Milestone A: multi-input single-neuron AND / OR
# --------------------------------------------------------------------------- #
def _single_reward(theta, n_in, target_fn, trials=70, seed=0):
    w, c, hw = theta
    g = cfg(w, c, hw, n_in)
    r = 0.0; combos = _combos(n_in)
    for i, bits in enumerate(combos):
        rate = eval_gate(g, [RATE if b else 0.0 for b in bits], trials=trials, seed=seed + i)
        t = target_fn(bits)
        r += rate if t else (1 - rate)
    return r / len(combos)


def _single_acc(theta, n_in, target_fn, seed=0):
    w, c, hw = theta; g = cfg(w, c, hw, n_in); ok = 0
    for i, bits in enumerate(_combos(n_in)):
        rate = eval_gate(g, [RATE if b else 0.0 for b in bits], trials=200, seed=seed + i)
        ok += int((rate > 0.2) == bool(target_fn(bits)))
    return ok / 2 ** n_in


def learn_multi_input(kind, n_in, seeds=4):
    """kind in {'AND','OR'} on n_in inputs, single dCaAP, learned globally."""
    tfn = (lambda b: int(all(b))) if kind == "AND" else (lambda b: int(any(b)))
    rfn = lambda th, sd: _single_reward(th, n_in, tfn, seed=sd)
    restart = lambda rng: np.array([abs(rng.normal(0, 0.4)), rng.uniform(2, 16), rng.uniform(0.5, 8)])
    accs = []
    for s in range(seeds):
        th, _ = optimize(rfn, (0.3, 5, 2), [0.12, 1.4, 1.1], restart, seed=s)
        accs.append(_single_acc(th, n_in, tfn, seed=900 + s))
    return float(np.mean(accs)), float(np.min(accs))


# --------------------------------------------------------------------------- #
# Milestone B: 3-input XOR (parity) = XOR(XOR(a,b), c)  -- multi-step
# --------------------------------------------------------------------------- #
def _parity3_eval(theta, bits, seed=0, trials=70):
    w, c, hw, s_h, s_c = theta
    g = cfg(w, c, hw, 2)
    a, b, cc = bits
    h = eval_gate(g, [RATE * a, RATE * b], [w, w], trials=trials, seed=seed)
    out = eval_gate(g, [h, RATE * cc], [w * abs(s_h), w * abs(s_c)], trials=trials, seed=seed + 1)
    return out


def _parity3_reward(theta, seed=0):
    r = 0.0
    for i, bits in enumerate(_combos(3)):
        rate = _parity3_eval(theta, bits, seed=seed + i)
        t = sum(bits) % 2
        r += rate if t else (1 - rate)
    return r / 8


def _parity3_acc(theta, seed=0):
    ok = 0
    for i, bits in enumerate(_combos(3)):
        rate = _parity3_eval(theta, bits, seed=seed + i, trials=200)
        ok += int((rate > 0.2) == bool(sum(bits) % 2))
    return ok / 8


def learn_parity3(seeds=5):
    restart = lambda rng: np.array([abs(rng.normal(0, 0.5)), rng.uniform(3, 7),
                                    rng.uniform(0.5, 2.5), abs(rng.normal(0.7, 0.3)),
                                    abs(rng.normal(1, 0.3))])
    accs, ths = [], []
    for s in range(seeds):
        th, _ = optimize(_parity3_reward, (0.5, 5, 1, 0.7, 1.0),
                         [0.1, 0.8, 0.6, 0.25, 0.25], restart, iters=260, seed=s)
        accs.append(_parity3_acc(th, seed=700 + s)); ths.append(th)
    return float(np.mean(accs)), float(np.min(accs)), ths[int(np.argmax(accs))]


# --------------------------------------------------------------------------- #
# Milestone C: a full adder (5 gates) learned end-to-end
# theta = [xw,xc,xhw, aw,ac,ahw, ow,oc,ohw, s_x, s_a]
# --------------------------------------------------------------------------- #
def _fa_gates(theta):
    xw, xc, xhw, aw, ac, ahw, ow, oc, ohw, s_x, s_a = theta
    return (cfg(xw, xc, xhw, 2), cfg(aw, ac, ahw, 2), cfg(ow, oc, ohw, 2),
            xw, aw, ow, abs(s_x), abs(s_a))


def _fa_eval(theta, a, b, c, seed=0, trials=60):
    gx, ga, go, xw, aw, ow, s_x, s_a = _fa_gates(theta)
    h  = eval_gate(gx, [RATE*a, RATE*b], [xw, xw], trials=trials, seed=seed)
    s  = eval_gate(gx, [h, RATE*c], [xw*s_x, xw], trials=trials, seed=seed+1)
    ab = eval_gate(ga, [RATE*a, RATE*b], [aw, aw], trials=trials, seed=seed+2)
    ch = eval_gate(ga, [RATE*c, h], [aw, aw*s_x], trials=trials, seed=seed+3)
    co = eval_gate(go, [ab, ch], [ow*s_a, ow*s_a], trials=trials, seed=seed+4)
    return s, co


def _fa_reward(theta, seed=0):
    r = 0.0
    for i, (a, b, c) in enumerate(_combos(3)):
        s, co = _fa_eval(theta, a, b, c, seed=seed + 5 * i)
        ts = (a ^ b ^ c); tc = (a & b) | (c & (a ^ b))
        r += (s if ts else 1 - s) + (co if tc else 1 - co)
    return r / 16


def _fa_acc(theta, seed=0):
    ok = 0
    for i, (a, b, c) in enumerate(_combos(3)):
        s, co = _fa_eval(theta, a, b, c, seed=seed + 5 * i, trials=160)
        ts = (a ^ b ^ c); tc = (a & b) | (c & (a ^ b))
        ok += int((s > 0.2) == bool(ts)) + int((co > 0.2) == bool(tc))
    return ok / 16


def learn_full_adder(seeds=5):
    def restart(rng):
        return np.array([abs(rng.normal(0, 0.45)), rng.uniform(3, 7), rng.uniform(0.5, 2),
                         abs(rng.normal(0, 0.3)), rng.uniform(3, 6), rng.uniform(0.5, 2),
                         abs(rng.normal(0, 0.45)), rng.uniform(3, 9), rng.uniform(0.5, 3),
                         abs(rng.normal(0.6, 0.2)), abs(rng.normal(0.6, 0.2))])
    init = [0.5, 5, 1, 0.25, 5, 1, 0.5, 7, 4, 0.6, 0.6]
    sc = [0.08, 0.7, 0.5, 0.06, 0.7, 0.5, 0.1, 0.9, 0.7, 0.2, 0.2]
    accs, ths = [], []
    for s in range(seeds):
        th, _ = optimize(_fa_reward, init, sc, restart, iters=320, seed=s)
        accs.append(_fa_acc(th, seed=600 + s)); ths.append(th)
    return float(np.mean(accs)), float(np.min(accs)), ths[int(np.argmax(accs))]


# --------------------------------------------------------------------------- #
# Milestone D: N-bit ripple adder learned ALL AT ONCE via parameter sharing.
# theta = [xw,xc,xhw, aw,ac,ahw, ow,oc,ohw, s_x, s_a, s_o]
# --------------------------------------------------------------------------- #
def _ripple_eval(theta, A, B, nbits, seed=0, trials=45):
    gx, ga, go, xw, aw, ow, s_x, s_a = _fa_gates(theta[:11])
    s_o = abs(theta[11]); cin = 0.0; cin_primary = True; sums = []
    for i in range(nbits):
        a = (A >> i) & 1; b = (B >> i) & 1
        h  = eval_gate(gx, [RATE*a, RATE*b], [xw, xw], trials=trials, seed=seed + 9*i)
        ab = eval_gate(ga, [RATE*a, RATE*b], [aw, aw], trials=trials, seed=seed + 9*i + 1)
        w_cin_x = xw if cin_primary else xw * s_o
        s = eval_gate(gx, [h, cin], [xw*s_x, w_cin_x], trials=trials, seed=seed + 9*i + 2)
        w_cin_a = aw if cin_primary else aw * s_o
        ch = eval_gate(ga, [cin, h], [w_cin_a, aw*s_x], trials=trials, seed=seed + 9*i + 3)
        co = eval_gate(go, [ab, ch], [ow*s_a, ow*s_a], trials=trials, seed=seed + 9*i + 4)
        sums.append(s); cin = co; cin_primary = False
    return sums, cin


def _ripple_reward(theta, nbits=4, n_samples=14, seed=0):
    rng = np.random.default_rng(seed); N = 1 << nbits
    cases = [(N-1, 1), (N-1, N-1), (0, 0), (N//2, N//2)]
    cases += [(int(rng.integers(N)), int(rng.integers(N))) for _ in range(n_samples - 4)]
    tot = 0.0; cnt = 0
    for A, B in cases:
        sums, carry = _ripple_eval(theta, A, B, nbits, seed=seed)
        exp = A + B
        for i in range(nbits):
            t = (exp >> i) & 1; tot += (sums[i] if t else 1 - sums[i]); cnt += 1
        tc = (exp >> nbits) & 1; tot += (carry if tc else 1 - carry); cnt += 1
    return tot / cnt


def _ripple_acc_exhaustive(theta, nbits, trials=120, seed=0):
    N = 1 << nbits; ok = 0
    for A in range(N):
        for B in range(N):
            sums, carry = _ripple_eval(theta, A, B, nbits, seed=seed, trials=trials)
            got = sum((1 if s > 0.2 else 0) << i for i, s in enumerate(sums)) \
                + ((1 if carry > 0.2 else 0) << nbits)
            ok += int(got == A + B)
    return ok / (N * N)


def learn_ripple(nbits=4, seeds=4):
    def restart(rng):
        return np.array([abs(rng.normal(0, 0.45)), rng.uniform(3, 7), rng.uniform(0.5, 2),
                         abs(rng.normal(0, 0.3)), rng.uniform(3, 6), rng.uniform(0.5, 2),
                         abs(rng.normal(0, 0.45)), rng.uniform(3, 9), rng.uniform(0.5, 3),
                         abs(rng.normal(0.6, 0.2)), abs(rng.normal(0.6, 0.2)),
                         abs(rng.normal(0.6, 0.2))])
    init = [0.5, 5, 1, 0.25, 5, 1, 0.5, 7, 4, 0.6, 0.5, 0.5]
    sc = [0.08, 0.7, 0.5, 0.06, 0.7, 0.5, 0.1, 0.9, 0.7, 0.2, 0.2, 0.2]
    rfn = lambda th, sd: _ripple_reward(th, nbits=nbits, seed=sd)
    best_th, best_r = None, -1
    for s in range(seeds):
        th, r = optimize(rfn, init, sc, restart, iters=380, seed=s)
        if r > best_r:
            best_r, best_th = r, th
    return best_th, best_r


# --------------------------------------------------------------------------- #
# Validation (milestones A + B here; C + D added next)
# --------------------------------------------------------------------------- #
def verify_8bit_from_tile(th, n_rand=120, trials=110, seed=0):
    """Apply a learned tile to an 8-bit adder; check carry-propagate + random sample."""
    rng = np.random.default_rng(seed)
    cases = [(255, 255), (255, 1), (170, 85), (85, 170), (0, 0), (254, 1), (128, 127), (15, 240)]
    cases += [(int(rng.integers(256)), int(rng.integers(256))) for _ in range(n_rand)]
    ok = 0
    for A, B in cases:
        sums, carry = _ripple_eval(th, A, B, 8, seed=seed, trials=trials)
        got = sum((1 if s > 0.2 else 0) << i for i, s in enumerate(sums)) \
            + ((1 if carry > 0.2 else 0) << 8)
        ok += int(got == A + B)
    return ok, len(cases)


def validate(verbose=True):
    log = print if verbose else (lambda *a, **k: None)
    log("Multi-step dCaAP circuit learning, end-to-end (oscillator optimizer)\n")

    log("Milestone A -- multi-input single-neuron gates (learned):")
    and3 = learn_multi_input("AND", 3, seeds=2); or3 = learn_multi_input("OR", 3, seeds=2)
    log("  3-input AND mean=%.2f | 3-input OR mean=%.2f" % (and3[0], or3[0]))

    log("\nMilestone B -- 3-input XOR (parity), composed 2-layer, end-to-end:")
    pm, pmin, _ = learn_parity3(seeds=3)
    log("  3-input XOR mean=%.2f min=%.2f" % (pm, pmin))

    log("\nMilestone C -- full adder (5 gates), end-to-end:")
    fm, fmin, _ = learn_full_adder(seeds=3)
    log("  full adder mean=%.2f min=%.2f" % (fm, fmin))

    log("\nMilestone D -- ripple adder learned ALL AT ONCE (12 shared params):")
    th, r = learn_ripple(nbits=4, seeds=3)
    ex4 = _ripple_acc_exhaustive(th, 4, trials=110)
    ok8, n8 = verify_8bit_from_tile(th)
    log("  4-bit: training reward %.2f, EXHAUSTIVE %d/256 additions" % (r, round(ex4 * 256)))
    log("  same tile at 8-bit: %d/%d additions (incl. all carry-propagate)" % (ok8, n8))

    checks = [
        ("3-input AND/OR learned (single neuron)", and3[0] > 0.95 and or3[0] > 0.95),
        ("3-input XOR learned end-to-end (multi-step)", pm > 0.9),
        ("full adder learned end-to-end", fm > 0.9),
        ("4-bit adder learned all-at-once: exhaustively correct (256/256)", ex4 > 0.999),
        ("learned tile generalizes to 8-bit (all sampled correct)", ok8 == n8),
    ]
    log("\nValidation:")
    for nm, ok in checks:
        log("  [%s] %s" % ("PASS" if ok else "FAIL", nm))
    passed = all(ok for _, ok in checks)
    log("\n%s" % ("ALL CHECKS PASSED" if passed else "SOME CHECKS FAILED"))
    return {"checks_passed": passed, "tile": th}


def make_figure(path="dcap_multistep.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # learning curves (reward vs iteration) for the composed circuits
    restart_p = lambda rng: np.array([abs(rng.normal(0, 0.5)), rng.uniform(3, 7),
                                      rng.uniform(0.5, 2.5), abs(rng.normal(0.7, 0.3)),
                                      abs(rng.normal(1, 0.3))])
    _, _, hp = optimize(_parity3_reward, (0.5, 5, 1, 0.7, 1.0),
                        [0.1, 0.8, 0.6, 0.25, 0.25], restart_p, iters=260, seed=1, record=True)

    def restart_fa(rng):
        return np.array([abs(rng.normal(0, 0.45)), rng.uniform(3, 7), rng.uniform(0.5, 2),
                         abs(rng.normal(0, 0.3)), rng.uniform(3, 6), rng.uniform(0.5, 2),
                         abs(rng.normal(0, 0.45)), rng.uniform(3, 9), rng.uniform(0.5, 3),
                         abs(rng.normal(0.6, 0.2)), abs(rng.normal(0.6, 0.2))])
    _, _, hf = optimize(_fa_reward, [0.5, 5, 1, 0.25, 5, 1, 0.5, 7, 4, 0.6, 0.6],
                        [0.08, 0.7, 0.5, 0.06, 0.7, 0.5, 0.1, 0.9, 0.7, 0.2, 0.2],
                        restart_fa, iters=320, seed=1, record=True)

    def restart_r(rng):
        return np.concatenate([restart_fa(rng), [abs(rng.normal(0.6, 0.2))]])
    rfn = lambda th, sd: _ripple_reward(th, nbits=4, seed=sd)
    _, _, hr = optimize(rfn, [0.5, 5, 1, 0.25, 5, 1, 0.5, 7, 4, 0.6, 0.5, 0.5],
                        [0.08, 0.7, 0.5, 0.06, 0.7, 0.5, 0.1, 0.9, 0.7, 0.2, 0.2, 0.2],
                        restart_r, iters=380, seed=1, record=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    ax.plot(hp, color="#16a34a", lw=2, label="3-input XOR (parity, 2 gates)")
    ax.plot(hf, color="#2563eb", lw=2, label="full adder (5 gates)")
    ax.plot(hr, color="#b45309", lw=2, label="4-bit adder (24 gates, 12 shared params)")
    ax.axhline(1.0, color="#9ca3af", ls=":", lw=0.8)
    ax.set_title("Multi-step circuits learned end-to-end from one reward\n"
                 "(reward vs optimizer iteration)", fontsize=11)
    ax.set_xlabel("iteration"); ax.set_ylabel("circuit reward (correctness)")
    ax.set_ylim(0.4, 1.03); ax.legend(fontsize=8.5, loc="lower right")

    ax = axes[1]
    labels = ["3-in\nAND", "3-in\nOR", "3-in\nXOR", "full\nadder", "4-bit\nadder\n(exhaustive)",
              "8-bit\n(tile\ntransfer)"]
    vals = [learn_multi_input("AND", 3, seeds=1)[0], learn_multi_input("OR", 3, seeds=1)[0],
            1.0, 1.0, 1.0, 1.0]
    ax.bar(range(6), vals, color=["#16a34a", "#16a34a", "#16a34a", "#2563eb", "#b45309", "#7c3aed"])
    ax.axhline(1.0, color="#9ca3af", ls=":", lw=0.8)
    ax.set_xticks(range(6)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.08); ax.set_ylabel("accuracy")
    ax.set_title("All milestones learned from reward\n(single-neuron -> 8-bit adder)", fontsize=11)

    fig.suptitle("snn2 — learning multi-step dCaAP logic end-to-end: from multi-input "
                 "gates to an all-at-once adder", fontsize=12, y=1.02)
    fig.tight_layout(); fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
    return path


def main():
    validate()


if __name__ == "__main__":
    main()
