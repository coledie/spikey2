"""
The repo's Izhikevich task (examples/izhikevich2007.ipynb), redesigned.

Same experiment -- random-state env, rate-coded input, reward-modulated LTP --
but expressed as DATA, run BATCHED, with a TRUE Izhikevich neuron instead of
the LIF stand-in the original notebook used. No template class, no train_func,
no Ray required to run it locally.

    python examples/izhikevich.py
"""
import time
import numpy as np
import snn2

# 1) One experiment. The whole spec is three meaningful lines; the preset owns
#    everything else (neuron=izhikevich, input=ratemap, synapse=ltp, reward set,
#    sizes, processing_time, a/b/c/d, input gain, ...).
spec = {"preset": "izhi_randstate", "lr": 0.1, "stdp_window": 100}
m = snn2.run(spec)
print("single experiment")
print("  reward=%.3f  out_rate=%.3f  |W|=%.1f" %
      (m["final_reward"], m["mean_out_rate"], m["weight_norm"]))

# 2) A sweep is just data: cartesian product of deltas over the preset.
specs = snn2.sweep({"preset": "izhi_randstate"},
                   {"lr": [0.0, 0.05, 0.1, 0.3], "stdp_window": [50, 100, 200]})

# 3) Logically-staggered / uneven runs compose for free -- give each lane its
#    own episode budget; the engine's active-mask handles the rest.
rng = np.random.default_rng(0)
for s in specs:
    s["len_episode"] = int(60 + 120 * rng.random())

# 4) One call runs all 12 in a single batched bucket (same shape -> one tensor).
t = time.time()
results = snn2.schedule(specs)            # local; swap for snn2.schedule_ray for multi-core
dt = time.time() - t
print("\nscheduled %d experiments, %d distinct results, %.1fs"
      % (len(specs), len(results), dt))

best = max(results.values(), key=lambda r: r["final_reward"])
print("best: lr=%s window=%s reward=%.3f"
      % (best["spec"]["lr"], best["spec"]["stdp_window"], best["final_reward"]))

# 5) Scaling out is one symbol change. With Ray installed:
#       results = snn2.schedule_ray(specs)        # bucket-per-process, own GIL each
#       analysis = snn2.tune_run({                # async ASHA search: staggered + early-stop
#           "preset": "izhi_randstate",
#           "lr": tune.loguniform(1e-2, 5e-1),
#           "stdp_window": tune.choice([50, 100, 200]),
#       }, num_samples=500)
