#!/usr/bin/env python3
"""
SPEC_with_MW_3.md section 8 - THE decisive gating experiment, run before
any more training or runtime code. For held-out (s, s') pairs with a known
real action, use inverse_query (1-step CEM over actions only - cheap,
per spec section 2.2) to recover a* = argmin_a d_M(s, a, s'), and compare
to the real a that actually produced that transition.

Decision rule (spec's own):
  mean error < 0.1 x action range AND std < 0.2 x action range -> VIABLE
  mean error larger, std small -> systematically biased, recoverable
  std large -> inverse is ill-posed, Plan B as specified will not work

Vectorized across BOTH rows and the CEM population in one batched
mw_distance() call per generation (not a per-row Python loop) - 5000 rows
x 32 candidates x 3 generations = 480,000 VAE forward passes, which the
VAE handles as fast batched tensor ops either way.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from mw_framework import MWDetector
from transition_domain import FEATURES, STATE_FEATURES, ACTION_FEATURES, NEXT_FEATURES

N_ROWS = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
POP = int(sys.argv[2]) if len(sys.argv) > 2 else 32
GENERATIONS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
ELITE_FRAC = 0.2

det = MWDetector.load("transition_mw_walker.ckpt")
held = pd.read_parquet("transitions_heldout.parquet")
rows = held.sample(N_ROWS, random_state=42).reset_index(drop=True)

S = rows[STATE_FEATURES].to_numpy()
SP = rows[NEXT_FEATURES].to_numpy()
A_true = rows[ACTION_FEATURES].to_numpy()
n = len(rows)
n_elite = max(1, int(POP * ELITE_FRAC))

mean = np.zeros((n, 6))
std = np.ones((n, 6)) * 0.5

t0 = time.time()
for gen in range(GENERATIONS):
    cand = mean[:, None, :] + std[:, None, :] * np.random.default_rng(gen).standard_normal((n, POP, 6))
    cand = np.clip(cand, -1.0, 1.0)

    big_s = np.repeat(S, POP, axis=0)
    big_sp = np.repeat(SP, POP, axis=0)
    big_a = cand.reshape(n * POP, 6)
    big_df = pd.DataFrame(
        np.concatenate([big_s, big_a, big_sp], axis=1),
        columns=STATE_FEATURES + ACTION_FEATURES + NEXT_FEATURES)
    dists = det.mw_distance(big_df[FEATURES]).reshape(n, POP)

    elite_idx = np.argsort(dists, axis=1)[:, :n_elite]
    elite = np.take_along_axis(cand, elite_idx[:, :, None], axis=1)
    mean = elite.mean(axis=1)
    std = elite.std(axis=1) + 1e-3
    print(f"  gen {gen}: mean dist of best candidate = {dists.min(axis=1).mean():.3f}, "
          f"mean std_norm = {np.linalg.norm(std, axis=1).mean():.3f}  "
          f"({time.time()-t0:.1f}s elapsed)")

a_star = mean
err = np.abs(a_star - A_true)  # per-dimension, per-row
action_range = 2.0  # each action in [-1, 1]

mean_err = err.mean()
std_err = err.std()
print(f"\nGating experiment result, {n} held-out (s,s') pairs:")
print(f"  mean |a* - a|  = {mean_err:.4f}  (threshold: < {0.1*action_range:.2f})")
print(f"  std  |a* - a|  = {std_err:.4f}  (threshold: < {0.2*action_range:.2f})")
print(f"  mean Euclidean ||a*-a|| per row = {np.linalg.norm(a_star - A_true, axis=1).mean():.4f} "
      f"(out of a max possible {np.sqrt(6)*2:.2f})")

if mean_err < 0.1 * action_range and std_err < 0.2 * action_range:
    verdict = "VIABLE - proceed to B1"
elif std_err < 0.2 * action_range:
    verdict = "SYSTEMATICALLY BIASED but recoverable - proceed cautiously"
else:
    verdict = "ILL-POSED - Plan B as specified will not work; fall back to Plan A"
print(f"\nSpec section 8 decision: {verdict}")

np.save("gating_a_star.npy", a_star)
np.save("gating_a_true.npy", A_true)
