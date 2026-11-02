#!/usr/bin/env python3
"""
Fast diagnostics on the gating experiment's bias (mean|a*-a|=0.626), using
the CURRENT checkpoint - no retraining needed, runs in seconds, while a
training chunk runs in the background. Two cheap hypotheses:

  H1: search under-optimized (pop=32, 3 gens was too small/short) - fix is
      free (just run inverse_query harder), no retraining needed.
  H2: the residual (a*-a) has STRUCTURE (a consistent scale/sign/offset
      per action dimension) rather than being unstructured noise - would
      point at a fixable normalization/VAE-reconstruction-bias issue
      rather than "the manifold genuinely doesn't know the action".
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from mw_framework import MWDetector
from transition_domain import FEATURES, STATE_FEATURES, ACTION_FEATURES, NEXT_FEATURES

det = MWDetector.load("transition_mw_walker.ckpt")
held = pd.read_parquet("transitions_heldout.parquet")
N = 500
rows = held.sample(N, random_state=42).reset_index(drop=True)
S = rows[STATE_FEATURES].to_numpy()
SP = rows[NEXT_FEATURES].to_numpy()
A_true = rows[ACTION_FEATURES].to_numpy()


def inverse_query(S, SP, pop, gens, seed0=0):
    n = len(S)
    n_elite = max(1, int(pop * 0.2))
    mean = np.zeros((n, 6))
    std = np.ones((n, 6)) * 0.5
    for gen in range(gens):
        cand = mean[:, None, :] + std[:, None, :] * np.random.default_rng(seed0 + gen).standard_normal((n, pop, 6))
        cand = np.clip(cand, -1.0, 1.0)
        big_s = np.repeat(S, pop, axis=0)
        big_sp = np.repeat(SP, pop, axis=0)
        big_a = cand.reshape(n * pop, 6)
        big_df = pd.DataFrame(np.concatenate([big_s, big_a, big_sp], axis=1),
                              columns=STATE_FEATURES + ACTION_FEATURES + NEXT_FEATURES)
        dists = det.mw_distance(big_df[FEATURES]).reshape(n, pop)
        elite_idx = np.argsort(dists, axis=1)[:, :n_elite]
        elite = np.take_along_axis(cand, elite_idx[:, :, None], axis=1)
        mean = elite.mean(axis=1)
        std = elite.std(axis=1) + 1e-3
    return mean


print("H1: does more CEM search budget shrink the bias (search problem) "
      "or not (manifold problem)?")
for pop, gens in [(32, 3), (64, 6), (128, 10)]:
    t0 = time.time()
    a_star = inverse_query(S, SP, pop, gens)
    err = np.abs(a_star - A_true)
    print(f"  pop={pop:>4d} gens={gens:>2d}: mean|a*-a|={err.mean():.4f}  "
          f"std={err.std():.4f}  ({time.time()-t0:.1f}s, N={N})")

print("\nH2: does the residual have structure (scale/sign bias per action dim) "
      "or look like unstructured noise? Using the pop=128 result above:")
resid = a_star - A_true  # signed residual, not abs
for i, name in enumerate(ACTION_FEATURES):
    r, atrue = resid[:, i], A_true[:, i]
    corr = np.corrcoef(atrue, a_star[:, i])[0, 1]
    print(f"  {name:<16s} mean_signed_resid={r.mean():+.3f}  "
          f"std_true_a={atrue.std():.3f}  std_a*={a_star[:,i].std():.3f}  "
          f"corr(a*, a_true)={corr:+.3f}")
