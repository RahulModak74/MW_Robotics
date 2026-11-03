#!/usr/bin/env python3
"""
Adjudicates a real critique (from ~/MW_Robotics/domain/EXPLANATION_OF_WALKER_FOR_AGENT.md):
"the joint form has no incentive to use the action" - i.e. because all
training transitions came from ONE fixed PD controller, a = f(s)
nearly deterministically, so inverse_query might recover a* just by
learning that s->a mapping, never actually using s' at all.

Test: for the same held-out s, run inverse_query twice - once against the
REAL s' (as before), once against a RANDOM OTHER row's s' (shuffled,
breaking the true correspondence but still a physically plausible s').
If the model genuinely uses s':
    a*_real should correlate with true a; a*_shuffled should NOT (wrong
    target, should recover something different/worse).
If the critique is right (model ignores s', just learned f(s)):
    a*_real and a*_shuffled should look nearly IDENTICAL to each other
    (both just predicting from s), and both would show the same
    correlation with true a - because that correlation is coming from
    the s->a confound, not from using s' at all.
"""
import sys, os
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
SP_real = rows[NEXT_FEATURES].to_numpy()
A_true = rows[ACTION_FEATURES].to_numpy()

rng = np.random.default_rng(123)
shuffle_idx = rng.permutation(N)
SP_shuffled = SP_real[shuffle_idx]  # a random OTHER row's s', paired with the SAME s


def inverse_query(S, SP, pop=128, gens=10, seed0=0):
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


print("Running inverse_query against the REAL s' ...")
a_star_real = inverse_query(S, SP_real)
print("Running inverse_query against a SHUFFLED (wrong) s' ...")
a_star_shuffled = inverse_query(S, SP_shuffled)

err_real = np.abs(a_star_real - A_true).mean()
err_shuffled = np.abs(a_star_shuffled - A_true).mean()
agreement = np.abs(a_star_real - a_star_shuffled).mean()
corr_real = np.mean([np.corrcoef(A_true[:, i], a_star_real[:, i])[0, 1] for i in range(6)])
corr_shuffled = np.mean([np.corrcoef(A_true[:, i], a_star_shuffled[:, i])[0, 1] for i in range(6)])

print(f"\nmean|a*_real - a_true|      = {err_real:.4f}   avg corr(a*_real, a_true)      = {corr_real:+.3f}")
print(f"mean|a*_shuffled - a_true|  = {err_shuffled:.4f}   avg corr(a*_shuffled, a_true)  = {corr_shuffled:+.3f}")
print(f"mean|a*_real - a*_shuffled| = {agreement:.4f}  (how similar are the two recovered actions to EACH OTHER)")
print(f"\nFor reference, typical |a| magnitude / action range: mean|a_true|={np.abs(A_true).mean():.3f}, range=2.0")

if agreement < 0.5 * err_real and corr_shuffled < 0.3 * corr_real:
    verdict = "Model DOES use s' meaningfully - shuffled s' gives a clearly worse, different result."
elif agreement < err_real * 0.3:
    verdict = "AMBIGUOUS - shuffled result differs from real-s' result, needs more inspection."
else:
    verdict = ("CONFOUND CONFIRMED: a*_real and a*_shuffled are nearly as similar to each other as to "
               "the truth. The model is mostly predicting a from s alone, NOT genuinely using s'. "
               "The earlier gating 'pass' was likely inflated by the fixed-PD-controller confound.")
print(f"\nVerdict: {verdict}")
