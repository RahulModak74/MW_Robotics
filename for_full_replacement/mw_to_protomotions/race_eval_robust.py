#!/usr/bin/env python3
"""
Don't retrain - re-evaluate. The race in race_demo.py judged each racer on
exactly ONE held-out seed, which is the same statistical trap this whole
session has been diagnosing elsewhere (a single rollout is noisy). Before
spending more compute on a retrain, check whether the winning policy's
1.38m result was representative or a lucky/unlucky single trial, with a
properly averaged held-out evaluation.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import gymnasium as gym
from race_demo import get_features, policy_action

theta = np.load("race_demo_best_theta.npy")
held_out_seeds = list(range(2000, 2020))  # 20 fresh seeds, disjoint from training/race


def pure_rollout(theta, seed, max_steps):
    env = gym.make("Walker2d-v5", exclude_current_positions_from_observation=False)
    obs, info = env.reset(seed=seed)
    data = env.unwrapped.data
    x0 = data.qpos[0]
    x_final, steps = x0, 0
    for t in range(max_steps):
        feat = get_features(data, t)
        act = policy_action(theta, feat)
        obs, r, term, trunc, info = env.step(act)
        x_final = data.qpos[0]
        steps += 1
        if term:
            break
    env.close()
    return x_final - x0, steps


dists, survs = [], []
for s in held_out_seeds:
    d, st = pure_rollout(theta, s, max_steps=600)
    dists.append(d)
    survs.append(st)
dists, survs = np.array(dists), np.array(survs)

print(f"Winning policy, {len(held_out_seeds)} fresh held-out seeds, cap 600 steps "
      f"(pure net displacement, no survival term mixed in):")
print(f"  mean distance  = {dists.mean():.2f} m   median = {np.median(dists):.2f} m")
print(f"  min = {dists.min():.2f} m   max = {dists.max():.2f} m")
print(f"  mean steps survived = {survs.mean():.1f} / 600")
print(f"  individual distances: {[round(d,2) for d in dists]}")
