#!/usr/bin/env python3
"""
Small, honest test of a gradient-free, CPU-only policy search: Cross-Entropy
Method (CEM) over a small linear policy, instead of PPO/backprop over a
neural network. This is the rigorous version of "keep a population of
candidate policies, find/refit toward the best ones, repeat" - the same
spirit as SVI approximating a posterior with a finite particle set instead
of exact MCMC (NUTS/HMC).

Policy: action = tanh(W @ features + b), a LINEAR map, not a neural net.
Features: pitch, pitch_rate, height, 6 joint qpos, 6 joint qvel = 15 dims.
Action: 6 joint torques (Walker2d-v5).
Parameter vector theta = W.flatten() + b = 6*15 + 6 = 96 numbers.

CEM loop: sample a population of theta vectors from a Gaussian, evaluate
each by rollout (steps survived before falling = fitness), keep the top
elite_frac, refit the Gaussian's mean/std to the elites, resample. Repeat
for `generations` rounds. No gradients, no backprop, no GPU - every
rollout is an independent CPU-bound simulation, trivially parallelizable
across cores (not done here, kept serial for a first honest check).

Comparison points:
  - untrained random policy (no search at all)
  - this CEM-found policy
  - the tuned PD controller from tune_balance.py: 263/1000 steps (~2.1s)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import time
import numpy as np
import gymnasium as gym

ACT_DIM = 6
FEAT_DIM = 15
DIM = ACT_DIM * FEAT_DIM + ACT_DIM


def get_features(data):
    return np.concatenate([
        [data.qpos[2], data.qvel[2], data.qpos[1]],  # pitch, pitch_rate, height
        data.qpos[3:9],
        data.qvel[3:9],
    ])


def unpack(theta):
    W = theta[:ACT_DIM * FEAT_DIM].reshape(ACT_DIM, FEAT_DIM)
    b = theta[ACT_DIM * FEAT_DIM:]
    return W, b


def policy_action(theta, feat):
    W, b = unpack(theta)
    return np.tanh(W @ feat + b)


def rollout(theta, seed, max_steps):
    env = gym.make("Walker2d-v5", exclude_current_positions_from_observation=False)
    obs, info = env.reset(seed=seed)
    data = env.unwrapped.data
    steps = 0
    for t in range(max_steps):
        feat = get_features(data)
        act = policy_action(theta, feat)
        obs, r, term, trunc, info = env.step(act)
        steps += 1
        if term:
            break
    env.close()
    return steps


def fitness(theta, seeds, max_steps):
    return np.mean([rollout(theta, s, max_steps) for s in seeds])


def cem(pop_size=60, elite_frac=0.2, generations=15, max_steps=400,
        eval_seeds=(0, 1), seed=0):
    rng = np.random.default_rng(seed)
    mean = np.zeros(DIM)
    std = np.ones(DIM) * 0.5
    n_elite = max(1, int(pop_size * elite_frac))
    for gen in range(generations):
        pop = mean + std * rng.standard_normal((pop_size, DIM))
        fits = np.array([fitness(theta, eval_seeds, max_steps) for theta in pop])
        elite_idx = np.argsort(fits)[-n_elite:]
        elite = pop[elite_idx]
        mean = elite.mean(axis=0)
        std = elite.std(axis=0) + 1e-3
        print(f"  gen {gen:2d}  pop_mean={fits.mean():6.1f}  pop_best={fits.max():6.1f}  "
              f"std_norm={np.linalg.norm(std):.3f}")
    return mean


if __name__ == "__main__":
    t0 = time.time()
    print("Running CEM: pop=60, generations=15, 2 seeds/candidate, cap 400 steps/rollout...")
    best_theta = cem()
    train_time = time.time() - t0
    print(f"\nCEM search finished in {train_time:.1f}s (CPU, serial, no GPU, no backprop)\n")

    held_out_seeds = list(range(100, 110))
    cem_perf = fitness(best_theta, held_out_seeds, max_steps=1000)
    print(f"CEM policy, held-out seeds (10), cap 1000 steps: {cem_perf:.1f} steps mean")

    rng = np.random.default_rng(999)
    random_theta = rng.standard_normal(DIM) * 0.5
    random_perf = fitness(random_theta, held_out_seeds, max_steps=1000)
    print(f"untrained random policy, same held-out seeds:     {random_perf:.1f} steps mean")

    print(f"\nFor reference (from tune_balance.py, same env, PD controller): 263/1000 steps")
