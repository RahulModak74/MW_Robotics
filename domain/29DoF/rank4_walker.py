#!/usr/bin/env python3
"""
REALISTIC_CLAIMS_NEXTSTEPS_NOV2nd.md's own load-bearing experiment, run
before any 29-DoF attempt: does a rank-4 head match the proven 96-param
linear policy on Walker2d? "If rank-4 fails on 6 DoF, the scaling
hypothesis is already broken and no amount of G1 work will fix it."

Same everything as mw_to_protomotions/cem_policy_search.py (features,
fitness=survival steps, CEM mechanics) and the SAME pop/gens/max_steps
that produced the proven 920.9-step linear-policy result
(domain/PLAN_A_PASSED.md: pop=80, gens=30, max_steps=400) - only the
policy parameterisation changes:

  linear:  a = tanh(W @ phi(s) + b),        W: (6,15)          -> 96 params
  rank-4:  a = tanh(U @ (V @ phi(s)) + b),  V: (4,15), U: (6,4) -> 90 params

Direct, apples-to-apples comparison, not a re-estimate.
"""
import sys, os, time
import numpy as np
import gymnasium as gym

ACT_DIM = 6
FEAT_DIM = 15
RANK = 4
DIM = RANK * FEAT_DIM + ACT_DIM * RANK + ACT_DIM  # 60 + 24 + 6 = 90


def get_features(data):
    return np.concatenate([
        [data.qpos[2], data.qvel[2], data.qpos[1]],
        data.qpos[3:9],
        data.qvel[3:9],
    ])


def unpack(theta):
    i = RANK * FEAT_DIM
    j = i + ACT_DIM * RANK
    V = theta[:i].reshape(RANK, FEAT_DIM)
    U = theta[i:j].reshape(ACT_DIM, RANK)
    b = theta[j:]
    return V, U, b


def policy_action(theta, feat):
    V, U, b = unpack(theta)
    z = V @ feat
    return np.tanh(U @ z + b)


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


def cem(pop_size=80, elite_frac=0.2, generations=30, max_steps=400,
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
              f"std_norm={np.linalg.norm(std):.3f}", flush=True)
    return mean


if __name__ == "__main__":
    print(f"Rank-4 head on Walker2d: DIM={DIM} params (vs linear's 96), "
          f"pop=80, gens=30, max_steps=400 - identical budget to the proven "
          f"linear-policy run (920.9 steps held-out)")
    t0 = time.time()
    best_theta = cem()
    train_time = time.time() - t0
    print(f"\nCEM search finished in {train_time:.1f}s", flush=True)

    held_out_seeds = list(range(100, 120))
    rank4_perf = fitness(best_theta, held_out_seeds, max_steps=1000)
    print(f"rank-4 policy, held-out seeds (20), cap 1000 steps: {rank4_perf:.1f} steps mean")
    print(f"linear-policy reference (same held-out protocol, PLAN_A_PASSED.md): 920.9 steps")

    rng = np.random.default_rng(999)
    random_theta = rng.standard_normal(DIM) * 0.5
    random_perf = fitness(random_theta, held_out_seeds, max_steps=1000)
    print(f"untrained random policy, same held-out seeds: {random_perf:.1f} steps")
    np.save("rank4_walker_theta.npy", best_theta)
