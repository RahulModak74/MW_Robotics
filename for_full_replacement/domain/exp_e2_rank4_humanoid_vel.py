#!/usr/bin/env python3
"""
E2 - Rank-4 humanoid with root linear velocities added to features.
Everything else identical to rank4_humanoid.py.
"""
import sys, os, time
import numpy as np
import gymnasium as gym

ACT_DIM = 17
FEAT_DIM = 45  # was 42; +3 for qvel[0:3] (vx, vy, vz)
RANK = 4
DIM = RANK * FEAT_DIM + ACT_DIM * RANK + ACT_DIM  # 265


def get_features(data):
    return np.concatenate([
        [data.qpos[2]],
        data.qpos[3:7],
        data.qvel[0:3],    # NEW: root linear velocities
        data.qpos[7:24],
        data.qvel[3:6],
        data.qvel[6:23],
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
    return 0.4 * np.tanh(U @ z + b)


def rollout(theta, env, seed, max_steps):
    obs, info = env.reset(seed=seed)
    data = env.unwrapped.data
    steps = 0
    for _ in range(max_steps):
        feat = get_features(data)
        act = policy_action(theta, feat)
        obs, r, term, trunc, info = env.step(act)
        steps += 1
        if term:
            break
    return steps


def fitness(theta, env, seeds, max_steps):
    return np.mean([rollout(theta, env, s, max_steps) for s in seeds])


def cem(pop_size=80, generations=30, max_steps=400, n_eval_seeds=3, seed=0):
    rng = np.random.default_rng(seed)
    env = gym.make("Humanoid-v5")
    mean = np.zeros(DIM); std = np.ones(DIM) * 0.3
    n_elite = max(1, int(pop_size * 0.2))
    for gen in range(generations):
        pop = mean + std * rng.standard_normal((pop_size, DIM))
        gen_seeds = rng.integers(0, 100_000, size=n_eval_seeds).tolist()
        fits = np.array([fitness(t, env, gen_seeds, max_steps) for t in pop])
        elite = pop[np.argsort(fits)[-n_elite:]]
        mean, std = elite.mean(0), elite.std(0) + 1e-3
        print(f"  gen {gen:2d}  pop_mean={fits.mean():7.1f}  "
              f"pop_best={fits.max():7.1f}  std_norm={np.linalg.norm(std):.3f}",
              flush=True)
    env.close()
    return mean


if __name__ == "__main__":
    gens = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    print(f"E2 - rank-4 + velocity features on Humanoid-v5")
    print(f"  DIM={DIM} (was 253 without velocities)")
    print(f"  budget: pop=80, gens={gens}, max_steps=400")
    print(f"  baseline: rank4_humanoid.py @ 30 gen = 99.0, @ 90 gen = 112.7")
    print()

    t0 = time.time()
    best_theta = cem(generations=gens)
    print(f"\nCEM finished in {time.time()-t0:.1f}s")

    env = gym.make("Humanoid-v5")
    held_out = fitness(best_theta, env, list(range(100, 120)), max_steps=1000)
    env.close()
    print(f"\nHeld-out mean (20 seeds, cap 1000): {held_out:.1f} steps")
    print(f"  rank4 baseline (30 gens, no vel): 99.0")
    print(f"  rank4 baseline (90 gens, no vel): 112.7")
    print(f"  random baseline: 26.1")

    np.save("exp_e2_rank4_humanoid_vel_theta.npy", best_theta)
