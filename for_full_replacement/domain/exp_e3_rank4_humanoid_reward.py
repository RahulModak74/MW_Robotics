#!/usr/bin/env python3
"""
E3 - Rank-4 humanoid with env-reward-sum fitness instead of survival steps.
Everything else identical to rank4_humanoid.py.
"""
import sys, os, time
import numpy as np
import gymnasium as gym

ACT_DIM = 17
FEAT_DIM = 42   # original features, NO velocities - E2 tested that separately
RANK = 4
DIM = RANK * FEAT_DIM + ACT_DIM * RANK + ACT_DIM   # 253


def get_features(data):
    return np.concatenate([
        [data.qpos[2]],
        data.qpos[3:7],
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


def rollout_reward(theta, env, seed, max_steps):
    obs, info = env.reset(seed=seed)
    data = env.unwrapped.data
    total_r = 0.0
    steps = 0
    for _ in range(max_steps):
        feat = get_features(data)
        act = policy_action(theta, feat)
        obs, r, term, trunc, info = env.step(act)
        total_r += float(r)
        steps += 1
        if term:
            break
    return total_r, steps


def fitness_reward(theta, env, seeds, max_steps):
    return float(np.mean([rollout_reward(theta, env, s, max_steps)[0] for s in seeds]))


def cem(pop_size=80, generations=30, max_steps=400, n_eval_seeds=3, seed=0):
    rng = np.random.default_rng(seed)
    env = gym.make("Humanoid-v5")
    mean = np.zeros(DIM); std = np.ones(DIM) * 0.3
    n_elite = max(1, int(pop_size * 0.2))
    for gen in range(generations):
        pop = mean + std * rng.standard_normal((pop_size, DIM))
        gen_seeds = rng.integers(0, 100_000, size=n_eval_seeds).tolist()
        fits = np.array([fitness_reward(t, env, gen_seeds, max_steps) for t in pop])
        elite = pop[np.argsort(fits)[-n_elite:]]
        mean, std = elite.mean(0), elite.std(0) + 1e-3
        print(f"  gen {gen:2d}  mean_reward={fits.mean():9.1f}  "
              f"best_reward={fits.max():9.1f}  std_norm={np.linalg.norm(std):.3f}",
              flush=True)
    env.close()
    return mean


if __name__ == "__main__":
    gens = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    print(f"E3 - rank-4 + env-reward-sum fitness on Humanoid-v5")
    print(f"  DIM={DIM}, budget: pop=80, gens={gens}, max_steps=400")
    print(f"  baseline: rank4_humanoid.py @ 30 gen = 99.0")
    print()

    t0 = time.time()
    best_theta = cem(generations=gens)
    print(f"\nCEM finished in {time.time()-t0:.1f}s")

    env = gym.make("Humanoid-v5")
    held_out_seeds = list(range(100, 120))
    rewards, steps = [], []
    for s in held_out_seeds:
        r, st = rollout_reward(best_theta, env, s, max_steps=1000)
        rewards.append(r); steps.append(st)
    env.close()
    print(f"\nHeld-out (20 seeds, cap 1000):")
    print(f"  mean sum_reward = {np.mean(rewards):9.1f}")
    print(f"  mean steps      = {np.mean(steps):6.1f}   (baseline 99.0)")
    print(f"  random baseline steps = 26.1")

    np.save("exp_e3_rank4_humanoid_reward_theta.npy", best_theta)
