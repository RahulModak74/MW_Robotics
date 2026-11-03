#!/usr/bin/env python3
"""
The natural next test after rank4_walker.py's clean pass (896.2 vs 920.9,
97.3% match) and cem_humanoid.py's confirmation that the dimensionality
wall is real at 731 flat params on Humanoid-v5 (97.2/1000, barely above
random's 26.1 - the SAME budget that nearly solves Walker2d). Does a
rank-4 head recover what the flat policy lost, on Humanoid, the same
way it matched linear on Walker2d?

a = 0.4 * tanh(U @ (V @ phi(s)) + b)
V: (4, 42), U: (17, 4), b: (17,) -> 168 + 68 + 17 = 253 params
(close to REALISTIC_CLAIMS_NEXTSTEPS_NOV2nd.md's own estimate, "~265")

Same features, same fitness (survival steps), same budget (pop=80,
gens=30, max_steps=400) as cem_humanoid.py, for a clean before/after.
"""
import sys, os, time
import numpy as np
import gymnasium as gym

ACT_DIM = 17
FEAT_DIM = 42
RANK = 4
DIM = RANK * FEAT_DIM + ACT_DIM * RANK + ACT_DIM  # 253


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


def rollout(theta, env, seed, max_steps):
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
    return steps


def fitness(theta, env, seeds, max_steps):
    return np.mean([rollout(theta, env, s, max_steps) for s in seeds])


def cem(pop_size=80, generations=30, max_steps=400, n_eval_seeds=3, seed=0):
    rng = np.random.default_rng(seed)
    env = gym.make("Humanoid-v5")
    mean = np.zeros(DIM)
    std = np.ones(DIM) * 0.3
    n_elite = max(1, int(pop_size * 0.2))
    for gen in range(generations):
        pop = mean + std * rng.standard_normal((pop_size, DIM))
        gen_seeds = rng.integers(0, 100_000, size=n_eval_seeds).tolist()
        fits = np.array([fitness(theta, env, gen_seeds, max_steps) for theta in pop])
        elite_idx = np.argsort(fits)[-n_elite:]
        elite = pop[elite_idx]
        mean = elite.mean(axis=0)
        std = elite.std(axis=0) + 1e-3
        print(f"  gen {gen:2d}  pop_mean={fits.mean():7.1f}  pop_best={fits.max():7.1f}  "
              f"std_norm={np.linalg.norm(std):.3f}", flush=True)
    env.close()
    return mean


if __name__ == "__main__":
    gens = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    print(f"Rank-4 head on Humanoid-v5: DIM={DIM} params (vs flat linear's 731), "
          f"pop=80, gens={gens}, max_steps=400")
    t0 = time.time()
    best_theta = cem(generations=gens)
    print(f"\nCEM finished in {time.time()-t0:.1f}s", flush=True)

    env = gym.make("Humanoid-v5")
    held_out_perf = fitness(best_theta, env, list(range(100, 120)), max_steps=1000)
    env.close()
    print(f"rank-4 policy, held-out (20 seeds), cap 1000: {held_out_perf:.1f} steps")
    print(f"flat-linear reference (cem_humanoid.py, same budget): 97.2 steps")
    print(f"random baseline (same seeds): 26.1 steps")
    np.save("rank4_humanoid_theta.npy", best_theta)
