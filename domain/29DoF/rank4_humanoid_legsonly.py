#!/usr/bin/env python3
"""
Disambiguation test: rank4_humanoid.py plateaued around 100-170/1000 steps
even with 3x the search budget - a genuine plateau, not a budget problem.
Two possible explanations: (a) still too many joints/DoF for this search
budget, or (b) Humanoid-v5's TRUE 3D balance is just a harder problem
than Walker2d's 2D-constrained one, independent of joint count (arms
don't touch the ground and barely affect falling dynamics either way).

This test freezes the 6 arm joints (both shoulders x2, both elbows),
leaving 11 active joints (abdomen x3, hips x3 x2, knees x2) - closer to
the doc's "legs-only, no arms" intermediate. If THIS also plateaus near
the 17-DoF result, that's evidence for (b): the bottleneck is 3D balance
itself, not joint count, and a true fix needs a genuinely 2D-constrained
or more-capable architecture, not just fewer DoF.
"""
import sys, os, time
import numpy as np
import gymnasium as gym

# Humanoid-v5 joint order: abdomen_z,y,x(0-2), r_hip_x,z,y(3-5), r_knee(6),
# l_hip_x,z,y(7-9), l_knee(10), r_shoulder1,2(11-12), r_elbow(13),
# l_shoulder1,2(14-15), l_elbow(16)
ACTIVE_IDX = list(range(0, 11))   # abdomen + both legs
FROZEN_IDX = list(range(11, 17))  # both arms

ACT_DIM = len(ACTIVE_IDX)  # 11
FEAT_DIM = 1 + 4 + 17 + 3 + 17  # same full feature set (42) - sees everything,
                                 # just doesn't ACT on the arms
RANK = 4
DIM = RANK * FEAT_DIM + ACT_DIM * RANK + ACT_DIM  # 168 + 44 + 11 = 223


def get_features(data):
    return np.concatenate([
        [data.qpos[2]], data.qpos[3:7], data.qpos[7:24],
        data.qvel[3:6], data.qvel[6:23],
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
    active = 0.4 * np.tanh(U @ z + b)
    full = np.zeros(17)
    full[ACTIVE_IDX] = active
    return full  # frozen joints get 0 torque


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
    print(f"Rank-4, ARMS FROZEN (11 active DoF, DIM={DIM} params), "
          f"pop=80, gens={gens}, max_steps=400")
    t0 = time.time()
    best_theta = cem(generations=gens)
    print(f"\nCEM finished in {time.time()-t0:.1f}s", flush=True)

    env = gym.make("Humanoid-v5")
    held_out_perf = fitness(best_theta, env, list(range(100, 120)), max_steps=1000)
    env.close()
    print(f"legs-only rank-4 policy, held-out (20 seeds), cap 1000: {held_out_perf:.1f} steps")
    print(f"full-17-DoF rank-4 reference (90 gens): 112.7 steps")
    print(f"full-17-DoF flat-linear reference (30 gens): 97.2 steps")
    print(f"random baseline: 26.1 steps")
    np.save("rank4_humanoid_legsonly_theta.npy", best_theta)
