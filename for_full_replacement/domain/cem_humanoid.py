#!/usr/bin/env python3
"""
Direct test of EXPLANATION_OF_WALKER_FOR_AGENT.md's Gap 4 warning ("the
dimensionality wall is real... do not attempt a 29-DoF humanoid with the
current linear policy") - but at Humanoid-v5's actual 17 DoF, a real
middle ground between Walker2d (96 params, proven) and the full 29-DoF
target the doc warns about. Same CEM recipe as run_plan_a.sh
(cem_policy_search.py), same crawl-before-walk order: survival/balance
fitness first (the easier sub-problem), same as Walker2d's own history.

Policy: a = 0.4 * tanh(W @ phi(s) + b). Humanoid-v5's actuator range is
[-0.4, 0.4], not [-1, 1] like Walker2d - scaled accordingly.

Features (42 dims): height, 4-dim root quaternion, 17 joint angles,
3-dim root angular velocity, 17 joint velocities.
"""
import sys, os, time
import numpy as np
import gymnasium as gym

ACT_DIM = 17
FEAT_DIM = 1 + 4 + 17 + 3 + 17   # 42
DIM = ACT_DIM * FEAT_DIM + ACT_DIM  # 731


def get_features(data):
    return np.concatenate([
        [data.qpos[2]],          # height
        data.qpos[3:7],          # root quaternion
        data.qpos[7:24],         # 17 joint angles
        data.qvel[3:6],          # root angular velocity
        data.qvel[6:23],         # 17 joint velocities
    ])


def unpack(theta):
    W = theta[:ACT_DIM * FEAT_DIM].reshape(ACT_DIM, FEAT_DIM)
    b = theta[ACT_DIM * FEAT_DIM:]
    return W, b


def policy_action(theta, feat):
    W, b = unpack(theta)
    return 0.4 * np.tanh(W @ feat + b)


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


def cem(pop_size, generations, max_steps, n_eval_seeds, seed=0):
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
              f"std_norm={np.linalg.norm(std):.3f}")
    env.close()
    return mean


if __name__ == "__main__":
    pop = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    gens = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    max_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 400
    print(f"CEM on Humanoid-v5: DIM={DIM} params ({DIM/96:.1f}x Walker2d's 96), "
          f"pop={pop}, gens={gens}, max_steps={max_steps}, fitness=survival steps")
    t0 = time.time()
    best_theta = cem(pop_size=pop, generations=gens, max_steps=max_steps, n_eval_seeds=3)
    print(f"\nCEM finished in {time.time()-t0:.1f}s")

    env = gym.make("Humanoid-v5")
    held_out_perf = fitness(best_theta, env, list(range(100, 120)), max_steps=1000)
    env.close()
    print(f"held-out mean survival (20 seeds, cap 1000): {held_out_perf:.1f} steps")

    rng = np.random.default_rng(999)
    random_theta = rng.standard_normal(DIM) * 0.3
    env = gym.make("Humanoid-v5")
    random_perf = fitness(random_theta, env, list(range(100, 120)), max_steps=1000)
    env.close()
    print(f"untrained random policy, same seeds: {random_perf:.1f} steps")
    np.save("cem_humanoid_theta.npy", best_theta)
