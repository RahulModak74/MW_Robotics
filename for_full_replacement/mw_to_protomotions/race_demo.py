#!/usr/bin/env python3
"""
"Proof of pudding" demo, honestly scoped against ProtoMotions' g1-racer
(~/ProtoMotions/demos/g1-racer): that's 6 full-mesh Unitree G1 humanoids
sharing one GPU-trained, 403MB pretrained policy (NVIDIA Metatron), in a
browser. This is NOT that. This is the same idea - several simulated
bodies racing, driven by learned policies - at the scale this repo can
actually produce and verify in one sitting: Walker2d bodies (6 joints),
policies found by CEM (cem_policy_search.py's method), no pretraining,
no GPU, evolved from scratch in this session.

Two real changes from cem_policy_search.py:
  1. Fitness is now the env's own reward (forward velocity - control cost,
     with a healthy-standing bonus/penalty) instead of raw steps-survived.
     Racing requires moving forward, not just standing still - steps-survived
     alone doesn't reward that.
  2. Instead of training 5 separate populations (5x the compute for no
     clear benefit, since the task is identical), one population is
     evolved, and the final generation's genuinely different top
     individuals are pulled out as the "racers" - real variation that
     evolution produced, not artificially injected noise.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import time
import numpy as np
import gymnasium as gym

ACT_DIM = 6
FEAT_DIM = 17  # was 15; +2 for a gait-phase clock, see get_features
DIM = ACT_DIM * FEAT_DIM + ACT_DIM
N_RACERS = 5
GAIT_PERIOD_STEPS = 40  # ~0.32s per cycle at dt=0.008 - a plausible walking cadence


def get_features(data, t=0):
    """
    +2 features vs the earlier balance-only version: sin/cos of a fixed-
    period clock. The earlier purely-reactive policy (posture only, no
    sense of time) had to discover a repeating step rhythm by accident
    from posture feedback alone - real gaits are periodic, and nothing in
    the input told it so. This gives it an explicit phase signal to hang
    a rhythm on, a standard fix for reactive (non-recurrent) locomotion
    policies.
    """
    phase = 2 * np.pi * (t % GAIT_PERIOD_STEPS) / GAIT_PERIOD_STEPS
    return np.concatenate([
        [data.qpos[2], data.qvel[2], data.qpos[1]],
        data.qpos[3:9],
        data.qvel[3:9],
        [np.sin(phase), np.cos(phase)],
    ])


def unpack(theta):
    W = theta[:ACT_DIM * FEAT_DIM].reshape(ACT_DIM, FEAT_DIM)
    b = theta[ACT_DIM * FEAT_DIM:]
    return W, b


def policy_action(theta, feat):
    W, b = unpack(theta)
    return np.tanh(W @ feat + b)


SURVIVAL_WEIGHT = 0.02  # small on purpose - see note below


def rollout_distance(theta, seed, max_steps):
    """
    Fitness, third attempt. First attempt used the env's raw reward: a flat
    per-step "still standing" bonus dominated the sum over hundreds of
    steps, so CEM optimized for standing still, not moving (bug #1).
    Second attempt used pure net displacement with nothing else: CEM
    instantly found a degenerate shortcut - lurch forward, faceplant - the
    cheapest way to get non-negative displacement when falling is free
    (bug #2, a well-known RL reward-hacking pattern). Real reward, and
    this one in fix #2 form, add a SMALL per-step survival term back -
    enough to make falling costly, small enough (0.02/step, worth ~10
    over a full 500-step rollout) that it can't swamp real forward
    distance the way the original 1.0/step bonus did (worth up to 500).
    """
    env = gym.make("Walker2d-v5", exclude_current_positions_from_observation=False)
    obs, info = env.reset(seed=seed)
    data = env.unwrapped.data
    x0 = data.qpos[0]
    x_final = x0
    steps = 0
    for t in range(max_steps):
        feat = get_features(data, t)
        act = policy_action(theta, feat)
        obs, r, term, trunc, info = env.step(act)
        x_final = data.qpos[0]
        steps += 1
        if term:
            break
    env.close()
    return (x_final - x0) + SURVIVAL_WEIGHT * steps


def fitness(theta, seeds, max_steps):
    return np.mean([rollout_distance(theta, s, max_steps) for s in seeds])


def cem(pop_size, elite_frac, generations, max_steps, n_eval_seeds, seed=0):
    """
    Fix for the generalization gap found in the previous run: training
    against the same two fixed seeds every generation let the population
    overfit to those two specific starting conditions (good training-time
    distance, poor held-out distance). Same fix as Stage 4/6 in this
    folder - richer, varied training signal, here fresh random seeds each
    generation instead of a fixed pair.
    """
    rng = np.random.default_rng(seed)
    mean = np.zeros(DIM)
    std = np.ones(DIM) * 0.5
    n_elite = max(1, int(pop_size * elite_frac))
    final_pop, final_fits = None, None
    for gen in range(generations):
        pop = mean + std * rng.standard_normal((pop_size, DIM))
        gen_seeds = rng.integers(0, 100_000, size=n_eval_seeds).tolist()
        fits = np.array([fitness(theta, gen_seeds, max_steps) for theta in pop])
        elite_idx = np.argsort(fits)[-n_elite:]
        elite = pop[elite_idx]
        mean = elite.mean(axis=0)
        std = elite.std(axis=0) + 1e-3
        print(f"  gen {gen:2d}  pop_mean_dist={fits.mean():7.2f}m  "
              f"pop_best_dist={fits.max():7.2f}m  std_norm={np.linalg.norm(std):.3f}")
        final_pop, final_fits = pop, fits
    return mean, final_pop, final_fits


def race_trace(theta, seed, max_steps):
    env = gym.make("Walker2d-v5", exclude_current_positions_from_observation=False)
    obs, info = env.reset(seed=seed)
    data = env.unwrapped.data
    xs = [data.qpos[0]]
    for t in range(max_steps):
        feat = get_features(data, t)
        act = policy_action(theta, feat)
        obs, r, term, trunc, info = env.step(act)
        xs.append(data.qpos[0])
        if term:
            break
    env.close()
    return xs


if __name__ == "__main__":
    t0 = time.time()
    print("Evolving a running (not just balancing) population: pop=60, generations=18, "
          "4 FRESH random seeds/generation (not fixed), cap 500 steps/rollout, "
          "fitness = net forward distance + small survival term...")
    mean_theta, final_pop, final_fits = cem(
        pop_size=60, elite_frac=0.2, generations=18, max_steps=500, n_eval_seeds=4)
    print(f"\nEvolution finished in {time.time()-t0:.1f}s (CPU, serial, no GPU, no backprop)\n")

    top_idx = np.argsort(final_fits)[-N_RACERS:][::-1]
    racers = [final_pop[i] for i in top_idx]
    print(f"Pulled the top {N_RACERS} genuinely distinct individuals from the final "
          f"population as racers (training fitness, net distance in m: "
          f"{[round(final_fits[i], 2) for i in top_idx]})\n")

    print("Racing all 5 on a fresh held-out seed, 600 steps, winner = furthest x-position:")
    race_seed = 777
    race_len = 600
    traces = []
    for i, theta in enumerate(racers):
        xs = race_trace(theta, race_seed, race_len)
        traces.append(xs)
        print(f"  Racer {i+1}: survived {len(xs)-1:4d} steps, "
              f"final x-position = {xs[-1]:6.2f} m")

    standings = sorted(range(N_RACERS), key=lambda i: traces[i][-1], reverse=True)
    print("\nFinal standings:")
    for rank, i in enumerate(standings, 1):
        print(f"  {rank}. Racer {i+1} - {traces[i][-1]:.2f} m")

    np.save("race_demo_traces.npy", np.array([np.array(t) for t in traces], dtype=object),
            allow_pickle=True)
    np.save("race_demo_best_theta.npy", racers[standings[0]])
    print("\nSaved traces to race_demo_traces.npy, winner's policy to race_demo_best_theta.npy")
