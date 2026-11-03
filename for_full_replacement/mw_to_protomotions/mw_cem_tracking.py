#!/usr/bin/env python3
"""
The gating experiment from SPEC_with_MW.md section 7.3, adapted honestly to
what this repo actually has (no AMASS mocap, no pretrained ProtoMotions
encoder, no G1/H1-2 robot - Walker2d, same as the rest of this folder).

This directly targets the failure found in race_demo.py: CEM on a scalar
reward (survival, or raw displacement) kept finding cheap shortcuts
(stand still / faceplant forward) because a scalar reward is weakly
informative. The spec's fix is to score against a REFERENCE MOTION
instead - a trajectory that must be tracked at every timestep, which
can't be gamed by standing still (deviates from a moving target) or by
falling (deviates massively, instantly detected).

Three pieces already exist in this repo and are reused, not rebuilt:
  1. Reference motion: classical_controller.py's WalkGaitController.reference()
     - a hand-authored KINEMATIC gait (hip/knee/ankle vs. phase). It failed
     as a direct dynamic controller (fell in <1s), but that's irrelevant
     here - it's being reused only as the shape of a target trajectory to
     track, exactly DeepMimic/ProtoMotions' own paradigm (a reference the
     dynamics-aware policy must learn to realize, not invent from scratch).
  2. MW validity layer: walker_domain.py's WalkerPostureVAE + MWDetector,
     already trained and proven (Stage 1: 1.000 ROC-AUC on injected faults).
     Reused directly via its mw_distance() method - no new manifold built.
  3. Richer/randomized training seeds per generation - the Stage 4/6/
     race_demo fix for overfitting to a fixed seed set.

New in this script: the low-rank (rank-4) CEM policy head and the short-
rollout (0.5s) scoring loop, per spec sections 3.1-3.2.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import numpy as np
import pandas as pd
import gymnasium as gym
from mw_framework import MWDetector
from walker_domain import WalkerPostureVAE, FEATURES
from datagen import collect
from classical_controller import WalkGaitController

RANK = 4
FEAT_DIM = 17  # pitch, pitch_rate, height, 6 qpos, 6 qvel, sin/cos phase
ACT_DIM = 6
DIM = RANK * FEAT_DIM + ACT_DIM * RANK + ACT_DIM  # 98 params

STRIDE_PERIOD = 1.0
DT = 0.008
GAIT = WalkGaitController(stride_period=STRIDE_PERIOD, dt=DT)


def get_features(data, t):
    phase = 2 * np.pi * ((t * DT / STRIDE_PERIOD) % 1.0)
    return np.concatenate([
        [data.qpos[2], data.qvel[2], data.qpos[1]],
        data.qpos[3:9],
        data.qvel[3:9],
        [np.sin(phase), np.cos(phase)],
    ])


def reference_angles(t):
    phase = (t * DT / STRIDE_PERIOD) % 1.0
    return GAIT.reference(phase)


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


FALL_PENALTY = 5.0  # far worse than any real in-window tracking error (~0.1-0.3)


def rollout(theta, env, seed, horizon_steps, det, collect_rows=True):
    """
    Runs the policy for horizon_steps (or until it falls). Returns mean
    per-step tracking error (rad^2), mean MW distance, steps survived,
    net forward distance.

    Bug found and fixed here: the first version averaged tracking error
    only over however many steps the episode actually survived, with NO
    penalty for falling early. That let a policy that fell after 5 steps
    with low error over those 5 steps score BETTER than one surviving
    longer and accumulating more real error - CEM was quietly rewarding
    immediate collapse, the same reward-hacking pattern race_demo.py's
    SURVIVAL_WEIGHT fix already addressed once, reintroduced here in a
    new form. Fix: average over the FULL horizon regardless of when it
    fell, filling the steps after a fall with a fixed large penalty.
    """
    obs, info = env.reset(seed=seed)
    data = env.unwrapped.data
    x0 = data.qpos[0]
    track_errs, rows = [], []
    steps = 0
    for t in range(horizon_steps):
        feat = get_features(data, t)
        act = policy_action(theta, feat)
        obs, r, term, trunc, info = env.step(act)
        ref = reference_angles(t)
        track_errs.append(np.mean((data.qpos[3:9] - ref) ** 2))
        if collect_rows:
            rows.append({
                "height": data.qpos[1], "pitch": data.qpos[2], "pitch_rate": data.qvel[2],
                "thigh": data.qpos[3], "leg": data.qpos[4], "foot": data.qpos[5],
                "thigh_left": data.qpos[6], "leg_left": data.qpos[7], "foot_left": data.qpos[8],
            })
        steps += 1
        if term:
            break
    n_missing = horizon_steps - steps
    mean_track_err = (sum(track_errs) + n_missing * FALL_PENALTY) / horizon_steps
    if collect_rows and rows:
        mean_mw_dist = float(det.mw_distance(pd.DataFrame(rows)[FEATURES]).mean())
    else:
        mean_mw_dist = FALL_PENALTY
    return mean_track_err, mean_mw_dist, steps, data.qpos[0] - x0


def fitness(theta, env, seeds, horizon_steps, det, w_track=1.0, lam=0.1):
    terrs, mwds = [], []
    for s in seeds:
        te, mwd, _, _ = rollout(theta, env, s, horizon_steps, det)
        terrs.append(te)
        mwds.append(mwd)
    mean_te, mean_mwd = float(np.mean(terrs)), float(np.mean(mwds))
    score = -(w_track * mean_te + lam * mean_mwd)
    return score, mean_te, mean_mwd


def cem(env, det, pop_size, n_eval_seeds, generations, horizon_steps, seed=0):
    rng = np.random.default_rng(seed)
    mean = np.zeros(DIM)
    std = np.ones(DIM) * 0.5
    n_elite = max(1, int(pop_size * 0.2))
    final_pop, final_scores = None, None
    for gen in range(generations):
        pop = mean + std * rng.standard_normal((pop_size, DIM))
        gen_seeds = rng.integers(0, 100_000, size=n_eval_seeds).tolist()
        results = [fitness(theta, env, gen_seeds, horizon_steps, det) for theta in pop]
        scores = np.array([r[0] for r in results])
        tes = np.array([r[1] for r in results])
        elite_idx = np.argsort(scores)[-n_elite:]
        elite = pop[elite_idx]
        mean = elite.mean(axis=0)
        std = elite.std(axis=0) + 1e-3
        best = np.argmax(scores)
        print(f"  gen {gen:2d}  pop_mean_score={scores.mean():7.3f}  "
              f"best_score={scores.max():7.3f}  best_track_err={tes[best]:.4f}rad^2  "
              f"std_norm={np.linalg.norm(std):.3f}")
        final_pop, final_scores = pop, scores
    best_idx = np.argmax(final_scores)
    return final_pop[best_idx], final_pop, final_scores


if __name__ == "__main__":
    t0 = time.time()
    DET_CACHE = "walker_mw_detector.ckpt"
    if os.path.exists(DET_CACHE):
        print(f"Step 1/3: loading cached MW validity layer from {DET_CACHE}...")
        det = MWDetector.load(DET_CACHE)
    else:
        print("Step 1/3: fitting the MW validity layer on real PD-controller trajectories "
              "(reusing walker_domain.py + datagen.py, same recipe as Stage 1)...")
        df = collect(n_episodes=60, max_steps=300)
        normal = df[df.label == "normal"].reset_index(drop=True)
        det = MWDetector(vae=WalkerPostureVAE(), epochs=150, lr=2e-3).fit(normal[FEATURES])
        det.save(DET_CACHE)
    print(f"  done in {time.time()-t0:.1f}s\n")

    print("Step 2/3: CEM search at the spec's OWN gating scale (section 7.3: '~10^4 "
          "short rollouts' to decide viability) - pop=64, 3 seeds/gen (randomized), "
          "60 generations = 11,520 rollouts, horizon=140 steps (~1.12s, now matched "
          "to the gait's 125-step period), fall-penalty fix applied. Scored by "
          "tracking_error + 0.1*MW_distance (spec section 3.2)...")
    t1 = time.time()
    env = gym.make("Walker2d-v5", exclude_current_positions_from_observation=False)
    horizon_steps = 140
    best_theta, final_pop, final_scores = cem(
        env, det, pop_size=64, n_eval_seeds=3, generations=60, horizon_steps=horizon_steps)
    search_time = time.time() - t1
    n_rollouts = 64 * 3 * 60
    print(f"  search finished in {search_time:.1f}s, {n_rollouts} short rollouts, "
          f"CPU, no GPU, no backprop\n")

    print("Step 3/3: full-episode evaluation of the winner on 20 fresh held-out seeds "
          "(600 steps each, NOT the 0.5s short rollouts used during search)...")
    tes, mwds, survs, dists = [], [], [], []
    for s in range(3000, 3020):
        te, mwd, steps, dist = rollout(best_theta, env, s, 600, det)
        tes.append(te); mwds.append(mwd); survs.append(steps); dists.append(dist)
    tes, mwds, survs, dists = map(np.array, (tes, mwds, survs, dists))
    env.close()

    print(f"\n  mean tracking error      = {tes.mean():.4f} rad^2  "
          f"(RMSE ~{np.sqrt(tes.mean())*180/np.pi:.1f} deg/joint)")
    print(f"  mean MW distance          = {mwds.mean():.3f}")
    print(f"  fall rate (<600 steps)    = {(survs < 600).mean():.1%}")
    print(f"  mean steps survived       = {survs.mean():.1f} / 600")
    print(f"  mean net forward distance = {dists.mean():.2f} m (not directly optimized - informational)")
    print(f"\nTotal wall-clock: {time.time()-t0:.1f}s")
