#!/usr/bin/env python3
"""
Disambiguation test #2: does a hand-designed PD balance controller (no
CEM, no learning at all) also plateau near ~100 steps on Humanoid-v5, or
does it do much better? If even a hand-tuned classical controller
struggles similarly, that confirms the bottleneck is 3D balance itself
(a task property), not something specific to CEM/rank-k search (a
method property) - the same disambiguation Walker2d's tune_balance.py
provided for the 2D case, now run on the 3D body.

Simple PD toward an upright pose, with a tilt-correction term using the
quaternion's x/y components as small-angle roll/pitch proxies (qpos[4]
~roll-ish, qpos[5]~pitch-ish for small tilts from upright [1,0,0,0]).
"""
import sys
import numpy as np
import gymnasium as gym

# joint order: abdomen_z,y,x | r_hip_x,z,y | r_knee | l_hip_x,z,y | l_knee | arms...
KP = np.array([2,2,2,  3,2,3, 3,  3,2,3, 3,  1,1,1, 1,1,1], dtype=float)
KD = np.array([0.3]*17)
TARGET = np.zeros(17)
TARGET[6] = -0.3   # right knee - slight bend
TARGET[10] = -0.3  # left knee - slight bend
BAL_KP, BAL_KD = 8.0, 1.5


def controller(data):
    qpos_j = data.qpos[7:24]
    qvel_j = data.qvel[6:23]
    roll_proxy = data.qpos[4]
    pitch_proxy = data.qpos[5]
    roll_rate = data.qvel[4]
    pitch_rate = data.qvel[5]

    target = TARGET.copy()
    pitch_corr = -BAL_KP * pitch_proxy - BAL_KD * pitch_rate
    roll_corr = -BAL_KP * roll_proxy - BAL_KD * roll_rate
    # hip_y (index 2 of each leg's 3) controls fwd/back; hip_x (index 0) controls side
    target[2] += pitch_corr   # right hip_y
    target[8] += pitch_corr   # left hip_y
    target[0] += roll_corr    # right hip_x
    target[6-6+0] = target[0]  # (no-op, keeping clarity)
    target[7] += roll_corr    # left hip_x

    torque = KP * (target - qpos_j) - KD * qvel_j
    return np.clip(torque, -0.4, 0.4)


def rollout(env, seed, max_steps):
    obs, info = env.reset(seed=seed)
    data = env.unwrapped.data
    steps = 0
    for t in range(max_steps):
        act = controller(data)
        obs, r, term, trunc, info = env.step(act)
        steps += 1
        if term:
            break
    return steps


if __name__ == "__main__":
    env = gym.make("Humanoid-v5")
    seeds = list(range(100, 120))
    results = [rollout(env, s, 1000) for s in seeds]
    env.close()
    print(f"Hand-designed PD balance controller, Humanoid-v5, 20 held-out seeds:")
    print(f"  mean = {np.mean(results):.1f}  individual = {results}")
    print(f"  for reference: random=26.1, flat-linear-CEM=97.2, rank4-CEM=112.7")
