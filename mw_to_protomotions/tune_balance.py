#!/usr/bin/env python3
"""Isolate standing balance only (reference = straight legs, no gait),
sweep the balance feedback sign/gain, see which keeps it upright longest."""
import numpy as np
import gymnasium as gym

def run(balance_kp, balance_kd, sign, n_steps=500, push_at=-1, push=0.0):
    env = gym.make("Walker2d-v5", exclude_current_positions_from_observation=False)
    obs, info = env.reset(seed=0)
    data = env.unwrapped.data
    kp = np.array([4.0, 4.0, 2.0, 4.0, 4.0, 2.0])
    kd = np.array([0.5, 0.5, 0.2, 0.5, 0.5, 0.2])
    ref0 = np.array([-0.3, -0.3, 0.0, -0.3, -0.3, 0.0])  # slightly bent standing posture
    for step in range(n_steps):
        qpos, qvel = data.qpos, data.qvel
        pitch, pitch_rate = qpos[2], qvel[2]
        bal = sign * (balance_kp * pitch + balance_kd * pitch_rate)
        ref = ref0.copy()
        ref[0] += bal; ref[3] += bal
        torque = kp * (ref - qpos[3:9]) - kd * qvel[3:9]
        torque = np.clip(torque, -1, 1)
        if step == push_at:
            data.qvel[0] += push
        obs, r, term, trunc, info = env.step(torque)
        if term:
            env.close()
            return step
    env.close()
    return n_steps

for sign in (+1, -1):
    for kp_b in (0.0, 1.0, 3.0, 6.0, 10.0):
        for kd_b in (0.0, 1.0, 2.0):
            steps = run(kp_b, kd_b, sign)
            if steps >= 490:
                print(f"sign={sign:+d} kp={kp_b:5.1f} kd={kd_b:4.1f} -> survived {steps}")
