#!/usr/bin/env python3
"""
Stage 4, honestly scoped: path_to_full_Robotics.md's Stage 4 asks for
training on real human motion-capture data (AMASS-style). AMASS itself
needs registration/download this environment doesn't have access to.
Substituting the nearest HONEST analog available: the same "many
independently-varied units, densely sampled" richer-datagen recipe
already proven to help the validator role for the bouncing leg
(FORECASTER.md Part 3, 15.6% -> 0.5% false-alarm rate) - but varying
PHYSICS PARAMETERS (floor friction, joint damping), not human motion
styles. This tests diversity-of-conditions, not diversity-of-skill -
a real, different, smaller question than what AMASS would test, and
documented as a substitution here, not claimed as the real thing.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import gymnasium as gym
import mujoco

FEATURES = ["height", "pitch", "pitch_rate", "thigh", "leg", "foot",
           "thigh_left", "leg_left", "foot_left"]


def collect_richer(n_units=40, max_steps=250, friction_range=(0.4, 1.3),
                   damping_range=(0.05, 0.35), seed0=100):
    """One 'unit' = one random (friction, damping) combination, run once,
    same PD controller as datagen.py. Returns rows + the (friction,
    damping) used, so held-out combos can be split off for testing."""
    kp = np.array([4.0, 4.0, 2.0, 4.0, 4.0, 2.0])
    kd = np.array([0.5, 0.5, 0.2, 0.5, 0.5, 0.2])
    rng = np.random.default_rng(seed0)
    rows = []
    for unit in range(n_units):
        friction = rng.uniform(*friction_range)
        damping = rng.uniform(*damping_range)
        env = gym.make("Walker2d-v5", exclude_current_positions_from_observation=False)
        obs, info = env.reset(seed=seed0 + unit)
        m, data = env.unwrapped.model, env.unwrapped.data
        m.geom_friction[0, 0] = friction            # floor
        m.dof_damping[3:9] = damping                # all 6 actuated joints
        for step in range(max_steps):
            pitch, pitch_rate = data.qpos[2], data.qvel[2]
            bal = -1 * (6.0 * pitch + 1.0 * pitch_rate)
            ref = np.zeros(6); ref[2] = bal; ref[5] = bal
            torque = np.clip(kp * (ref - data.qpos[3:9]) - kd * data.qvel[3:9], -1, 1)
            rows.append({
                "height": data.qpos[1], "pitch": data.qpos[2], "pitch_rate": data.qvel[2],
                "thigh": data.qpos[3], "leg": data.qpos[4], "foot": data.qpos[5],
                "thigh_left": data.qpos[6], "leg_left": data.qpos[7], "foot_left": data.qpos[8],
                "friction": friction, "damping": damping, "unit": unit, "t": step,
            })
            obs, r, term, trunc, info = env.step(torque)
            if term:
                break
        env.close()
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = collect_richer()
    print(df.groupby("unit")[["friction", "damping"]].first().describe())
    print(f"total rows: {len(df)}")
    df.to_csv("walker_richer.csv", index=False)
