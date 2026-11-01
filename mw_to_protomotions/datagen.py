#!/usr/bin/env python3
"""
Stage 1 data: real coupled-body trajectories from the tuned PD ankle-
balance controller (see PROGRESS_LOG.md - 263/1000 steps, 2.1s, honest
partial result), across many seeds/initial conditions, so there's a
real dataset of "normal standing" states (early in each trajectory,
before it falls) to train a Walker2d posture domain on.

FEATURES (9): height, pitch, pitch_rate, and the 6 actuated joint angles.
Excludes rootx/x-velocity (absolute position isn't physically bounded -
same choice other domains in this repo make for unconstrained
coordinates).
"""
import numpy as np
import pandas as pd
import gymnasium as gym

FEATURES = ["height", "pitch", "pitch_rate", "thigh", "leg", "foot",
           "thigh_left", "leg_left", "foot_left"]


def collect(n_episodes=60, max_steps=300, normal_frac=0.6, seed0=0):
    """Run the tuned PD balance controller from many seeds. Returns rows
    labelled 'normal' (the first normal_frac of each episode's steps,
    before any real instability) and 'near_fall' (the last portion,
    naturally organic examples of a degrading posture - not injected)."""
    kp = np.array([4.0, 4.0, 2.0, 4.0, 4.0, 2.0])
    kd = np.array([0.5, 0.5, 0.2, 0.5, 0.5, 0.2])
    bal_kp, bal_kd = 6.0, 1.0
    rows = []
    for ep in range(n_episodes):
        env = gym.make("Walker2d-v5", exclude_current_positions_from_observation=False)
        obs, info = env.reset(seed=seed0 + ep)
        data = env.unwrapped.data
        traj = []
        for step in range(max_steps):
            pitch, pitch_rate = data.qpos[2], data.qvel[2]
            bal = -1 * (bal_kp * pitch + bal_kd * pitch_rate)
            ref = np.zeros(6); ref[2] = bal; ref[5] = bal
            torque = np.clip(kp * (ref - data.qpos[3:9]) - kd * data.qvel[3:9], -1, 1)
            traj.append({
                "height": data.qpos[1], "pitch": data.qpos[2], "pitch_rate": data.qvel[2],
                "thigh": data.qpos[3], "leg": data.qpos[4], "foot": data.qpos[5],
                "thigh_left": data.qpos[6], "leg_left": data.qpos[7], "foot_left": data.qpos[8],
            })
            obs, r, term, trunc, info = env.step(torque)
            if term:
                break
        env.close()
        n = len(traj)
        for i, row in enumerate(traj):
            row["label"] = "normal" if i < n * normal_frac else "near_fall"
            row["episode"] = ep
            row["t"] = i
        rows.extend(traj)
    return pd.DataFrame(rows)


def inject_impossible(df: pd.DataFrame, n=200, seed=0) -> pd.DataFrame:
    """Hand-made, clearly impossible postures (not from physics at all) -
    the equivalent of the other domains' injected faults: a joint beyond
    its physical limit, torso at an impossible pitch/height."""
    rng = np.random.default_rng(seed)
    base = df[df.label == "normal"].sample(n, replace=True, random_state=seed).reset_index(drop=True)
    out = base.copy()
    kind = rng.choice(["joint_limit", "pitch_flip", "height_impossible"], size=n)
    for i in range(n):
        if kind[i] == "joint_limit":
            j = rng.choice(["thigh", "leg", "thigh_left", "leg_left"])
            out.loc[i, j] = rng.uniform(1.5, 3.0) * rng.choice([-1, 1])   # way past [-2.618,0]
        elif kind[i] == "pitch_flip":
            out.loc[i, "pitch"] = rng.uniform(1.5, 3.0) * rng.choice([-1, 1])
        else:
            out.loc[i, "height"] = rng.choice([rng.uniform(0.1, 0.5), rng.uniform(2.5, 4.0)])
    out["anomaly_type"] = kind
    return out


if __name__ == "__main__":
    df = collect()
    print(df.label.value_counts())
    print(df[FEATURES].describe().T[["mean", "std", "min", "max"]])
    df.to_csv("walker_trajectories.csv", index=False)
    anom = inject_impossible(df)
    anom.to_csv("walker_impossible.csv", index=False)
    print("saved walker_trajectories.csv, walker_impossible.csv")
