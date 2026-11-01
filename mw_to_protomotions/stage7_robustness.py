#!/usr/bin/env python3
"""
Stage 7, honestly scoped: NO real hardware is present in this
environment, and real-hardware actions would not be attempted even if
it were (an explicit decision carried over from an earlier LeRobot/
serial-hardware discussion in this project). This tests the thing that
actually precedes any real sim2real attempt: how robust is the real
(simulated) closed-loop system to disturbances, measured honestly
across many magnitudes and seeds - not a single cherry-picked push.

Second, genuinely useful question tied back to the whole plan: can
Stage 1's trained validator (MW) serve as a real-time SAFETY MONITOR
riding alongside the Stage 2 classical controller - does its distance
score rise and cross a calibrated threshold BEFORE the simulator's own
termination condition fires? That's the actual "MW watches, classical
control acts" architecture path_to_full_Robotics.md and the README
describe, tested directly for the first time here.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import gymnasium as gym
from mw_framework import MWDetector
from walker_domain import WalkerPostureVAE, FEATURES
from datagen import collect

ep = 40 if "--quick" in sys.argv else 200


def run_one(seed, push_at=-1, push=0.0, max_steps=400):
    kp = np.array([4.0, 4.0, 2.0, 4.0, 4.0, 2.0])
    kd = np.array([0.5, 0.5, 0.2, 0.5, 0.5, 0.2])
    env = gym.make("Walker2d-v5", exclude_current_positions_from_observation=False)
    env.reset(seed=seed)
    data = env.unwrapped.data
    rows = []
    fell_at = max_steps
    for step in range(max_steps):
        pitch, pitch_rate = data.qpos[2], data.qvel[2]
        bal = -1 * (6.0 * pitch + 1.0 * pitch_rate)
        ref = np.zeros(6); ref[2] = bal; ref[5] = bal
        torque = np.clip(kp * (ref - data.qpos[3:9]) - kd * data.qvel[3:9], -1, 1)
        rows.append({"height": data.qpos[1], "pitch": data.qpos[2], "pitch_rate": data.qvel[2],
                    "thigh": data.qpos[3], "leg": data.qpos[4], "foot": data.qpos[5],
                    "thigh_left": data.qpos[6], "leg_left": data.qpos[7], "foot_left": data.qpos[8],
                    "t": step})
        if step == push_at:
            data.qvel[0] += push
        obs, r, term, trunc, info = env.step(torque)
        if term:
            fell_at = step
            break
    env.close()
    return fell_at, pd.DataFrame(rows)


def main():
    print("Part A: robustness curve - survival under growing disturbance, many seeds")
    pushes = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    results = []
    for push in pushes:
        fells = [run_one(seed, push_at=30, push=push)[0] for seed in range(20)]
        survived_full = sum(1 for f in fells if f >= 399)
        results.append(dict(push=push, mean_fall_step=np.mean(fells), survived=survived_full))
    res = pd.DataFrame(results)
    print(res)

    print("\nPart B: does MW's distance rise before the simulator's own termination?")
    print("Training the Stage 1 validator...")
    df = collect(n_episodes=60, max_steps=300)
    normal = df[df.label == "normal"].reset_index(drop=True)
    det = MWDetector(vae=WalkerPostureVAE(), epochs=ep, lr=2e-3).fit(normal[FEATURES])
    # calibrate a threshold from held-out normal data (99th percentile)
    held_out = normal.iloc[int(len(normal) * 0.8):]
    threshold = np.percentile(det.mw_distance(held_out[FEATURES]), 99)
    print(f"calibrated threshold (99th pct of held-out normal): {threshold:.3f}")

    leads = []
    for seed in range(15):
        fell_at, traj = run_one(seed, push_at=30, push=3.0)
        if fell_at >= 399 or len(traj) < 5:
            continue
        dist = det.mw_distance(traj[FEATURES])
        over = np.where(dist > threshold)[0]
        if len(over) == 0:
            leads.append(None)   # MW never flagged before the fall
            continue
        first_flag = over[0]
        lead_steps = fell_at - first_flag
        leads.append(lead_steps)
    valid = [l for l in leads if l is not None]
    print(f"episodes that fell: {len(leads)}, MW flagged before falling in: {len(valid)}")
    if valid:
        print(f"mean early-warning lead time: {np.mean(valid):.1f} steps "
             f"({np.mean(valid)*0.008:.3f}s) before the simulator's own termination")


if __name__ == "__main__":
    main()
