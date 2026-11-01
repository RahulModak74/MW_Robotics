#!/usr/bin/env python3
"""
Stage 3 retest, open-loop: the closed-loop test (stage3_reference_test.py)
found naive interpolation tracked slightly better, diagnosed as likely
caused by the balance correction washing out reference-path differences
since both paths share an endpoint. This retest removes that correction
entirely - pure feedforward PD tracking of the reference joints, no live
balance term - to isolate what the reference PATH itself contributes,
the way FORECASTER.md's open-loop test actually isolates it.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import torch
import gymnasium as gym
import mujoco
from mw_framework import MWDetector
from walker_domain import WalkerPostureVAE, FEATURES
from datagen import collect

JOINTS = ["thigh", "leg", "foot", "thigh_left", "leg_left", "foot_left"]


def to_latent(det, row):
    x = np.array([[row[f] for f in FEATURES]])
    xn = det.norm_.transform(x)
    xt = torch.tensor(xn, dtype=torch.float32)
    with torch.no_grad():
        mu, _ = det.vae_.encode(xt)
    return mu[0]


def mw_path(det, start_row, end_row, n_points):
    z0, z1 = to_latent(det, start_row), to_latent(det, end_row)
    tt = torch.linspace(0, 1, n_points)[:, None]
    path = (1 - tt) * z0 + tt * z1
    with torch.no_grad():
        phys = det.vae_.to_physical(det.vae_.decode(path))
    return np.stack([phys[j].numpy() for j in JOINTS], axis=1)


def naive_path(start_row, end_row, n_points):
    s = np.array([start_row[j] for j in JOINTS])
    e = np.array([end_row[j] for j in JOINTS])
    tt = np.linspace(0, 1, n_points)[:, None]
    return (1 - tt) * s + tt * e


def track_openloop(start_qpos, start_qvel, ref_path):
    """NO balance term at all - pure feedforward PD tracking of the given
    joint-angle reference path, so the comparison isolates the path
    itself, not the live safety net."""
    env = gym.make("Walker2d-v5", exclude_current_positions_from_observation=False)
    env.reset(seed=0)
    m, d = env.unwrapped.model, env.unwrapped.data
    d.qpos[:] = start_qpos
    d.qvel[:] = start_qvel
    mujoco.mj_forward(m, d)
    kp = np.array([4.0, 4.0, 2.0, 4.0, 4.0, 2.0])
    kd = np.array([0.5, 0.5, 0.2, 0.5, 0.5, 0.2])
    fell = False
    for step in range(len(ref_path)):
        ref = ref_path[step]
        torque = np.clip(kp * (ref - d.qpos[3:9]) - kd * d.qvel[3:9], -1, 1)
        obs, r, term, trunc, info = env.step(torque)
        if term:
            fell = True
            break
    final_qpos = d.qpos.copy()
    env.close()
    return final_qpos, fell, step


def main(epochs):
    print("Training WalkerPostureVAE (same as Stage 1/3)...")
    df = collect(n_episodes=60, max_steps=300)
    normal = df[df.label == "normal"].reset_index(drop=True)
    det = MWDetector(vae=WalkerPostureVAE(), epochs=epochs, lr=2e-3).fit(normal[FEATURES])

    n_points = 20   # shorter hop - open loop with no balance term falls fast on long hops
    results = []
    for ep_id in df.episode.unique()[:15]:
        ep_df = df[df.episode == ep_id].reset_index(drop=True)
        if len(ep_df) < n_points + 5:
            continue
        i0, i1 = 0, n_points
        start_row, end_row = ep_df.iloc[i0], ep_df.iloc[i1]
        qpos0 = np.array([0.0, start_row.height, start_row.pitch,
                          start_row.thigh, start_row.leg, start_row.foot,
                          start_row.thigh_left, start_row.leg_left, start_row.foot_left])
        qvel0 = np.zeros(9); qvel0[2] = start_row.pitch_rate
        target_joints = np.array([end_row[j] for j in JOINTS])

        mw_ref = mw_path(det, start_row, end_row, n_points)
        naive_ref = naive_path(start_row, end_row, n_points)
        mw_final, mw_fell, mw_steps = track_openloop(qpos0, qvel0, mw_ref)
        nv_final, nv_fell, nv_steps = track_openloop(qpos0, qvel0, naive_ref)
        mw_err = np.abs(mw_final[3:9] - target_joints).mean()
        nv_err = np.abs(nv_final[3:9] - target_joints).mean()
        results.append(dict(ep=ep_id, mw_fell=mw_fell, nv_fell=nv_fell,
                            mw_steps=mw_steps, nv_steps=nv_steps, mw_err=mw_err, nv_err=nv_err))

    res = pd.DataFrame(results)
    print(res)
    print(f"\nOpen-loop, no balance term:")
    print(f"MW path:    fell {res.mw_fell.sum()}/{len(res)}, mean steps {res.mw_steps.mean():.1f}, "
         f"mean final joint error {res.mw_err.mean():.4f}")
    print(f"Naive path: fell {res.nv_fell.sum()}/{len(res)}, mean steps {res.nv_steps.mean():.1f}, "
         f"mean final joint error {res.nv_err.mean():.4f}")


if __name__ == "__main__":
    ep = 40 if "--quick" in sys.argv else 200
    main(ep)
