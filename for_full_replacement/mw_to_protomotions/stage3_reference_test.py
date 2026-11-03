#!/usr/bin/env python3
"""
Stage 3: does MW's manifold walk - the same mechanism already proven
for the robot leg (FORECASTER.md) and for battery degradation - make a
better REFERENCE trajectory for the real PD controller to track, on
this harder, coupled body, than a naive straight-line interpolation of
raw joint angles?

Method: pick a real (current, target) pair of postures from the same
real trajectory, N steps apart. Build two reference paths between them:
 (a) MW path: encode both, walk the straight line in LATENT space,
     decode each step - the proven mechanism, extended here.
 (b) naive path: straight line in RAW joint-angle space, no manifold.
Track each with the SAME PD joint controller (plus the balance term),
starting the real simulator fresh from the current state, and measure
how closely the real resulting trajectory matches the real target, and
whether the walker stays upright while tracking it.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import torch
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
    return np.stack([phys[j].numpy() for j in JOINTS], axis=1)   # (n_points, 6)


def naive_path(start_row, end_row, n_points):
    s = np.array([start_row[j] for j in JOINTS])
    e = np.array([end_row[j] for j in JOINTS])
    tt = np.linspace(0, 1, n_points)[:, None]
    return (1 - tt) * s + tt * e


def track(start_qpos, start_qvel, ref_path, extra_steps=0):
    """Replay the real simulator from (start_qpos, start_qvel), tracking
    ref_path (one 6-vector reference per step) with the PD+balance
    controller. Returns final qpos, whether it fell, and tracking error."""
    import gymnasium as gym
    import mujoco
    env = gym.make("Walker2d-v5", exclude_current_positions_from_observation=False)
    env.reset(seed=0)
    m, d = env.unwrapped.model, env.unwrapped.data
    d.qpos[:] = start_qpos
    d.qvel[:] = start_qvel
    mujoco.mj_forward(m, d)
    kp = np.array([4.0, 4.0, 2.0, 4.0, 4.0, 2.0])
    kd = np.array([0.5, 0.5, 0.2, 0.5, 0.5, 0.2])
    fell = False
    n = len(ref_path) + extra_steps
    for step in range(n):
        ref = ref_path[min(step, len(ref_path) - 1)]
        pitch, pitch_rate = d.qpos[2], d.qvel[2]
        bal = -1 * (6.0 * pitch + 1.0 * pitch_rate)
        ref_c = ref.copy()
        ref_c[0] += bal; ref_c[3] += bal
        torque = np.clip(kp * (ref_c - d.qpos[3:9]) - kd * d.qvel[3:9], -1, 1)
        obs, r, term, trunc, info = env.step(torque)
        if term:
            fell = True
            break
    final_qpos = d.qpos.copy()
    env.close()
    return final_qpos, fell, step


def main(epochs):
    print("Collecting real trajectories + training WalkerPostureVAE...")
    df = collect(n_episodes=60, max_steps=300)
    normal = df[df.label == "normal"].reset_index(drop=True)
    det = MWDetector(vae=WalkerPostureVAE(), epochs=epochs, lr=2e-3).fit(normal[FEATURES])

    n_points = 50
    results = []
    rng = np.random.default_rng(0)
    # pick several real (start, target) pairs from different episodes,
    # separated by a real stretch of real dynamics
    for ep_id in df.episode.unique()[:15]:
        ep_df = df[df.episode == ep_id].reset_index(drop=True)
        if len(ep_df) < n_points + 5:
            continue
        i0 = 0
        i1 = min(len(ep_df) - 1, n_points)
        start_row, end_row = ep_df.iloc[i0], ep_df.iloc[i1]

        qpos0 = np.array([0.0, start_row.height, start_row.pitch,
                          start_row.thigh, start_row.leg, start_row.foot,
                          start_row.thigh_left, start_row.leg_left, start_row.foot_left])
        qvel0 = np.zeros(9); qvel0[2] = start_row.pitch_rate

        target_joints = np.array([end_row[j] for j in JOINTS])

        mw_ref = mw_path(det, start_row, end_row, n_points)
        naive_ref = naive_path(start_row, end_row, n_points)

        mw_final, mw_fell, mw_steps = track(qpos0, qvel0, mw_ref)
        nv_final, nv_fell, nv_steps = track(qpos0, qvel0, naive_ref)

        mw_err = np.abs(mw_final[3:9] - target_joints).mean()
        nv_err = np.abs(nv_final[3:9] - target_joints).mean()
        results.append(dict(ep=ep_id, mw_fell=mw_fell, nv_fell=nv_fell,
                            mw_steps=mw_steps, nv_steps=nv_steps,
                            mw_err=mw_err, nv_err=nv_err))

    res = pd.DataFrame(results)
    print(res)
    print(f"\nMW path:    fell {res.mw_fell.sum()}/{len(res)}, mean steps survived "
         f"{res.mw_steps.mean():.1f}, mean final joint error {res.mw_err.mean():.4f}")
    print(f"Naive path: fell {res.nv_fell.sum()}/{len(res)}, mean steps survived "
         f"{res.nv_steps.mean():.1f}, mean final joint error {res.nv_err.mean():.4f}")


if __name__ == "__main__":
    ep = 40 if "--quick" in sys.argv else 200
    main(ep)
