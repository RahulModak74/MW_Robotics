#!/usr/bin/env python3
"""
Stretch test (beyond the 7 planned stages): does Stage 1's result scale
up from Walker2d (6 joints, 2D) to MuJoCo's full Humanoid-v5 (17 joints,
true 3D, free-floating base with quaternion orientation) - much closer
to the actual ProtoMotions target than Walker2d? Honest scope limit
stated up front: FEATURES below use height + the 17 joint angles only,
NOT orientation (quaternion/roll/pitch/yaw) or velocities - a real
simplification, not hidden, to keep this a same-day stretch test rather
than a full redo of Stage 1's design for 3D. Falling over is primarily
an orientation phenomenon, so this tests less of the "is it upright"
question than Stage 1 did for Walker2d - stated plainly, not papered
over.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import gymnasium as gym
from mw_framework import MWDetector, roc_auc
from mw_framework.core import MWVAE, bound
import torch

N_JOINTS = 17
JOINT_NAMES = [f"j{i}" for i in range(N_JOINTS)]
FEATURES = ["height"] + JOINT_NAMES


def collect(n_episodes=40, max_steps=150, seed0=0, normal_frac=0.6):
    env0 = gym.make("Humanoid-v5")
    env0.reset(seed=0)
    m0 = env0.unwrapped.model
    lo = m0.jnt_range[1:1 + N_JOINTS, 0].copy()
    hi = m0.jnt_range[1:1 + N_JOINTS, 1].copy()
    env0.close()

    kp, kd = 8.0, 0.8
    rows = []
    for ep in range(n_episodes):
        env = gym.make("Humanoid-v5")
        obs, info = env.reset(seed=seed0 + ep)
        data = env.unwrapped.data
        ref = data.qpos[7:7 + N_JOINTS].copy()   # hold the reset pose
        traj = []
        for step in range(max_steps):
            joints = data.qpos[7:7 + N_JOINTS]
            joint_vel = data.qvel[6:6 + N_JOINTS]
            torque = np.clip(kp * (ref - joints) - kd * joint_vel, -0.4, 0.4)
            traj.append({"height": data.qpos[2],
                        **{JOINT_NAMES[i]: joints[i] for i in range(N_JOINTS)}})
            obs, r, term, trunc, info = env.step(torque)
            if term or trunc:
                break
        env.close()
        n = len(traj)
        for i, row in enumerate(traj):
            row["label"] = "normal" if i < n * normal_frac else "near_fall"
            row["episode"] = ep
        rows.extend(traj)
    return pd.DataFrame(rows), lo, hi


class HumanoidPostureVAE(MWVAE):
    def __init__(self, latent_dim=8, hidden_dim=128, dropout=0.05, law_tolerance=0.05,
                low=None, high=None, **kwargs):
        kwargs.setdefault("n_features", len(FEATURES))
        kwargs.setdefault("feature_names", FEATURES)
        kwargs.setdefault("obs_sigma", [0.05] * len(FEATURES))
        super().__init__(latent_dim=latent_dim, hidden_dim=hidden_dim, dropout=dropout, **kwargs)
        self.law_tolerance = law_tolerance
        self.low = low if low is not None else [-3.0] * N_JOINTS
        self.high = high if high is not None else [3.0] * N_JOINTS

    def get_config(self):
        cfg = super().get_config()
        cfg.update(law_tolerance=self.law_tolerance, low=list(self.low), high=list(self.high))
        return cfg

    def physics(self, x) -> None:
        tol = self.law_tolerance
        for i, name in enumerate(JOINT_NAMES):
            bound(f"c1_joint_limit_{i}", x[name], low=self.low[i], high=self.high[i], tolerance=tol)
        bound("c2_height_band", x["height"], low=0.8, high=1.6, tolerance=tol)


def inject_impossible(df, lo, hi, n=200, seed=0):
    rng = np.random.default_rng(seed)
    base = df[df.label == "normal"].sample(n, replace=True, random_state=seed).reset_index(drop=True)
    out = base.copy()
    kind = rng.choice(["joint_limit", "height_impossible"], size=n)
    for i in range(n):
        if kind[i] == "joint_limit":
            j = rng.integers(0, N_JOINTS)
            out.loc[i, JOINT_NAMES[j]] = hi[j] + rng.uniform(1.0, 2.0)
        else:
            out.loc[i, "height"] = rng.choice([rng.uniform(0.1, 0.4), rng.uniform(2.0, 3.0)])
    out["anomaly_type"] = kind
    return out


def main(epochs):
    print("Collecting real Humanoid-v5 trajectories (PD-hold-pose controller)...")
    df, lo, hi = collect(n_episodes=40, max_steps=150)
    print(df.label.value_counts())
    normal = df[df.label == "normal"].reset_index(drop=True)
    near_fall = df[df.label == "near_fall"].reset_index(drop=True)
    train, test = normal.iloc[:int(len(normal)*0.8)], normal.iloc[int(len(normal)*0.8):]

    print(f"\nTraining HumanoidPostureVAE ({N_JOINTS} joints, 18 features total)...")
    det = MWDetector(vae=HumanoidPostureVAE(low=list(lo), high=list(hi)), epochs=epochs, lr=2e-3).fit(train[FEATURES])
    false_alarm = (det.predict(test) == -1).mean()
    print(f"false alarms on held-out NORMAL rows: {false_alarm:.1%}")

    anom = inject_impossible(df, lo, hi, n=200)
    dn, da = det.mw_distance(test), det.mw_distance(anom)
    print("\nInjected-fault test:")
    for k in np.unique(anom.anomaly_type):
        print(f"  {k:<20s} ROC-AUC {roc_auc(dn, da[(anom.anomaly_type==k).values]):.3f}")

    d_nf = det.mw_distance(near_fall)
    print(f"\nOrganic near-fall ROC-AUC: {roc_auc(dn, d_nf):.3f}")


if __name__ == "__main__":
    ep = 40 if "--quick" in sys.argv else 200
    main(ep)
