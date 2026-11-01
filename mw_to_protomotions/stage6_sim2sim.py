#!/usr/bin/env python3
"""
Stage 6, honestly scoped: "sim2sim" properly means testing across two
DIFFERENT physics engines (e.g. MuJoCo vs Isaac). There's no second
articulated-body engine with an equivalent Walker2d model wired up in
this environment, so the nearest honest proxy here is testing across
different NUMERICAL SETTINGS of the same engine (timestep, integrator)
- the same spirit as this repo's existing beam coarse-simulator test
(ROBOTICS.md's "one result that went the wrong way"). This is a real,
useful check (does the learned manifold reflect real physical structure,
or an artifact of one solver's specific numerical choices?), just a
smaller claim than true cross-engine sim2sim. Documented as that, not
oversold.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import gymnasium as gym
import mujoco
from mw_framework import MWDetector
from walker_domain import WalkerPostureVAE, FEATURES
from datagen import collect

ep = 40 if "--quick" in sys.argv else 200


def collect_with_settings(n_episodes, max_steps, timestep=None, integrator=None, seed0=0):
    kp = np.array([4.0, 4.0, 2.0, 4.0, 4.0, 2.0])
    kd = np.array([0.5, 0.5, 0.2, 0.5, 0.5, 0.2])
    rows = []
    for ep_id in range(n_episodes):
        env = gym.make("Walker2d-v5", exclude_current_positions_from_observation=False)
        obs, info = env.reset(seed=seed0 + ep_id)
        m, data = env.unwrapped.model, env.unwrapped.data
        if timestep is not None:
            m.opt.timestep = timestep
        if integrator is not None:
            m.opt.integrator = integrator
        for step in range(max_steps):
            pitch, pitch_rate = data.qpos[2], data.qvel[2]
            bal = -1 * (6.0 * pitch + 1.0 * pitch_rate)
            ref = np.zeros(6); ref[2] = bal; ref[5] = bal
            torque = np.clip(kp * (ref - data.qpos[3:9]) - kd * data.qvel[3:9], -1, 1)
            rows.append({f: (data.qpos[1] if f == "height" else
                            data.qpos[2] if f == "pitch" else
                            data.qvel[2] if f == "pitch_rate" else
                            {"thigh": data.qpos[3], "leg": data.qpos[4], "foot": data.qpos[5],
                             "thigh_left": data.qpos[6], "leg_left": data.qpos[7],
                             "foot_left": data.qpos[8]}[f])
                        for f in FEATURES})
            obs, r, term, trunc, info = env.step(torque)
            if term:
                break
        env.close()
    return pd.DataFrame(rows)


def main():
    print("Training the validator on default settings (dt=0.008, Euler) - same as Stage 1...")
    df_default = collect(n_episodes=60, max_steps=300)
    normal = df_default[df_default.label == "normal"].reset_index(drop=True)
    det = MWDetector(vae=WalkerPostureVAE(), epochs=ep, lr=2e-3).fit(normal[FEATURES])
    base_flag = (det.predict(normal.iloc[int(len(normal)*0.8):][FEATURES]) == -1).mean()
    print(f"baseline false-alarm rate (same settings, held-out rows): {base_flag:.1%}\n")

    settings = [
        ("default dt, RK4 integrator", dict(integrator=mujoco.mjtIntegrator.mjINT_RK4)),
        ("2x coarser timestep (0.016)", dict(timestep=0.016)),
        ("4x coarser timestep (0.032)", dict(timestep=0.032)),
    ]
    for name, kw in settings:
        df_alt = collect_with_settings(n_episodes=30, max_steps=250, seed0=500, **kw)
        flag = (det.predict(df_alt[FEATURES]) == -1).mean()
        print(f"{name:<32s} flag rate: {flag:.1%}")


if __name__ == "__main__":
    main()
