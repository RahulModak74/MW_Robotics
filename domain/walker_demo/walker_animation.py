"""Analogous to robot_motion.py — but for Walker2d, the frames are real
MuJoCo qpos/qvel (real physics), and MW distance is computed per step via
the trained WalkerPostureVAE. The manifold is the gauge, not the animation."""
from __future__ import annotations
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_DOMAIN = os.path.dirname(_HERE)
_REPO = os.path.dirname(_DOMAIN)
sys.path.insert(0, _DOMAIN)
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "mw_to_protomotions"))  # same fix as walker_app.py

import numpy as np
import pandas as pd
import gymnasium as gym


def run_policy_rollout(theta, max_steps=1000, push_at=-1, push=0.0,
                       det=None, seed=0):
    """Real MuJoCo rollout driven by a CEM policy. Returns state stream +
    MW distance stream + summary. Schema matches the Flask demo's
    {frames, domain, equations} but with qpos/qvel as the frame content."""
    from mw_to_protomotions.cem_policy_search import get_features, policy_action

    env = gym.make("Walker2d-v5",
                   exclude_current_positions_from_observation=False)
    obs, info = env.reset(seed=seed)
    data = env.unwrapped.data
    x0 = data.qpos[0]

    qpos_log, qvel_log, posture_rows = [], [], []
    steps = 0
    fell_at = None

    for t in range(max_steps):
        act = policy_action(theta, get_features(data))
        if t == push_at:
            data.qvel[0] += push

        qpos_log.append([float(v) for v in data.qpos])
        qvel_log.append([float(v) for v in data.qvel])
        posture_rows.append({
            "height": float(data.qpos[1]),
            "pitch":  float(data.qpos[2]),
            "pitch_rate": float(data.qvel[2]),
            "thigh":      float(data.qpos[3]), "leg":      float(data.qpos[4]),
            "foot":       float(data.qpos[5]),
            "thigh_left": float(data.qpos[6]), "leg_left": float(data.qpos[7]),
            "foot_left":  float(data.qpos[8]),
        })

        obs, r, term, trunc, info = env.step(act)
        steps += 1
        if term:
            fell_at = t
            break
    env.close()

    mw_distances = []
    mw_error = None
    if det is None:
        mw_error = "detector_not_loaded"
    elif posture_rows:
        from walker_domain import FEATURES
        try:
            df_posture = pd.DataFrame(posture_rows)[FEATURES]
            d = det.mw_distance(df_posture)
            mw_distances = [float(v) for v in d]
        except Exception as e:
            import traceback
            mw_error = f"{type(e).__name__}: {e}"
            print(f"[MW] mw_distance failed: {mw_error}", flush=True)
            traceback.print_exc()

    return {
        "qpos": qpos_log,
        "qvel": qvel_log,
        "mw_distances": mw_distances,
        "mw_error": mw_error,
        "steps": steps,
        "forward_distance": float(data.qpos[0] - x0),
        "fell_at": fell_at,
        "seed": seed,
        "push_at": push_at,
        "push": push,
    }
