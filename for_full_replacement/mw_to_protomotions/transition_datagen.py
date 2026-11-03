"""
mw_to_protomotions/transition_datagen
======================================
Emit (s, a, s') transitions from real MuJoCo rollouts, using the
pybamm_like recipe from mujoco_experiments/03 — many independent units,
continuously varied physics parameters.
"""
import numpy as np
import pandas as pd
import gymnasium as gym
import mujoco
from mw_framework.domains.transition import (
    STATE_FEATURES, ACTION_FEATURES, FEATURES)


def _state_row(data) -> dict:
    return {
        "height": float(data.qpos[1]),
        "pitch": float(data.qpos[2]),
        "pitch_rate": float(data.qvel[2]),
        "thigh": float(data.qpos[3]), "leg": float(data.qpos[4]),
        "foot": float(data.qpos[5]),
        "thigh_left": float(data.qpos[6]), "leg_left": float(data.qpos[7]),
        "foot_left": float(data.qpos[8]),
    }


def collect_transitions(n_units: int = 20, n_steps: int = 250,
                        seed0: int = 100,
                        friction_range=(0.4, 1.3),
                        damping_range=(0.05, 0.35)) -> pd.DataFrame:
    """One 'unit' = one MuJoCo instance with its own sampled physics.
    Runs the PD controller from datagen.py, logging every transition."""
    kp = np.array([4.0, 4.0, 2.0, 4.0, 4.0, 2.0])
    kd = np.array([0.5, 0.5, 0.2, 0.5, 0.5, 0.2])
    rng = np.random.default_rng(seed0)
    rows = []
    for unit in range(n_units):
        fr = rng.uniform(*friction_range)
        dp = rng.uniform(*damping_range)
        env = gym.make("Walker2d-v5", exclude_current_positions_from_observation=False)
        env.reset(seed=seed0 + unit)
        m, d = env.unwrapped.model, env.unwrapped.data
        m.geom_friction[0, 0] = fr
        m.dof_damping[3:9] = dp
        for step in range(n_steps):
            s = _state_row(d)
            pitch, pitch_rate = s["pitch"], s["pitch_rate"]
            bal = -1.0 * (6.0 * pitch + 1.0 * pitch_rate)
            ref = np.zeros(6); ref[2] = bal; ref[5] = bal
            a = np.clip(kp * (ref - d.qpos[3:9]) - kd * d.qvel[3:9], -1, 1)
            _, _, term, trunc, _ = env.step(a)
            sp = _state_row(d)
            row = {f"s_{k}": s[k] for k in STATE_FEATURES}
            row.update({ACTION_FEATURES[i]: float(a[i]) for i in range(6)})
            row.update({f"sp_{k}": sp[k] for k in STATE_FEATURES})
            row["unit"] = unit
            row["step"] = step
            row["friction"] = fr
            row["damping"] = dp
            rows.append(row)
            if term or trunc:
                break
        env.close()
    return pd.DataFrame(rows)
