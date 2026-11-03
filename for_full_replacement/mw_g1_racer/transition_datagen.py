#!/usr/bin/env python3
"""
Data generator for SPEC_with_MW_3.md's gating experiment (section 8).
Same recipe as mw_to_protomotions/stage4_richer_datagen.py (many
independently-randomized physics "units", densely sampled - already proven
there to fix a real generalization gap), extended to log every (s, a, s')
transition instead of just labeled posture rows.

Honest scope cut from the spec's literal "500k-2M / 1M transitions" target:
that scale needs ~13 hours of SVI training at this repo's measured
epoch cost on a single CPU core. This generator can produce that many rows
cheaply - the cut is in how many of them get trained on later, not here.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import numpy as np
import pandas as pd
import gymnasium as gym

STATE_FEATURES = ["height", "pitch", "pitch_rate", "thigh", "leg", "foot",
                  "thigh_left", "leg_left", "foot_left"]
ACTION_FEATURES = ["tau_thigh", "tau_leg", "tau_foot",
                   "tau_thigh_left", "tau_leg_left", "tau_foot_left"]

KP = np.array([4.0, 4.0, 2.0, 4.0, 4.0, 2.0])
KD = np.array([0.5, 0.5, 0.2, 0.5, 0.5, 0.2])
BAL_KP, BAL_KD = 6.0, 1.0


def state_row(data, prefix=""):
    return {
        f"height{prefix}": data.qpos[1], f"pitch{prefix}": data.qpos[2],
        f"pitch_rate{prefix}": data.qvel[2],
        f"thigh{prefix}": data.qpos[3], f"leg{prefix}": data.qpos[4], f"foot{prefix}": data.qpos[5],
        f"thigh_left{prefix}": data.qpos[6], f"leg_left{prefix}": data.qpos[7],
        f"foot_left{prefix}": data.qpos[8],
    }


def collect_transitions(n_units=300, n_steps=300, seed0=0,
                        friction_range=(0.4, 1.3), damping_range=(0.05, 0.35)):
    """One persistent env, physics params re-randomized per unit (avoids
    gym.make()'s per-call XML-parse overhead - the main lesson learned
    from this folder's earlier CEM experiments)."""
    rng = np.random.default_rng(seed0)
    env = gym.make("Walker2d-v5", exclude_current_positions_from_observation=False)
    m, data = env.unwrapped.model, env.unwrapped.data
    rows = []
    for unit in range(n_units):
        friction = rng.uniform(*friction_range)
        damping = rng.uniform(*damping_range)
        m.geom_friction[0, 0] = friction
        m.dof_damping[3:9] = damping
        obs, info = env.reset(seed=seed0 + 1000 + unit)
        for step in range(n_steps):
            s_row = state_row(data)
            pitch, pitch_rate = data.qpos[2], data.qvel[2]
            bal = -1 * (BAL_KP * pitch + BAL_KD * pitch_rate)
            ref = np.zeros(6); ref[2] = bal; ref[5] = bal
            torque = np.clip(KP * (ref - data.qpos[3:9]) - KD * data.qvel[3:9], -1, 1)
            a_row = dict(zip(ACTION_FEATURES, torque))
            obs, r, term, trunc, info = env.step(torque)
            sp_row = state_row(data, prefix="_next")
            row = {**s_row, **a_row, **sp_row, "unit": unit, "friction": friction, "damping": damping}
            rows.append(row)
            if term:
                break
    env.close()
    return pd.DataFrame(rows)


def filter_transitions(df, v_max_delta=0.25):
    """Prior filters, spec section 3.1: reject joint-limit violations,
    ground penetration, and non-reachable (teleporting) transitions."""
    n0 = len(df)
    THIGH_LO, THIGH_HI = -2.61799388, 0.0
    LEG_LO, LEG_HI = -2.61799388, 0.0
    FOOT_LO, FOOT_HI = -0.78539816, 0.78539816
    keep = pd.Series(True, index=df.index)
    for j, lo, hi in [("thigh", THIGH_LO, THIGH_HI), ("leg", LEG_LO, LEG_HI),
                      ("foot", FOOT_LO, FOOT_HI), ("thigh_left", THIGH_LO, THIGH_HI),
                      ("leg_left", LEG_LO, LEG_HI), ("foot_left", FOOT_LO, FOOT_HI)]:
        keep &= df[f"{j}_next"].between(lo - 0.1, hi + 0.1)
    keep &= df["height_next"] > 0.0  # no ground penetration
    # reachability applies to POSITIONS only - pitch_rate is already a
    # velocity, and velocities legitimately jump fast during contact
    # (bug found via smoke test: this filter was keeping 4.6% of rows
    # before excluding pitch_rate, 95% of real rows have pitch_rate
    # deltas > 0.25 rad/s/step - that's normal contact dynamics, not a
    # teleport)
    position_features = [f for f in STATE_FEATURES if f != "pitch_rate"]
    for f in position_features:
        keep &= (df[f"{f}_next"] - df[f]).abs() <= v_max_delta
    n1 = keep.sum()
    print(f"  prior filters: kept {n1}/{n0} rows ({n1/n0:.1%})")
    return df[keep].reset_index(drop=True)


if __name__ == "__main__":
    t0 = time.time()
    df = collect_transitions(n_units=10, n_steps=50)
    print(f"smoke test: {len(df)} rows in {time.time()-t0:.1f}s")
    df2 = filter_transitions(df)
    print(df2.head())
