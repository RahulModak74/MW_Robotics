#!/usr/bin/env python3
"""
Stage 6 follow-up: Stage 6 found the validator doesn't generalise well
across different NUMERICAL solver settings (17-55% flag rate), even
though richer PHYSICAL-parameter training fixed the equivalent gap for
physical conditions (Stage 4). Natural next question: does the SAME
richer-training-data fix work here too - train across multiple solver
settings instead of one, then test on a setting none of them used?
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
from stage6_sim2sim import collect_with_settings

ep = 40 if "--quick" in sys.argv else 200

print("Validator A: single setting (default, dt=0.008, Euler) - same as Stage 1/6...")
df_a = collect(n_episodes=60, max_steps=300)
normal_a = df_a[df_a.label == "normal"].reset_index(drop=True)
det_a = MWDetector(vae=WalkerPostureVAE(), epochs=ep, lr=2e-3).fit(normal_a[FEATURES])

print("Validator B: trained across FOUR solver settings (default, RK4, dt=0.012, dt=0.02)...")
rows = []
settings_train = [
    dict(), dict(integrator=mujoco.mjtIntegrator.mjINT_RK4),
    dict(timestep=0.012), dict(timestep=0.02),
]
for i, kw in enumerate(settings_train):
    d = collect_with_settings(n_episodes=20, max_steps=250, seed0=200 + i * 100, **kw)
    rows.append(d)
df_b = pd.concat(rows, ignore_index=True)
det_b = MWDetector(vae=WalkerPostureVAE(), epochs=ep, lr=2e-3).fit(df_b[FEATURES])

print("\nTesting both on settings NEITHER was specifically tuned to confirm on:")
test_settings = [
    ("dt=0.016 (2x coarser, Stage 6's worst case)", dict(timestep=0.016)),
    ("dt=0.032 (4x coarser)", dict(timestep=0.032)),
    ("RK4 integrator", dict(integrator=mujoco.mjtIntegrator.mjINT_RK4)),
]
for name, kw in test_settings:
    df_test = collect_with_settings(n_episodes=30, max_steps=250, seed0=5000, **kw)
    flag_a = (det_a.predict(df_test[FEATURES]) == -1).mean()
    flag_b = (det_b.predict(df_test[FEATURES]) == -1).mean()
    print(f"{name:<45s} single-setting={flag_a:5.1%}   multi-setting={flag_b:5.1%}")
