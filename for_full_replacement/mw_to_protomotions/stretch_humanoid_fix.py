#!/usr/bin/env python3
"""
Follow-up to stretch_humanoid.py's diagnosed weak point: false alarms
were 9.1% (vs Walker2d's 0.4%) and didn't improve with more epochs.
Simplest hypothesis to test first: is it just not enough TRAINING DATA
at this higher dimensionality (40 episodes x ~90 normal steps each),
before reaching for the richer-diversity fix that worked for Stages
4/6? Compare 40 vs 120 episodes, same controller, same everything else.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from mw_framework import MWDetector
from stretch_humanoid import collect, HumanoidPostureVAE, FEATURES

ep = 40 if "--quick" in sys.argv else 200

print("Collecting a FIXED held-out test set (disjoint seeds)...")
df_test, _, _ = collect(n_episodes=20, max_steps=150, seed0=9000)
normal_test = df_test[df_test.label == "normal"].reset_index(drop=True)

for n_episodes in (40, 80, 120):
    df, lo, hi = collect(n_episodes=n_episodes, max_steps=150, seed0=0)
    normal = df[df.label == "normal"].reset_index(drop=True)
    det = MWDetector(vae=HumanoidPostureVAE(low=list(lo), high=list(hi)), epochs=ep, lr=2e-3).fit(normal[FEATURES])
    flag = (det.predict(normal_test[FEATURES]) == -1).mean()
    print(f"n_episodes={n_episodes:>3d}  normal_rows={len(normal):>5d}  flag_rate_on_held_out={flag:.1%}")
