#!/usr/bin/env python3
"""
Stage 4, additional iteration: not just one A/B point (21.8% -> 2.4%),
but a dose-response curve - does more training-data diversity keep
helping, or plateau? Same FORECASTER.md-style discipline: measure, don't
assume a single comparison tells the whole story.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from mw_framework import MWDetector
from walker_domain import WalkerPostureVAE, FEATURES
from stage4_richer_datagen import collect_richer

ep = 40 if "--quick" in sys.argv else 200

print("Fixed held-out test set (fresh seeds, disjoint from all training)...")
df_test = collect_richer(n_units=20, max_steps=250, seed0=9000)

for n_units in (5, 10, 20, 40, 80):
    df_train = collect_richer(n_units=n_units, max_steps=250, seed0=100)
    det = MWDetector(vae=WalkerPostureVAE(), epochs=ep, lr=2e-3).fit(df_train[FEATURES])
    flag = (det.predict(df_test[FEATURES]) == -1).mean()
    print(f"n_units={n_units:>3d}  rows={len(df_train):>5d}  flag_rate={flag:.1%}")
