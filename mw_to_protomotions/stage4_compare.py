#!/usr/bin/env python3
"""
Stage 4 test: same question FORECASTER.md Part 3 already answered for
the bouncing leg (does richer, multi-condition training data generalise
better to unseen conditions than a narrow single-config dataset?), now
on the coupled Walker2d body. Train validator A on ONE physics config
(datagen.py's default - friction=0.7, damping=0.1, matching Stage 1),
validator B on many random configs (stage4_richer_datagen.py). Test
both on a FRESH set of held-out (friction, damping) combinations neither
was trained on. Honest framing: this tests condition-diversity, not
motion-diversity (no real AMASS access here) - see this file's and
stage4_richer_datagen.py's docstrings.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from mw_framework import MWDetector
from walker_domain import WalkerPostureVAE, FEATURES
from datagen import collect
from stage4_richer_datagen import collect_richer

ep = 40 if "--quick" in sys.argv else 200

print("Validator A: single physics config (Stage 1's original data)...")
df_a = collect(n_episodes=60, max_steps=300)
normal_a = df_a[df_a.label == "normal"].reset_index(drop=True)
det_a = MWDetector(vae=WalkerPostureVAE(), epochs=ep, lr=2e-3).fit(normal_a[FEATURES])

print("Validator B: many random physics configs (richer datagen)...")
df_b = collect_richer(n_units=40, max_steps=250, seed0=100)
det_b = MWDetector(vae=WalkerPostureVAE(), epochs=ep, lr=2e-3).fit(df_b[FEATURES])

print("Held-out test: FRESH random (friction, damping) combos, seed range disjoint from both...")
df_test = collect_richer(n_units=20, max_steps=250, seed0=9000)

flag_a = (det_a.predict(df_test[FEATURES]) == -1).mean()
flag_b = (det_b.predict(df_test[FEATURES]) == -1).mean()
print(f"\nFlag rate on held-out physics conditions:")
print(f"  Validator A (single config)   : {flag_a:.1%}")
print(f"  Validator B (richer, 40 units): {flag_b:.1%}")
