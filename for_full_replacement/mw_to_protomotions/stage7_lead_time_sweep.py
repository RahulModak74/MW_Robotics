#!/usr/bin/env python3
"""
Stage 7, additional iteration: Part B of stage7_robustness.py showed MW
gives a 22.9-step early warning at ONE disturbance condition (push=3.0
at step 30, 15 falling episodes). Does that warning hold up across
DIFFERENT push sizes (does a bigger/faster fall still get flagged in
time, or does warning time shrink to nothing?) and different push
timings? Real dose-response, not one lucky number.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from mw_framework import MWDetector
from walker_domain import WalkerPostureVAE, FEATURES
from datagen import collect
from stage7_robustness import run_one

ep = 40 if "--quick" in sys.argv else 200

print("Training the Stage 1 validator...")
df = collect(n_episodes=60, max_steps=300)
normal = df[df.label == "normal"].reset_index(drop=True)
det = MWDetector(vae=WalkerPostureVAE(), epochs=ep, lr=2e-3).fit(normal[FEATURES])
held_out = normal.iloc[int(len(normal) * 0.8):]
threshold = np.percentile(det.mw_distance(held_out[FEATURES]), 99)
print(f"calibrated threshold: {threshold:.3f}\n")


def lead_time_for(push, push_at, n_seeds=12, max_steps=400):
    leads, n_fell = [], 0
    for seed in range(n_seeds):
        fell_at, traj = run_one(seed, push_at=push_at, push=push, max_steps=max_steps)
        if fell_at >= max_steps - 1 or len(traj) < 5:
            continue
        n_fell += 1
        dist = det.mw_distance(traj[FEATURES])
        over = np.where(dist > threshold)[0]
        if len(over) > 0:
            leads.append(fell_at - over[0])
    return n_fell, leads


print("Lead time vs push SEVERITY (fixed timing, step 30):")
for push in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
    n_fell, leads = lead_time_for(push, push_at=30)
    mean_lead = np.mean(leads) if leads else float("nan")
    print(f"  push={push:.1f}  fell={n_fell:>2d}/12  flagged_early={len(leads):>2d}/{n_fell}  "
         f"mean_lead={mean_lead:5.1f} steps")

print("\nLead time vs push TIMING (fixed severity, push=3.0):")
for push_at in (10, 30, 60, 100):
    n_fell, leads = lead_time_for(3.0, push_at=push_at)
    mean_lead = np.mean(leads) if leads else float("nan")
    print(f"  push_at={push_at:>3d}  fell={n_fell:>2d}/12  flagged_early={len(leads):>2d}/{n_fell}  "
         f"mean_lead={mean_lead:5.1f} steps")
