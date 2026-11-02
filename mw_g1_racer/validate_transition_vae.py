#!/usr/bin/env python3
"""
Spec section 4.3's required validation metrics, run against whatever
checkpoint currently exists - used both as a final report AND as a cheap
mid-training check (is more training worth the wall-clock before
committing to it?).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from mw_framework import MWDetector, roc_auc
from transition_domain import FEATURES, STATE_FEATURES, ACTION_FEATURES, NEXT_FEATURES

det = MWDetector.load("transition_mw_walker.ckpt")
train = pd.read_parquet("transitions_train.parquet")
held = pd.read_parquet("transitions_heldout.parquet")

# V1: reconstruction floor
train_d = det.mw_distance(train[FEATURES].sample(5000, random_state=0))
held_d = det.mw_distance(held[FEATURES])
print(f"V1 reconstruction floor: train mean mw_distance={train_d.mean():.3f}, "
      f"held-out mean={held_d.mean():.3f} (ratio {held_d.mean()/train_d.mean():.2f}x, "
      f"pass if < 1.5x)")

# V2: injected faults
rng = np.random.default_rng(0)
base = held.sample(500, replace=True, random_state=1).reset_index(drop=True)


def joint_limit_fault(df):
    out = df.copy()
    j = rng.choice(["thigh_next", "leg_next", "thigh_left_next", "leg_left_next"], size=len(out))
    for i in range(len(out)):
        out.loc[i, j[i]] = rng.uniform(1.5, 3.0) * rng.choice([-1, 1])
    return out


def teleport_fault(df):
    out = df.copy()
    j = rng.choice(["thigh_next", "leg_next", "foot_next"], size=len(out))
    for i in range(len(out)):
        out.loc[i, j[i]] = out.loc[i, j[i].replace("_next", "")] + rng.uniform(1.0, 2.0) * rng.choice([-1, 1])
    return out


def action_saturation_fault(df):
    out = df.copy()
    j = rng.choice(ACTION_FEATURES, size=len(out))
    for i in range(len(out)):
        out.loc[i, j[i]] = rng.uniform(2.0, 5.0) * rng.choice([-1, 1])
    return out


faults = {
    "joint_limit": joint_limit_fault(base),
    "teleport": teleport_fault(base),
    "action_saturation": action_saturation_fault(base),
}
print("\nV2 injected fault AUC (target > 0.90):")
for name, fdf in faults.items():
    y_true = np.concatenate([np.zeros(len(base)), np.ones(len(fdf))])
    d_normal = det.mw_distance(base[FEATURES])
    d_fault = det.mw_distance(fdf[FEATURES])
    scores = np.concatenate([d_normal, d_fault])
    auc = roc_auc(y_true, scores)
    print(f"  {name:<20s} AUC = {auc:.3f}")

print("\nV3: organic near-fall rows aren't separately labeled in this "
      "transition dataset (unlike walker_domain's state-only one) - "
      "using HIGH mw_distance quantile vs LOW as an informal proxy instead:")
all_d = det.mw_distance(held[FEATURES])
print(f"  held-out mw_distance: p10={np.percentile(all_d,10):.2f} "
      f"p50={np.percentile(all_d,50):.2f} p90={np.percentile(all_d,90):.2f} "
      f"p99={np.percentile(all_d,99):.2f}")
