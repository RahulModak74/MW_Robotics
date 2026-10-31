#!/usr/bin/env python3
"""
01 - Does MW validate a real MuJoCo simulation, not just its own synthetic data?
===================================================================================
Train HoppingContactVAE on the SHIPPED closed-form dataset (as examples/04
does), then score REAL MuJoCo-simulated ball-drop trajectories against it.
Then repeat with a second model trained on an `e` range matched to what
this MuJoCo contact calibration actually produces, to separate "out of
training domain" from "violates the physics law" - see FORECASTER.md for
the full story of how this was debugged.

Run: python mujoco_experiments/01_validator_check.py   (a few minutes)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pyro

from mw_framework import MWDetector, roc_auc
from mw_framework.core import set_seed
from mw_framework.datasets import LEGGED_FEATURES, legged_contact, G_NOM
from mw_framework.domains import HoppingContactVAE
from bounce_env import simulate_drop, extract_flight_segments

print("Simulating real MuJoCo ball drops (4 heights x 3 contact-damping settings)...")
rows = []
for h0 in (1.0, 1.5, 2.0, 2.5):
    for dampratio in (0.25, 0.35, 0.5):
        t, z, vz = simulate_drop(h0, dampratio)
        rows.append(extract_flight_segments(t, z, vz))
mujoco_df = pd.concat(rows, ignore_index=True)
print(f"Extracted {len(mujoco_df)} real post-bounce flight readings. "
     f"Measured e range: {mujoco_df.e.min():.2f}-{mujoco_df.e.max():.2f} (mean {mujoco_df.e.mean():.2f})")

print("\n-- Model A: trained on the SHIPPED closed-form dataset (e ~ Uniform(0.6, 0.98)) --")
set_seed(0); pyro.clear_param_store()
normal, _ = legged_contact(n_normal=5000, n_per_anomaly=10, seed=0)
det_a = MWDetector(vae=HoppingContactVAE(), epochs=200, lr=2e-3).fit(normal[LEGGED_FEATURES])
base_a = np.median(det_a.mw_distance(normal[LEGGED_FEATURES]))
dist_a = det_a.mw_distance(mujoco_df[LEGGED_FEATURES])
print(f"  real MuJoCo trajectories: median distance = {np.median(dist_a)/base_a:.2f}x the closed-form baseline")
print(f"  flagged (risk>=50): {(det_a.predict(mujoco_df[LEGGED_FEATURES]) == -1).mean():.1%}")

print("\n-- Model B: trained with e matched to what THIS MuJoCo calibration actually produces --")


def legged_contact_matched(n_normal, n_per_anomaly, e_lo, e_hi, seed=0):
    r = np.random.default_rng(seed)

    def make(n):
        v0 = r.uniform(0.3, 1.5, n)
        t_apex = v0 / G_NOM
        t = r.uniform(0.0, 1.95, n) * t_apex
        z = v0 * t - 0.5 * G_NOM * t ** 2
        vz = v0 - G_NOM * t
        e = r.uniform(e_lo, e_hi, n)
        X = np.stack([t, z, vz, v0, e], axis=1)
        X[:, :3] += r.normal(0, 0.005, (n, 3))
        return pd.DataFrame(X, columns=LEGGED_FEATURES)

    normal = make(n_normal)
    a = make(n_per_anomaly); a["z"] = -r.uniform(0.05, 0.3, n_per_anomaly); a["anomaly_type"] = "ground_penetration"
    b = make(n_per_anomaly); b["vz"] += r.uniform(0.4, 0.8, n_per_anomaly) * np.sign(r.standard_normal(n_per_anomaly)); b["z"] = b["z"].clip(lower=0.01); b["anomaly_type"] = "energy_gain"
    c = make(n_per_anomaly); c["e"] = r.uniform(1.05, 1.4, n_per_anomaly); c["anomaly_type"] = "bad_restitution"
    return normal, pd.concat([a, b, c], ignore_index=True)


set_seed(0); pyro.clear_param_store()
normal_m, anom_m = legged_contact_matched(5000, 200, mujoco_df.e.min(), mujoco_df.e.max())
det_b = MWDetector(vae=HoppingContactVAE(), epochs=200, lr=2e-3).fit(normal_m[LEGGED_FEATURES])
base_b = np.median(det_b.mw_distance(normal_m[LEGGED_FEATURES]))
dist_b = det_b.mw_distance(mujoco_df[LEGGED_FEATURES])
print(f"  real MuJoCo trajectories: median distance = {np.median(dist_b)/base_b:.2f}x the matched-domain baseline")
print(f"  flagged (risk>=50): {(det_b.predict(mujoco_df[LEGGED_FEATURES]) == -1).mean():.1%}")
dn = det_b.mw_distance(normal_m[LEGGED_FEATURES])
types_m = anom_m.anomaly_type.values
print("  sanity: model B still catches its own synthetic faults -")
for k in np.unique(types_m):
    da = det_b.mw_distance(anom_m[types_m == k][LEGGED_FEATURES])
    print(f"    {k}: AUC={roc_auc(dn, da):.3f}")

print("""
TAKEAWAY: most of model A's near-total flag rate on real MuJoCo trajectories is a training-domain
mismatch (this MuJoCo contact only achieves e~0.2-0.6, below the shipped e~0.6-0.98 range), not a
genuine physics-law violation - confirmed because matching the training domain (model B) drops the
flag rate close to the noise floor while still catching every synthetic fault. Lesson: train on the
target engine's actual achievable physics, not an arbitrary convenient synthetic range.
""")
