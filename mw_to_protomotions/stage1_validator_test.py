#!/usr/bin/env python3
"""
Stage 1 test: does MW's validator mechanism - proven six times already
on single-piece domains - extend to a coupled, many-joint body at all?
Train on real "normal standing" rows from the tuned PD controller,
measure false alarms on held-out normal rows, and ROC-AUC against two
kinds of fault: (a) hand-injected impossible postures (joint beyond
limit, flipped pitch - the equivalent of the other domains' injected
faults), and (b) the controller's own REAL near-fall states (organic,
not injected - the most honest test available, since they're real
physics, not a synthetic anomaly).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from mw_framework import MWDetector, roc_auc
from walker_domain import WalkerPostureVAE, FEATURES
from datagen import collect, inject_impossible

quick = "--quick" in sys.argv
ep = 40 if quick else 200

print("Collecting real trajectories from the tuned PD controller...")
df = collect(n_episodes=60, max_steps=300)
normal = df[df.label == "normal"].reset_index(drop=True)
near_fall = df[df.label == "near_fall"].reset_index(drop=True)
print(f"normal rows: {len(normal)}, near_fall rows: {len(near_fall)}")

train, test = normal.iloc[: int(len(normal) * 0.8)], normal.iloc[int(len(normal) * 0.8):]

print("\nTraining WalkerPostureVAE on real standing states...")
det = MWDetector(vae=WalkerPostureVAE(), epochs=ep, lr=2e-3).fit(train[FEATURES])
false_alarm = (det.predict(test) == -1).mean()
print(f"false alarms on held-out NORMAL rows: {false_alarm:.1%}")

print("\nInjected-fault test (hand-made impossible postures):")
anom = inject_impossible(df, n=300)
dn, da = det.mw_distance(test), det.mw_distance(anom)
for k in np.unique(anom.anomaly_type):
    print(f"  {k:<20s} ROC-AUC {roc_auc(dn, da[(anom.anomaly_type == k).values]):.3f}")

print("\nOrganic-fault test (the controller's OWN real near-fall states, not injected):")
d_nf = det.mw_distance(near_fall)
print(f"  near_fall (real physics, no synthetic injection) ROC-AUC {roc_auc(dn, d_nf):.3f}")

print("\nDoes the learned manifold itself respect the joint limits? (2000 generated postures)")
import torch
v = det.vae_
with torch.no_grad():
    gen = v.to_physical(v.decode(torch.randn(2000, v.latent_dim)))
res = WalkerPostureVAE.residuals(gen)
print(f"  mean |pitch| of generated postures: {res['pitch_abs'].mean():.3f} rad "
     f"(real standing data mean |pitch|: {normal.pitch.abs().mean():.3f})")
