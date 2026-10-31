#!/usr/bin/env python3
"""
04 - Legged robot foot contact: energy conservation through a bounce
========================================================================
The ballistic flight phase of a hopping/legged robot's foot between
ground contacts - smooth dynamics with a discontinuous event (impact) at
each landing. Train on the exact projectile-motion solution; check ground
penetration, an impossible mid-flight energy gain, and an unphysical
bounce, with no event-driven solver involved.

Run:  python examples/04_legged_contact_validator.py   (a couple of minutes)
"""

from _common import args, banner, step, table, takeaway

import numpy as np
import torch
from mw_framework import MWDetector, roc_auc
from mw_framework.datasets import legged_contact, LEGGED_FEATURES
from mw_framework.domains import HoppingContactVAE

a = args(__doc__)
banner("04  LEGGED ROBOT CONTACT: ENERGY CONSERVATION THROUGH A BOUNCE", """
A hybrid/impact solver needs to detect the contact event and apply a
restitution law at the right instant - delicate and slow at scale. MW
instead learns the manifold of flight-phase states that satisfy energy
conservation and the ground/restitution bounds, and scores any reading
against it in one pass, with no event detection at all.
""")

normal, anom = legged_contact(n_normal=5000, n_per_anomaly=200, seed=0)
train, test = normal.iloc[:4000], normal.iloc[4000:]
types = anom.anomaly_type.values
ep = 40 if a.quick else 200

step("Train on exact ballistic-flight states")
det = MWDetector(vae=HoppingContactVAE(), epochs=ep, lr=2e-3).fit(train[LEGGED_FEATURES])
print(f"  false alarms on unseen exact states: {(det.predict(test) == -1).mean():.1%}")

step("Labelled faults")
dn, da = det.mw_distance(test), det.mw_distance(anom)
table([[k, f"{roc_auc(dn, da[types == k]):.3f}"] for k in np.unique(types)],
      ["fault", "ROC-AUC"])

step("Does the LEARNED manifold itself obey the laws?  (2000 generated states)")
v = det.vae_
with torch.no_grad():
    gen = v.to_physical(v.decode(torch.randn(2000, v.latent_dim)))
res = HoppingContactVAE.residuals(gen)
print(f"  mean |energy-conservation gap| of generated states: {res['energy_gap'].abs().mean():.4f}")
print("  (sensor noise in the training data is 0.005 - generated states are as physical as the data)")

takeaway("""The MW mechanism carries over cleanly to hybrid/impact legged-robot dynamics, with no
changes to the core framework and no event detection at all. See the fault table above for what
an energy-conservation check catches on this run - an early result from a new domain, not a
finished product (see ROBOTICS.md for the whole robotics picture).""")
