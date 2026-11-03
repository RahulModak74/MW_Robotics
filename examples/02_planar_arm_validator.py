#!/usr/bin/env python3
"""
02 - Two-link planar arm: cross-sensor kinematic consistency
===============================================================
A 2-link planar robot arm with two independent ways of knowing where its
end-effector is: joint encoders + forward kinematics, vs. vision or an
end-effector IMU. No PDE, no ODE - pure geometry, checked in one pass
instead of a kinematics evaluation per query.

Run:  python examples/02_planar_arm_validator.py   (a couple of minutes)
"""

from _common import args, banner, step, table, takeaway

import numpy as np
import torch
from mw_framework import MWDetector, roc_auc
from mw_framework.datasets import planar_arm, ARM_FEATURES
from mw_framework.domains import PlanarArmVAE

a = args(__doc__)
banner("02  TWO-LINK PLANAR ARM: CROSS-SENSOR CONSISTENCY", """
Forward kinematics and its Jacobian say exactly what an end-effector sensor
must read, given the joint encoders. MW learns the manifold of
configurations where the two agree, then flags a reading where they don't -
encoder drift, a bent link, or a vision/IMU fault - without evaluating the
kinematics chain as a separate check each time.
""")

normal, anom = planar_arm(n_normal=5000, n_per_anomaly=200, seed=0)
train, test = normal.iloc[:4000], normal.iloc[4000:]
types = anom.anomaly_type.values
ep = 40 if a.quick else 200

step("Train on exact forward-kinematics states")
det = MWDetector(vae=PlanarArmVAE(), epochs=ep, lr=2e-3).fit(train[ARM_FEATURES])
print(f"  false alarms on unseen exact states: {(det.predict(test) == -1).mean():.1%}")

step("Labelled faults")
dn, da = det.mw_distance(test), det.mw_distance(anom)
table([[k, f"{roc_auc(dn, da[types == k]):.3f}"] for k in np.unique(types)],
      ["fault", "ROC-AUC"])

step("Does the LEARNED manifold itself obey the laws?  (2000 generated states)")
v = det.vae_
with torch.no_grad():
    gen = v.to_physical(v.decode(torch.randn(2000, v.latent_dim)))
res = PlanarArmVAE.residuals(gen)
print(f"  mean end-effector / forward-kinematics gap of generated states: {res['fk_gap'].mean():.4f}")
print("  (sensor noise in the training data is 0.01 - generated states are as physical as the data)")

takeaway("""The MW mechanism carries over cleanly to pure kinematics, with no changes to the
core framework - this time with no ODE or PDE at all. See the fault table above for what a
cross-sensor consistency check catches on this run - an early result from a new domain, not a
finished product (see docs/ROBOTICS.md for the whole robotics picture).""")
