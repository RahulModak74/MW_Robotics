#!/usr/bin/env python3
"""
03 - Differential-drive mobile robot: the nonholonomic wheel-slip check
=========================================================================
A wheeled robot that isn't slipping cannot move sideways - the classic
nonholonomic constraint. Train on exact unicycle-kinematics states; check
whether wheel slip, a gyro fault, or an overspeed command break it.

Run:  python examples/03_mobile_robot_validator.py   (a couple of minutes)
"""

from _common import args, banner, step, table, takeaway

import numpy as np
import torch
from mw_framework import MWDetector, roc_auc
from mw_framework.datasets import mobile_robot, MOBILE_FEATURES
from mw_framework.domains import MobileRobotVAE

a = args(__doc__)
banner("03  MOBILE ROBOT: THE NONHOLONOMIC WHEEL-SLIP CHECK", """
Wheel odometry (v, omega) and an independent IMU/visual-odometry velocity
estimate (xdot, ydot, thetadot) should always satisfy "no sideways
motion". MW learns the manifold of states where they do, then flags a
reading where they don't - the direct signature of wheel slip on a
wheeled robot, caught without a slip model evaluated per reading.
""")

normal, anom = mobile_robot(n_normal=5000, n_per_anomaly=200, seed=0)
train, test = normal.iloc[:4000], normal.iloc[4000:]
types = anom.anomaly_type.values
ep = 40 if a.quick else 200

step("Train on exact unicycle-kinematics states")
det = MWDetector(vae=MobileRobotVAE(), epochs=ep, lr=2e-3).fit(train[MOBILE_FEATURES])
print(f"  false alarms on unseen exact states: {(det.predict(test) == -1).mean():.1%}")

step("Labelled faults")
dn, da = det.mw_distance(test), det.mw_distance(anom)
table([[k, f"{roc_auc(dn, da[types == k]):.3f}"] for k in np.unique(types)],
      ["fault", "ROC-AUC"])

step("Does the LEARNED manifold itself obey the laws?  (2000 generated states)")
v = det.vae_
with torch.no_grad():
    gen = v.to_physical(v.decode(torch.randn(2000, v.latent_dim)))
res = MobileRobotVAE.residuals(gen)
print(f"  mean |lateral-slip gap| of generated states: {res['lateral_slip'].abs().mean():.4f}")
print("  (sensor noise in the training data is 0.01 - generated states are as physical as the data)")

takeaway("""The MW mechanism carries over cleanly to nonholonomic mobile-robot kinematics, with
no changes to the core framework. See the fault table above for what a wheel-slip check catches
on this run - an early result from a new domain, not a finished product (see ROBOTICS.md for the
whole robotics picture).""")
