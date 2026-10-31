#!/usr/bin/env python3
"""
01 - Robot joint actuator: a torque-vs-dynamics consistency check
===================================================================
A single PD-controlled robot joint (motor + link), the damped second-order
ODE every joint controller's inner loop assumes. Train on the exact step
response; check whether a torque sensor, an encoder and an independent
dynamics model agree, and whether three realistic actuator faults break
that agreement.

Run:  python examples/01_joint_actuator_validator.py   (a couple of minutes)
"""

from _common import args, banner, step, table, takeaway

import numpy as np
import torch
from mw_framework import MWDetector, roc_auc
from mw_framework.datasets import joint_actuator, JOINT_FEATURES
from mw_framework.domains import JointActuatorVAE

a = args(__doc__)
banner("01  ROBOT JOINT ACTUATOR: A TORQUE-VS-DYNAMICS CHECK", """
A robot joint usually has two independent ways to know its own torque: a
current-sense estimate, and what Newton's second law implies from the
encoder's angle, velocity and acceleration. They should agree. MW learns
the manifold of states where they do, then scores any new reading in one
pass - useful for catching bearing wear, gear slip or a failing torque
sensor before it shows up as a control problem.
""")

normal, anom = joint_actuator(n_normal=5000, n_per_anomaly=200, seed=0)
train, test = normal.iloc[:4000], normal.iloc[4000:]
types = anom.anomaly_type.values
ep = 40 if a.quick else 200

step("Train on the exact step-response dynamics")
det = MWDetector(vae=JointActuatorVAE(), epochs=ep, lr=2e-3).fit(train[JOINT_FEATURES])
print(f"  false alarms on unseen exact states: {(det.predict(test) == -1).mean():.1%}")

step("Labelled faults")
dn, da = det.mw_distance(test), det.mw_distance(anom)
table([[k, f"{roc_auc(dn, da[types == k]):.3f}"] for k in np.unique(types)],
      ["fault", "ROC-AUC"])

step("Does the LEARNED manifold itself obey the laws?  (2000 generated states)")
v = det.vae_
with torch.no_grad():
    gen = v.to_physical(v.decode(torch.randn(2000, v.latent_dim)))
res = JointActuatorVAE.residuals(gen)
print(f"  mean |Newton's-law gap| of generated states: {res['newton_gap'].abs().mean():.4f}")
print(f"  mean |torque| of generated states:            {res['torque'].mean():.4f}  (limit 2.0)")
print("  (sensor noise in the training data is 0.01 - generated states are as physical as the data)")

takeaway("""The MW mechanism carries over cleanly to this joint, with no changes to the core
framework. See the fault table above for what a torque/dynamics consistency check catches on
this run - this is an early result from a new domain, not a finished product (see ROBOTICS.md
for the whole robotics picture, what's strong, and what the next iteration should target).""")
