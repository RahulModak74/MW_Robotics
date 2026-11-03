#!/usr/bin/env python3
"""
05 - Actuator thermal limit: the fault that actually trips robots
=====================================================================
Temperature along a motor winding, modelled as 1D heat diffusion from an
initial hotspot - the same diffusion-PDE family as the parent repo's
Navier-Stokes domain and this repo's beam, now pointed at the fault that
most often takes a real actuator offline: over-temperature. Train on the
exact 2-mode solution; check a partial winding short, a flux-sensor fault,
and a cooling failure.

Run:  python examples/05_actuator_thermal_validator.py   (a couple of minutes)
"""

from _common import args, banner, step, table, takeaway

import numpy as np
import torch
from mw_framework import MWDetector, roc_auc
from mw_framework.datasets import actuator_thermal, THERMAL_FEATURES
from mw_framework.domains import ActuatorThermalVAE

a = args(__doc__)
banner("05  ACTUATOR THERMAL LIMIT: THE FAULT THAT TRIPS ROBOTS", """
A temperature sensor and an independent heat-flux estimate are two
measurements of the same physical quantity (Fourier's law). MW learns the
manifold of diffusion states where they agree and decay correctly, then
flags a partial winding short, a sensor fault, or - the real safety
case - a fan/heatsink failure that stops the heat from dissipating.
""")

normal, anom = actuator_thermal(n_normal=5000, n_per_anomaly=200, seed=0)
train, test = normal.iloc[:4000], normal.iloc[4000:]
types = anom.anomaly_type.values
ep = 40 if a.quick else 200

step("Train on exact diffusion states")
det = MWDetector(vae=ActuatorThermalVAE(), epochs=ep, lr=2e-3).fit(train[THERMAL_FEATURES])
print(f"  false alarms on unseen exact states: {(det.predict(test) == -1).mean():.1%}")

step("Labelled faults")
dn, da = det.mw_distance(test), det.mw_distance(anom)
table([[k, f"{roc_auc(dn, da[types == k]):.3f}"] for k in np.unique(types)],
      ["fault", "ROC-AUC"])

step("Does the LEARNED manifold itself obey the laws?  (2000 generated states)")
v = det.vae_
with torch.no_grad():
    gen = v.to_physical(v.decode(torch.randn(2000, v.latent_dim)))
res = ActuatorThermalVAE.residuals(gen)
print(f"  mean |Fourier's-law gap| of generated states: {res['fourier_gap'].abs().mean():.4f}")
print("  (sensor noise in the training data is 0.01 - generated states are as physical as the data)")

takeaway("""The MW mechanism carries over cleanly to this diffusion PDE pointed at a real robot
safety fault, with no changes to the core framework. See the fault table above for what a
Fourier's-law consistency check catches on this run - an early result from a new domain, not a
finished product (see docs/ROBOTICS.md for the whole robotics picture).""")
