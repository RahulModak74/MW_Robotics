#!/usr/bin/env python3
"""
Train all six MW_Robotics domains once (on their SHIPPED closed-form
datasets - the same ones examples/00..05 use, not MuJoCo; this demo is
not a MuJoCo comparison, it's an interactive showcase of the six trained
manifolds) and cache the checkpoints to ./checkpoints/*.pth, so app.py
starts instantly instead of retraining every launch.

Run once before starting the app:
    python train_and_cache.py
Re-run any time to retrain (e.g. after editing a domain's physics).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyro

from mw_framework import MWDetector
from mw_framework.core import set_seed
from mw_framework.datasets import (cantilever_beam, ROBOT_FEATURES,
                                   joint_actuator, JOINT_FEATURES,
                                   planar_arm, ARM_FEATURES,
                                   mobile_robot, MOBILE_FEATURES,
                                   legged_contact, LEGGED_FEATURES,
                                   actuator_thermal, THERMAL_FEATURES)
from mw_framework.domains import (FlexibleRobotVAE, JointActuatorVAE, PlanarArmVAE,
                                  MobileRobotVAE, HoppingContactVAE, ActuatorThermalVAE)

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(HERE, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

DOMAINS = {
    "flexible_robot":  (cantilever_beam, ROBOT_FEATURES, FlexibleRobotVAE),
    "joint_actuator":  (joint_actuator, JOINT_FEATURES, JointActuatorVAE),
    "planar_arm":      (planar_arm, ARM_FEATURES, PlanarArmVAE),
    "mobile_robot":    (mobile_robot, MOBILE_FEATURES, MobileRobotVAE),
    "legged_contact":  (legged_contact, LEGGED_FEATURES, HoppingContactVAE),
    "actuator_thermal": (actuator_thermal, THERMAL_FEATURES, ActuatorThermalVAE),
}


def main():
    for name, (gen, feats, cls) in DOMAINS.items():
        path = os.path.join(CKPT_DIR, f"{name}.pth")
        print(f"Training {name} ...", flush=True)
        set_seed(0)
        pyro.clear_param_store()
        normal, _ = gen(n_normal=5000, n_per_anomaly=10, seed=0)
        det = MWDetector(vae=cls(), epochs=200, lr=2e-3).fit(normal[feats])
        det.save(path)
        print(f"  saved {path}")
    print("\nAll six domains cached. Start the app with: python app.py")


if __name__ == "__main__":
    main()
