"""
Fast tests (about 1 min on a laptop CPU).   Run:  pytest -q
They check that the machinery works, not benchmark numbers.
"""

import os
import sys

import numpy as np
import pyro
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mw_framework import (MWDetector, MWNormaliser, MWTrainer, MWVAE,
                          LorentzianCone, roc_auc, set_seed, geodesic, pullback_metric)
from mw_framework.datasets import (circle, cantilever_beam, ROBOT_FEATURES,
                                   joint_actuator, JOINT_FEATURES,
                                   planar_arm, ARM_FEATURES,
                                   mobile_robot, MOBILE_FEATURES,
                                   legged_contact, LEGGED_FEATURES,
                                   actuator_thermal, THERMAL_FEATURES)
from mw_framework.domains import (FlexibleRobotVAE, JointActuatorVAE, PlanarArmVAE,
                                  MobileRobotVAE, HoppingContactVAE, ActuatorThermalVAE)


def test_normaliser_roundtrip():
    X = np.random.default_rng(0).normal(5, 2, (100, 3))
    n = MWNormaliser.fit(X, ["a", "b", "c"])
    assert np.allclose(n.inverse(n.transform(X, clip=None)), X, atol=1e-4)
    assert MWNormaliser.from_dict(n.to_dict()).feature_names == ["a", "b", "c"]


def test_detector_learns_circle():
    # a domain-agnostic toy manifold, used only to sanity-check the core
    # machinery itself (encoder/decoder/calibration), not a robotics claim.
    normal, anom = circle(800, 200, seed=0)
    det = MWDetector(latent_dim=2, hidden_dim=32, epochs=300, lr=3e-3).fit(normal.iloc[:600])
    auc = roc_auc(det.mw_distance(normal.iloc[600:]), det.mw_distance(anom))
    assert auc > 0.8
    assert set(np.unique(det.predict(anom))) <= {-1, 1}
    r = det.risk_score(anom)
    assert r.min() >= 0 and r.max() <= 100


def test_sklearn_sign_convention():
    normal, anom = circle(600, 100, seed=1)
    det = MWDetector(latent_dim=2, hidden_dim=32, epochs=60, lr=3e-3).fit(normal.iloc[:500])
    # sklearn: score_samples higher = more normal
    assert det.score_samples(normal.iloc[500:]).mean() > det.score_samples(anom).mean()


ROBOTICS_DOMAINS = [FlexibleRobotVAE, JointActuatorVAE, PlanarArmVAE,
                   MobileRobotVAE, HoppingContactVAE, ActuatorThermalVAE]


@pytest.mark.parametrize("cls", ROBOTICS_DOMAINS)
def test_robotics_domain_runs_one_epoch(cls):
    set_seed(0)
    pyro.clear_param_store()
    vae = cls()
    X = torch.randn(64, vae.n_features) * 0.5
    MWTrainer(vae, epochs=1, verbose=False).fit(X)
    m, s = vae.mw_distance(X, mc_samples=3)
    assert m.shape == (64,) and torch.isfinite(m).all()


# Each domain's strongest, cleanly-reproduced fault (see ROBOTICS.md for the
# full table, including the faults each domain does NOT yet catch well).
_STRONGEST_FAULT = {
    "flexible_beam":  (cantilever_beam, ROBOT_FEATURES, FlexibleRobotVAE, "no_decay"),
    "joint_actuator": (joint_actuator, JOINT_FEATURES, JointActuatorVAE, "no_settle"),
    "planar_arm":     (planar_arm, ARM_FEATURES, PlanarArmVAE, "velocity_glitch"),
    "mobile_robot":   (mobile_robot, MOBILE_FEATURES, MobileRobotVAE, "overspeed"),
    "legged_contact": (legged_contact, LEGGED_FEATURES, HoppingContactVAE, "bad_restitution"),
    "actuator_thermal": (actuator_thermal, THERMAL_FEATURES, ActuatorThermalVAE, "cooling_failure"),
}


@pytest.mark.parametrize("name", list(_STRONGEST_FAULT))
def test_robotics_domain_detects_its_strongest_fault(name):
    gen, features, cls, fault = _STRONGEST_FAULT[name]
    set_seed(0)
    pyro.clear_param_store()
    normal, anom = gen(n_normal=1500, n_per_anomaly=80, seed=0)
    det = MWDetector(vae=cls(), epochs=60, lr=2e-3).fit(normal[features])
    picked = anom[anom.anomaly_type == fault]
    auc = roc_auc(det.mw_distance(normal[features]), det.mw_distance(picked[features]))
    # 0.75, not 0.85: this quick regime (1500 rows, 60 epochs) is smaller/faster than the
    # full example (5000 rows, 200 epochs) and has more retraining variance - e.g.
    # planar_arm/velocity_glitch measured 0.889 in the full run, 0.809-0.81 here.
    assert auc > 0.75


def test_geometry():
    set_seed(0)
    pyro.clear_param_store()
    vae = MWVAE(2, latent_dim=2, dropout=0.0)
    z = torch.randn(4, 2)
    G = pullback_metric(vae, z)
    assert G.shape == (4, 2, 2)
    assert torch.all(torch.linalg.eigvalsh(G) >= -1e-5)      # positive semi-definite
    path, L, Ls = geodesic(vae, z[0], z[1], n_points=8, steps=20)
    assert path.shape == (8, 2) and L <= Ls * 1.05


def test_lorentzian_cone():
    t = np.arange(10.0)
    soh = 1 - 0.01 * t
    cone = LorentzianCone(["soh"], c=0.02, arrow={"soh": -1}, arrow_tol={"soh": 0.001})
    assert all(v.label == "TIMELIKE" for v in cone.scan(t, soh[:, None]))
    assert cone.classify(0, [1.0], 1, [0.9]).label == "SPACELIKE"
    assert cone.classify(0, [0.9], 1, [0.95]).label == "ARROW_VIOLATION"
    assert cone.classify(1, [0.9], 1, [0.9]).label == "NO_TIME_STEP"
