"""
mw_framework.domains.planar_arm
=================================
A 2-link planar robot arm: no PDE, no ODE - pure kinematics, the part of a
robot Isaac Sim computes almost for free. The point here is not speed but
CROSS-SENSOR CONSISTENCY: a robot usually has two independent ways to know
where its end-effector is (joint encoders + forward kinematics, vs. vision
or an end-effector IMU). They should agree. When they don't, something is
wrong with the encoders, the kinematic model, or the vision system - and a
learned manifold of "configurations where they agree" can say which kind of
disagreement it is from the shape of the mismatch, no solver involved.

FEATURES
    theta1, theta2            joint angles              (encoders)
    theta1dot, theta2dot      joint angular rates        (encoders)
    x, y                      end-effector position      (vision / IMU)
    xdot, ydot                end-effector velocity      (vision / IMU)

LAWS (kinematic identities - exact geometry, no solver)
    C1  (x, y) == forward_kinematics(theta1, theta2)        two independent
                                        position estimates must agree.
    C2  (xdot, ydot) == Jacobian(theta) @ (theta1dot, theta2dot)   two
                                        independent velocity estimates must
                                        agree.
    C3  |theta1|, |theta2| within the arm's physical joint limits.

WHY this instead of recomputing kinematics every time?
    Forward kinematics is cheap to compute. What is NOT free is knowing,
    from a stream of sensor readings alone, whether the two sensors still
    agree with each other and with every other configuration the arm has
    shown - i.e. catching calibration drift or a bent link before it causes
    a collision, from the readings alone, no geometry model evaluated per
    check beyond the one-pass manifold score.
"""

from __future__ import annotations

from typing import Dict

import torch

from ..core import MWVAE, constraint, bound

FEATURES = ["theta1", "theta2", "theta1dot", "theta2dot", "x", "y", "xdot", "ydot"]

L1, L2 = 1.0, 0.8
THETA1_LIMIT, THETA2_LIMIT = 1.6, 2.4     # physical joint-limit stops; normal data stays well inside


class PlanarArmVAE(MWVAE):

    def __init__(self, latent_dim: int = 5, hidden_dim: int = 96,
                 dropout: float = 0.05, law_tolerance: float = 0.03, **kwargs):
        kwargs.setdefault("n_features", len(FEATURES))
        kwargs.setdefault("feature_names", FEATURES)
        kwargs.setdefault("obs_sigma", [0.05] * len(FEATURES))
        super().__init__(latent_dim=latent_dim, hidden_dim=hidden_dim, dropout=dropout, **kwargs)
        self.law_tolerance = law_tolerance

    def get_config(self):
        cfg = super().get_config()
        cfg.update(law_tolerance=self.law_tolerance)
        return cfg

    def physics(self, x: Dict[str, torch.Tensor]) -> None:
        tol = self.law_tolerance
        t1, t2 = x["theta1"], x["theta2"]
        s1, c1 = torch.sin(t1), torch.cos(t1)
        s12, c12 = torch.sin(t1 + t2), torch.cos(t1 + t2)
        x_fk = L1 * c1 + L2 * c12
        y_fk = L1 * s1 + L2 * s12
        constraint("c1_fk_x", x["x"] - x_fk, tol, kind="studentt", df=3)
        constraint("c1_fk_y", x["y"] - y_fk, tol, kind="studentt", df=3)

        t1d, t2d = x["theta1dot"], x["theta2dot"]
        xdot_fk = -L1 * s1 * t1d - L2 * s12 * (t1d + t2d)
        ydot_fk = L1 * c1 * t1d + L2 * c12 * (t1d + t2d)
        constraint("c2_jac_x", x["xdot"] - xdot_fk, tol, kind="studentt", df=3)
        constraint("c2_jac_y", x["ydot"] - ydot_fk, tol, kind="studentt", df=3)

        bound("c3_theta1_limit", t1.abs(), high=THETA1_LIMIT, tolerance=tol)
        bound("c3_theta2_limit", t2.abs(), high=THETA2_LIMIT, tolerance=tol)

    @staticmethod
    def residuals(x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        t1, t2 = x["theta1"], x["theta2"]
        s1, c1 = torch.sin(t1), torch.cos(t1)
        s12, c12 = torch.sin(t1 + t2), torch.cos(t1 + t2)
        x_fk, y_fk = L1 * c1 + L2 * c12, L1 * s1 + L2 * s12
        return {"fk_gap": ((x["x"] - x_fk) ** 2 + (x["y"] - y_fk) ** 2).sqrt()}
