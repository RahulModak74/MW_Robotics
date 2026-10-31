"""
mw_framework.domains.mobile_robot
===================================
A differential-drive / unicycle mobile robot. The kinematics are an ODE
(dx/dt = v*cos(theta), dy/dt = v*sin(theta), dtheta/dt = omega), but the
law that matters for fault detection is an algebraic one at every instant:
a wheeled robot that isn't slipping cannot move sideways. That is the
textbook nonholonomic constraint, and it is exactly the signature of the
most common wheeled-robot fault - wheel slip.

FEATURES
    x, y, theta        pose                         (odometry / SLAM)
    v, omega           commanded body speed, turn rate   (wheel encoders)
    xdot, ydot, thetadot   instantaneous velocity      (IMU / visual odometry -
                                                        an INDEPENDENT estimate)

LAWS (kinematic identities and a drivetrain limit - no solver)
    C1  xdot*sin(theta) - ydot*cos(theta) == 0     nonholonomic constraint:
                                        no sideways motion. A slipping wheel
                                        is exactly a violation of this.
    C2  thetadot - omega == 0          the gyro/visual heading rate must
                                        match the commanded turn rate.
    C3  |v| <= v_max                   drivetrain speed limit.

WHY this instead of integrating the ODE every time?
    The ODE is cheap. What matters operationally is a live check of whether
    the two independent velocity estimates (wheel encoders vs. IMU/visual
    odometry) still agree with the physics of a non-slipping wheeled robot -
    a one-pass manifold score per reading, not a dead-reckoning integration.
"""

from __future__ import annotations

from typing import Dict

import torch

from ..core import MWVAE, constraint, bound

FEATURES = ["x", "y", "theta", "v", "omega", "xdot", "ydot", "thetadot"]
V_MAX = 2.0


class MobileRobotVAE(MWVAE):

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
        nonholonomic = x["xdot"] * torch.sin(x["theta"]) - x["ydot"] * torch.cos(x["theta"])
        constraint("c1_nonholonomic", nonholonomic, tol, kind="studentt", df=3)
        constraint("c2_heading_rate", x["thetadot"] - x["omega"], tol, kind="studentt", df=3)
        bound("c3_speed_limit", x["v"].abs(), high=V_MAX, tolerance=tol)

    @staticmethod
    def residuals(x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {"lateral_slip": x["xdot"] * torch.sin(x["theta"]) - x["ydot"] * torch.cos(x["theta"])}
