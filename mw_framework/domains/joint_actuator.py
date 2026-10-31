"""
mw_framework.domains.joint_actuator
=====================================
A single robot joint: a PD-controlled motor-and-link pair modelled as a
damped second-order linear system, I*theta'' + b*theta' + k*theta = tau.
This is the ODE every joint controller's inner loop assumes - the rigid,
cheap-to-simulate half of a robot (its compliant links are PDEs; see
flexible_robot.py). The dataset is the EXACT underdamped step response,
closed form, no solver.

FEATURES (what a motor-driven joint can sense)
    t       time since the step command
    theta   joint angle                 (encoder)
    omega   angular velocity            (encoder derivative / tachometer)
    alpha   angular acceleration        (IMU / double-differentiated encoder)
    tau     torque                      (current-sense torque estimate -
                                         an INDEPENDENT estimate of the same
                                         physical quantity the dynamics imply)
    energy  generalised mechanical energy (KE + spring PE), nondimensional

LAWS
    C1  tau - (I*alpha + b*omega + k*theta) == 0   Newton's second law for
                                        this joint - a torque sensor and an
                                        encoder-derived estimate of the same
                                        quantity must agree. Bearing wear or
                                        a slipping gear breaks it.
    C2  energy(t) <= energy(0) * exp(-2*zeta*omega_n*t)   viscous decay
                                        envelope set by the joint's own
                                        damping ratio and natural frequency.
    C3  |tau| <= tau_max                actuator saturation / torque limit.

WHY this instead of simulating the ODE every time?
    This ODE is already cheap to integrate - the point here is not speed but
    CONSISTENCY CHECKING: does the torque sensor agree with what the encoder
    and the known dynamics imply? That is a per-state check, one encoder-
    decoder pass, no integration - useful even when the solver itself is fast.
"""

from __future__ import annotations

from typing import Dict

import torch

from ..core import MWVAE, constraint, bound

FEATURES = ["t", "theta", "omega", "alpha", "tau", "energy"]

I_NOM, B_NOM, K_NOM = 1.0, 0.4, 1.0
WN = (K_NOM / I_NOM) ** 0.5
ZETA = B_NOM / (2 * (I_NOM * K_NOM) ** 0.5)
TAU_MAX = 2.0           # actuator torque limit; normal data peaks ~1.4


class JointActuatorVAE(MWVAE):

    def __init__(self, latent_dim: int = 4, hidden_dim: int = 96,
                 dropout: float = 0.05, law_tolerance: float = 0.03, **kwargs):
        kwargs.setdefault("n_features", len(FEATURES))
        kwargs.setdefault("feature_names", FEATURES)
        kwargs.setdefault("obs_sigma", [0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
        super().__init__(latent_dim=latent_dim, hidden_dim=hidden_dim, dropout=dropout, **kwargs)
        self.law_tolerance = law_tolerance

    def get_config(self):
        cfg = super().get_config()
        cfg.update(law_tolerance=self.law_tolerance)
        return cfg

    def physics(self, x: Dict[str, torch.Tensor]) -> None:
        tol = self.law_tolerance
        constraint("c1_newton_second_law",
                  x["tau"] - (I_NOM * x["alpha"] + B_NOM * x["omega"] + K_NOM * x["theta"]),
                  tol, kind="studentt", df=3)
        ceiling = torch.exp(-2 * ZETA * WN * torch.clamp(x["t"], min=0.0))
        bound("c2_energy_envelope", x["energy"] - 1.0 * ceiling, high=0.0, tolerance=tol)
        bound("c3_torque_limit", x["tau"].abs(), high=TAU_MAX, tolerance=tol)

    @staticmethod
    def residuals(x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            "newton_gap": x["tau"] - (I_NOM * x["alpha"] + B_NOM * x["omega"] + K_NOM * x["theta"]),
            "torque": x["tau"].abs(),
        }
