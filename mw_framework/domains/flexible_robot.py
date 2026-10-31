"""
mw_framework.domains.flexible_robot
====================================
A flexible-link robot arm / soft-robot segment: the Euler-Bernoulli beam PDE.

    rho*A * w_tt + EI * w_xxxx = f(x, t)

This is the governing PDE behind flexible-link manipulators and many soft
robot segments (a rigid robot's joints are ODEs; its compliant links and
soft actuators are this PDE). The dataset is the EXACT closed-form modal
solution for a cantilever beam released from an initial deflection and
left to decay under damping - no numerical solver, same spirit as the
Taylor-Green vortex in the parent repo's navier_stokes.py.

FEATURES (what a strain-gauge / IMU instrumented flexible link can sense)
    x       position along the link, 0 (fixed base) to 1 (free tip)
    t       time since release
    w       deflection                     (vision / encoder)
    w_t     deflection velocity            (IMU / finite difference)
    w_xx    curvature                      (strain gauge, DIRECT sensor)
    M       bending moment                 (from joint torque + lever arm,
                                             an INDEPENDENT estimate of the
                                             same physical quantity w_xx reports)
    energy  modal mechanical energy (KE + PE), nondimensional

LAWS (kinematic/constitutive identities and physical bounds - no solver)
    C1  M - EI_nominal * w_xx == 0     constitutive law (Euler-Bernoulli):
                                        moment and curvature are the SAME
                                        physical quantity seen by two sensors.
                                        A crack / local stiffness loss breaks it.
    C2  energy(t) <= energy(0) * exp(-2*zeta*omega_1*t)   viscous energy-decay
                                        envelope set by the slowest (mode-1) rate.
    C3  |w_xx| <= kappa_max             strain limit (material yield bound).

WHY this instead of simulating the PDE every time?
    A time-stepping beam/FEM solver integrates mode or mesh dynamics forward:
    cost grows with the number of modes/elements and shrinks the timestep
    needed for stability. MW learns the manifold of states that SATISFY the
    laws above and scores any new state - from a real sensor or from a cheap
    simulator - in one encoder-decoder pass, O(n) in the number of features.

SCOPE (see docs/LIMITATIONS.md and ROBOTICS.md)
    This validates whether a GIVEN state is physical. It does not forecast
    the NEXT state from the current one - it is not yet a drop-in replacement
    for a forward-time robot simulator. See ROBOTICS.md for what would close
    that gap.
"""

from __future__ import annotations

from typing import Dict

import torch

from ..core import MWVAE, constraint, bound

FEATURES = ["x", "t", "w", "w_t", "w_xx", "M", "energy"]

# first two cantilever eigenvalues (beta_n * L), standard result, L = 1 (nondimensional)
BETA = (1.8751040687, 4.6940911330)
EI_NOMINAL = 1.0          # nondimensional stiffness used to build "M" from "w_xx"
KAPPA_MAX = 9.0           # strain/yield limit on curvature; normal data peaks ~6.8, so this
                          # is a loose physical ceiling, not a tight operating band
ZETA = 0.02               # modal damping ratio used to generate the data


class FlexibleRobotVAE(MWVAE):

    def __init__(self, latent_dim: int = 4, hidden_dim: int = 128,
                 dropout: float = 0.05, law_tolerance: float = 0.03,
                 omega1: float = BETA[0] ** 2, **kwargs):
        kwargs.setdefault("n_features", len(FEATURES))
        kwargs.setdefault("feature_names", FEATURES)
        # x, t are trusted coordinates; sensed/derived fields need to reconstruct tightly
        kwargs.setdefault("obs_sigma", [0.05, 0.05] + [0.05] * 5)
        super().__init__(latent_dim=latent_dim, hidden_dim=hidden_dim, dropout=dropout, **kwargs)
        self.omega1 = omega1
        self.law_tolerance = law_tolerance

    def get_config(self):
        cfg = super().get_config()
        cfg.update(omega1=self.omega1, law_tolerance=self.law_tolerance)
        return cfg

    def physics(self, x: Dict[str, torch.Tensor]) -> None:
        tol = self.law_tolerance
        # C1 — constitutive law: moment sensor and curvature sensor read the
        # same physical quantity. StudentT: a real crack is rare but physical,
        # the law should bend for it rather than forbid it outright.
        constraint("c1_moment_curvature", x["M"] - EI_NOMINAL * x["w_xx"], tol,
                   kind="studentt", df=3)
        # C2 — energy cannot exceed the damped envelope set by the slowest mode.
        ceiling = torch.exp(-2 * ZETA * self.omega1 * torch.clamp(x["t"], min=0.0))
        e0 = 1.0  # energies are generated normalised so energy(0) ~= 1
        bound("c2_energy_envelope", x["energy"] - e0 * ceiling, high=0.0, tolerance=tol)
        # C3 — strain / yield limit.
        bound("c3_strain_limit", x["w_xx"].abs(), high=KAPPA_MAX, tolerance=tol)

    @staticmethod
    def residuals(x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Law residuals of any field (raw data or decoded samples)."""
        return {
            "moment_curvature_gap": x["M"] - EI_NOMINAL * x["w_xx"],
            "strain": x["w_xx"].abs(),
        }
