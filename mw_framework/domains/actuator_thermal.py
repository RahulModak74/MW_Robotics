"""
mw_framework.domains.actuator_thermal
=======================================
Temperature along a motor winding or actuator housing: 1D heat diffusion
from an initial hotspot, insulated ends - the same PDE family as the fluid
and beam domains (a diffusion equation), applied to the fault that actually
trips robots in the field: motor over-temperature. The dataset is the EXACT
2-mode cosine-series solution of the insulated-rod heat equation, same
"modal expansion, no solver" spirit as flexible_robot.py's beam.

FEATURES
    x      position along the winding             (thermocouple location)
    t      time since the hotspot formed
    T      temperature                             (thermocouple)
    dTdx   temperature gradient                     (two thermocouples, or
                                                     a distributed sensor)
    flux   heat flux                                (a current-based or
                                                     independent flux estimate
                                                     - the SAME physical
                                                     quantity Fourier's law
                                                     ties to dTdx)
    excess normalised excess temperature relative to the worst point at t=0

LAWS (a constitutive identity and physical bounds - no solver)
    C1  flux - (-k * dTdx) == 0        Fourier's law: a gradient sensor and
                                        a flux estimate are two measurements
                                        of the same physical quantity. A
                                        partial winding short breaks it.
    C2  excess(t) <= excess(0) * exp(-alpha*pi^2*t)   diffusive decay
                                        envelope set by the slowest mode.
    C3  T <= T_max                     over-temperature shutdown limit -
                                        the actual safety bound a real
                                        actuator trips on (loose here; not
                                        exercised by the three faults below,
                                        included because it is the real
                                        constraint a deployed motor has).

WHY this instead of solving the heat equation every time?
    The diffusion PDE is one more instance of exactly the Taylor-Green
    pattern in the parent repo's Navier-Stokes domain, now pointed at a
    robotics safety fault. The payoff is the same: one encoder-decoder
    pass checks whether a reading is consistent with diffusive physics,
    no mesh, no time-stepping.
"""

from __future__ import annotations

from typing import Dict

import torch

from ..core import MWVAE, constraint, bound

FEATURES = ["x", "t", "T", "dTdx", "flux", "excess"]
ALPHA_TH = 0.05
K_COND = 1.0
T_MAX = 3.0


class ActuatorThermalVAE(MWVAE):

    def __init__(self, latent_dim: int = 4, hidden_dim: int = 96,
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
        constraint("c1_fourier_law", x["flux"] - (-K_COND * x["dTdx"]), tol, kind="studentt", df=3)
        ceiling = torch.exp(-ALPHA_TH * (3.14159265 ** 2) * torch.clamp(x["t"], min=0.0))
        bound("c2_decay_envelope", x["excess"] - 1.0 * ceiling, high=0.0, tolerance=tol)
        bound("c3_overtemp_limit", x["T"], high=T_MAX, tolerance=tol)

    @staticmethod
    def residuals(x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {"fourier_gap": x["flux"] - (-K_COND * x["dTdx"])}
