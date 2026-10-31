"""
mw_framework.domains.legged_contact
=====================================
The ballistic flight phase of a hopping or legged robot's foot (or body, in
a SLIP-model abstraction common in legged-robotics research) between ground
contacts. This is the part of legged locomotion that is hardest for a
classical solver to handle cleanly: the dynamics are smooth in flight but
have a discontinuous event (impact, with a restitution law) at contact.
A manifold of "physically admissible flight states" sidesteps the event-
detection machinery a solver needs, by scoring each sampled instant against
energy conservation directly - no event, no solver, just a state check.

FEATURES
    t     time since launch (last ground contact)
    z     height                      (leg encoder / lidar)
    vz    vertical velocity           (IMU)
    v0    launch speed (this flight phase's initial condition)
    e     coefficient of restitution of the previous bounce

LAWS (exact projectile-motion identity and physical bounds - no solver)
    C1  vz^2 + 2*g*z - v0^2 == 0        energy conservation during ballistic
                                        flight (nondimensional g=1) - an
                                        actuator cannot add energy mid-flight.
    C2  z >= 0                          non-penetration: the foot/body
                                        cannot go through the ground.
    C3  0 < e <= 1                      a passive bounce cannot add energy;
                                        e > 1 is unphysical.

WHY this instead of an event-driven solver?
    A hybrid/impact solver needs to detect the contact event and apply the
    restitution law at exactly the right time step - delicate and slow at
    scale. Scoring a sampled flight-phase reading against the conservation
    law and the two bounds is a single encoder-decoder pass, with no event
    detection needed at all.
"""

from __future__ import annotations

from typing import Dict

import torch

from ..core import MWVAE, constraint, bound

FEATURES = ["t", "z", "vz", "v0", "e"]
G_NOM = 1.0


class HoppingContactVAE(MWVAE):

    def __init__(self, latent_dim: int = 4, hidden_dim: int = 80,
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
        energy_gap = x["vz"] ** 2 + 2 * G_NOM * x["z"] - x["v0"] ** 2
        constraint("c1_energy_conservation", energy_gap, tol, kind="studentt", df=3)
        bound("c2_non_penetration", x["z"], low=0.0, tolerance=tol)
        bound("c3_restitution_low", x["e"], low=0.0, tolerance=tol)
        bound("c3_restitution_high", x["e"], high=1.0, tolerance=tol)

    @staticmethod
    def residuals(x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {"energy_gap": x["vz"] ** 2 + 2 * G_NOM * x["z"] - x["v0"] ** 2}
