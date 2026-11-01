"""
Stage 1: a coupled, many-joint, contact-rich body as an MW domain - not
one of this repo's existing six single-piece domains, but the harder
case path_to_full_Robotics.md's Stage 1 asks for: all of Walker2d's
joints learned together, from real trajectories (datagen.py), with
SOFT priors (not exact closed-form laws - full coupled body dynamics
don't reduce to three clean equations the way one joint or one beam
does). Soft priors here: the 6 joints' exact physical limits (from the
MuJoCo model's own XML - these ARE exact, not approximate), plus a
soft upright/height band matching the real controller's own stable
operating range.
"""
from __future__ import annotations
from typing import Dict
import torch
from mw_framework.core import MWVAE, bound

FEATURES = ["height", "pitch", "pitch_rate", "thigh", "leg", "foot",
           "thigh_left", "leg_left", "foot_left"]

# exact joint limits, read directly from the MuJoCo model (confirmed via
# model.jnt_range earlier this session)
THIGH_LO, THIGH_HI = -2.61799388, 0.0
LEG_LO, LEG_HI = -2.61799388, 0.0
FOOT_LO, FOOT_HI = -0.78539816, 0.78539816


class WalkerPostureVAE(MWVAE):
    def __init__(self, latent_dim: int = 6, hidden_dim: int = 96,
                 dropout: float = 0.05, law_tolerance: float = 0.05, **kwargs):
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
        # exact joint limits - these are real, hard physical facts about
        # the robot's own hardware, not a soft approximation
        bound("c1_thigh_limit", x["thigh"], low=THIGH_LO, high=THIGH_HI, tolerance=tol)
        bound("c1_leg_limit", x["leg"], low=LEG_LO, high=LEG_HI, tolerance=tol)
        bound("c1_foot_limit", x["foot"], low=FOOT_LO, high=FOOT_HI, tolerance=tol)
        bound("c1_thigh_left_limit", x["thigh_left"], low=THIGH_LO, high=THIGH_HI, tolerance=tol)
        bound("c1_leg_left_limit", x["leg_left"], low=LEG_LO, high=LEG_HI, tolerance=tol)
        bound("c1_foot_left_limit", x["foot_left"], low=FOOT_LO, high=FOOT_HI, tolerance=tol)
        # soft balance envelope - matches the real controller's own stable
        # operating range (not the env's termination bound, which is wider)
        bound("c2_pitch_band", x["pitch"], low=-1.0, high=1.0, tolerance=tol)
        bound("c2_height_band", x["height"], low=0.8, high=2.0, tolerance=tol)

    @staticmethod
    def residuals(x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {"pitch_abs": x["pitch"].abs()}
