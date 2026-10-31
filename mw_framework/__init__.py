"""
Modak-Walawalkar (MW) Framework
===============================
Physics-constrained Bayesian manifold learning.

    Encode physics as priors.  Learn the manifold of physically possible
    states from normal data.  Score anything new by its MW distance from
    that manifold.

Companion code for the theory repository:
    https://github.com/RahulModak74/MW_Geometric_Intelligence

Authors: Rahul Modak, Dr. Rahul Walawalkar
"""

from .core import (MWNormaliser, MWVAE, MWTrainer, Calibration, calibrate,
                   constraint, bound, risk_score, risk_status, roc_auc,
                   set_seed, get_device, TIER_SIGMA)
from .config import ConfigVAE, build_from_config, load_config
from .detector import MWDetector
from .algebra import ManifoldAlgebra, SubVAE
from .geometry import (pullback_metric, magnification, geodesic, geodesic_distance,
                       curve_length, LorentzianCone)
from .checkpoint import save_checkpoint, load_checkpoint

__version__ = "1.0.0"
__author__ = "Rahul Modak, Dr. Rahul Walawalkar"

__all__ = [
    "MWNormaliser", "MWVAE", "MWTrainer", "Calibration", "calibrate",
    "constraint", "bound", "risk_score", "risk_status", "roc_auc",
    "set_seed", "get_device", "TIER_SIGMA",
    "ConfigVAE", "build_from_config", "load_config", "MWDetector",
    "ManifoldAlgebra", "SubVAE",
    "pullback_metric", "magnification", "geodesic", "geodesic_distance",
    "curve_length", "LorentzianCone",
    "save_checkpoint", "load_checkpoint",
]
