"""
mw_framework.config
===================
Describe a new domain in JSON - no Python needed.

{
  "domain": "flexible_robot",
  "features": [
     {"name": "GNSS_L1", "baseline": -128, "sigma": 2.0, "tier": 1},
     ...
  ],
  "constraints": [
     {"name": "l1_l2_link", "type": "linear",
      "terms": {"GNSS_L1": 1.0, "GNSS_L2": -1.0}, "constant": -3.0,
      "tolerance": 1.0},
     {"name": "gnss_floor", "type": "range", "feature": "GNSS_L1",
      "min": -140, "max": -115, "tolerance": 1.0}
  ],
  "architecture": {"latent_dim": 4, "hidden_dim": 64, "dropout": 0.1},
  "training":     {"epochs": 200, "lr": 1e-3, "batch_size": 128}
}

Constraint types (all in PHYSICAL units):
    linear : sum_i c_i * feature_i + constant  ~ 0   (+- tolerance)
    range  : min <= feature <= max                   (+- tolerance)
    ratio  : feature_a / feature_b - value     ~ 0   (+- tolerance)
Optional "kind": "normal" | "laplace" | "studentt" (default normal).

WHY JSON?
    Domain experts (roboticists, mechanical engineers) know the laws, not
    PyTorch. A config lets them encode those laws and try MW in minutes.
    When a law is too rich for JSON, subclass MWVAE (see mw_framework/domains).
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Union

import torch

from .core import MWVAE, MWNormaliser, TIER_SIGMA, constraint, bound

DEFAULTS = {
    "architecture": {"latent_dim": 4, "hidden_dim": 64, "dropout": 0.1, "kl_weight": 0.1},
    "training": {"epochs": 200, "lr": 1e-3, "batch_size": 128,
                 "calibration_split": 0.2, "calibration_percentile": 99.0,
                 "normal_label_col": None, "normal_label_value": None, "seed": 0},
}


def load_config(cfg: Union[str, Dict]) -> Dict:
    """Load and validate a domain config (path or dict). Fills defaults."""
    if isinstance(cfg, str):
        with open(cfg) as f:
            cfg = json.load(f)
    cfg = json.loads(json.dumps(cfg))          # deep copy
    if "features" not in cfg or not cfg["features"]:
        raise ValueError("config needs a non-empty 'features' list")
    names = [f["name"] for f in cfg["features"]]
    if len(set(names)) != len(names):
        raise ValueError("duplicate feature names in config")
    for f in cfg["features"]:
        f.setdefault("baseline", 0.0)
        f.setdefault("sigma", 1.0)
        if f["sigma"] <= 0:
            raise ValueError(f"feature {f['name']}: sigma must be > 0")
    for c in cfg.setdefault("constraints", []):
        _validate_constraint(c, set(names))
    for block, d in DEFAULTS.items():
        cfg.setdefault(block, {})
        for k, v in d.items():
            cfg[block].setdefault(k, v)
    cfg.setdefault("domain", "custom")
    return cfg


def _validate_constraint(c: Dict, names: set) -> None:
    t = c.get("type")
    c.setdefault("name", f"{t}_{abs(hash(json.dumps(c, sort_keys=True))) % 10**6}")
    c.setdefault("tolerance", 1.0)
    c.setdefault("kind", "normal")
    used = []
    if t == "linear":
        used = list(c["terms"].keys())
        c.setdefault("constant", 0.0)
    elif t == "range":
        used = [c["feature"]]
    elif t == "ratio":
        used = [c["numerator"], c["denominator"]]
    else:
        raise ValueError(f"constraint '{c.get('name')}': unknown type '{t}'")
    missing = [u for u in used if u not in names]
    if missing:
        raise ValueError(f"constraint '{c['name']}' uses unknown features {missing}")


def obs_sigma_from_config(cfg: Dict) -> List[float]:
    out = []
    for f in cfg["features"]:
        if "obs_sigma" in f:
            out.append(float(f["obs_sigma"]))
        else:
            out.append(TIER_SIGMA.get(int(f.get("tier", 2)), TIER_SIGMA[2]))
    return out


def normaliser_from_config(cfg: Dict) -> MWNormaliser:
    return MWNormaliser([f["baseline"] for f in cfg["features"]],
                        [f["sigma"] for f in cfg["features"]],
                        [f["name"] for f in cfg["features"]])


class ConfigVAE(MWVAE):
    """MWVAE whose physics() is generated from the JSON 'constraints' list."""

    def __init__(self, constraints: Optional[List[Dict]] = None, **kwargs):
        super().__init__(**kwargs)
        self.constraints = constraints or []

    def get_config(self) -> Dict:
        cfg = super().get_config()
        cfg["constraints"] = self.constraints
        return cfg

    def physics(self, x: Dict[str, torch.Tensor]) -> None:
        for c in self.constraints:
            t = c["type"]
            if t == "linear":
                r = sum(float(w) * x[n] for n, w in c["terms"].items()) + float(c["constant"])
                constraint(c["name"], r, c["tolerance"], c["kind"])
            elif t == "range":
                bound(c["name"], x[c["feature"]], c.get("min"), c.get("max"), c["tolerance"])
            elif t == "ratio":
                den = x[c["denominator"]]
                den = torch.where(den.abs() < 1e-6, torch.full_like(den, 1e-6), den)
                constraint(c["name"], x[c["numerator"]] / den - float(c["value"]),
                           c["tolerance"], c["kind"])


def build_from_config(cfg: Union[str, Dict]):
    """Return (ConfigVAE, MWNormaliser, cfg) ready for MWTrainer."""
    cfg = load_config(cfg)
    norm = normaliser_from_config(cfg)
    a = cfg["architecture"]
    vae = ConfigVAE(
        constraints=cfg["constraints"],
        n_features=len(cfg["features"]), latent_dim=a["latent_dim"],
        hidden_dim=a["hidden_dim"], dropout=a["dropout"], kl_weight=a["kl_weight"],
        obs_sigma=obs_sigma_from_config(cfg), feature_names=norm.feature_names,
        norm_mean=norm.mean.tolist(), norm_std=norm.std.tolist())
    return vae, norm, cfg
