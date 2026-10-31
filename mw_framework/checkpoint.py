"""
mw_framework.checkpoint
=======================
One .pth file holds everything needed to score new data:
weights, constructor config, normaliser, calibration, and the class path.
"""

from __future__ import annotations

import importlib
from typing import Dict, Optional, Tuple

import torch

from .core import Calibration, MWNormaliser

FORMAT = "mw_framework/1"


def save_checkpoint(path: str, vae: torch.nn.Module, norm: MWNormaliser,
                    calibration: Calibration, domain: str = "custom",
                    extra: Optional[Dict] = None) -> None:
    cls = type(vae)
    torch.save({
        "format": FORMAT,
        "class": f"{cls.__module__}.{cls.__qualname__}",
        "init_kwargs": vae.get_config(),
        "state_dict": {k: v.cpu() for k, v in vae.state_dict().items()},
        "normaliser": norm.to_dict(),
        "calibration": calibration.to_dict(),
        "domain": domain,
        "extra": extra or {},
    }, path)


def load_checkpoint(path: str) -> Tuple[torch.nn.Module, MWNormaliser, Calibration, Dict]:
    """
    Returns (vae, normaliser, calibration, meta).
    The VAE class must be importable (e.g. mw_framework.config.ConfigVAE or
    mw_framework.domains.flexible_robot.FlexibleRobotVAE). Classes defined inside a
    script's __main__ can only be reloaded from that same script.
    """
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("format") != FORMAT:
        raise ValueError(f"{path}: not an {FORMAT} checkpoint")
    mod_name, _, cls_name = ck["class"].rpartition(".")
    cls = getattr(importlib.import_module(mod_name), cls_name)
    vae = cls(**ck["init_kwargs"])
    vae.load_state_dict(ck["state_dict"])
    vae.eval()
    meta = {"domain": ck["domain"], "extra": ck["extra"], "class": ck["class"]}
    return vae, MWNormaliser.from_dict(ck["normaliser"]), \
        Calibration(**ck["calibration"]), meta
