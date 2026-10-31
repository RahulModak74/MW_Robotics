"""
mw_framework.detector
=====================
scikit-learn style anomaly detector over the MW VAE.

    from mw_framework import MWDetector
    det = MWDetector(config="your_domain.json").fit(normal_df)   # or vae=FlexibleRobotVAE()
    det.predict(test_df)          #  1 normal, -1 anomaly   (sklearn convention)
    det.risk_score(test_df)       #  0-100, higher = more anomalous
    det.uncertainty(test_df)      #  MC-dropout std of the MW distance
    det.feature_attribution(df)   #  which features drove the score
    det.explain(df.iloc[[0]])     #  human-readable breakdown of one row

Three ways to say what the domain is:
    MWDetector(config=...)        JSON config: features + physics constraints
    MWDetector(vae=MyPhysicsVAE(...))  your own MWVAE subclass
    MWDetector()                  no physics, scales fitted from data (baseline)

Sign conventions follow scikit-learn's OutlierMixin:
    score_samples / decision_function : HIGHER = MORE NORMAL
    mw_distance / risk_score          : HIGHER = MORE ANOMALOUS
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import pyro
import torch
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.utils.validation import check_is_fitted

from .checkpoint import load_checkpoint, save_checkpoint
from .config import build_from_config
from .core import (MWNormaliser, MWTrainer, MWVAE, calibrate, get_device,
                   risk_score, risk_status, set_seed)


class MWDetector(BaseEstimator, OutlierMixin):

    def __init__(self, config: Optional[Union[str, Dict]] = None,
                 vae: Optional[MWVAE] = None,
                 latent_dim: int = 4, hidden_dim: int = 64, dropout: float = 0.1,
                 epochs: Optional[int] = None, lr: Optional[float] = None,
                 batch_size: Optional[int] = None,
                 calibration_split: float = 0.2, calibration_percentile: float = 99.0,
                 mc_samples: int = 10, random_state: int = 0, verbose: bool = False):
        self.config = config
        self.vae = vae
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.calibration_split = calibration_split
        self.calibration_percentile = calibration_percentile
        self.mc_samples = mc_samples
        self.random_state = random_state
        self.verbose = verbose

    # ── fit ──────────────────────────────────────────────────────────────────
    def fit(self, X, y=None):
        """
        Train on NORMAL data only (y is ignored).

        WHY normal-only?
            Anomalies are rare, diverse, and often unseen (zero-day attacks,
            new failure modes). Learn the one thing you have plenty of -
            normal physics - and flag anything that departs from it.
        """
        set_seed(self.random_state)
        pyro.clear_param_store()
        self.domain_ = "custom"

        if self.config is not None:
            vae, norm, cfg = build_from_config(self.config)
            self.domain_ = cfg["domain"]
        elif self.vae is not None:
            vae = copy.deepcopy(self.vae)
            vae._pyro_name = f"mw_vae_{id(vae)}"
            norm = MWNormaliser(vae.norm_mean.cpu().numpy(), vae.norm_std.cpu().numpy(),
                                vae.feature_names)
            if np.allclose(norm.std, 1.0) and np.allclose(norm.mean, 0.0):
                Xr = self._to_numpy(X, vae.feature_names)
                norm = MWNormaliser.fit(Xr, vae.feature_names)
                vae.set_normaliser(norm)
        else:
            names = list(X.columns) if isinstance(X, pd.DataFrame) else \
                [f"f{i}" for i in range(np.asarray(X).shape[1])]
            norm = MWNormaliser.fit(self._to_numpy(X, names), names)
            vae = MWVAE(n_features=len(names), latent_dim=self.latent_dim,
                        hidden_dim=self.hidden_dim, dropout=self.dropout,
                        feature_names=names, norm_mean=norm.mean.tolist(),
                        norm_std=norm.std.tolist())

        self.feature_names_ = list(norm.feature_names)
        self.norm_ = norm
        Xn = norm.transform(self._to_numpy(X, self.feature_names_))

        rng = np.random.default_rng(self.random_state)
        idx = rng.permutation(len(Xn))
        n_cal = int(len(Xn) * self.calibration_split)
        cal_idx, tr_idx = idx[:n_cal], idx[n_cal:]
        if n_cal < 10:                       # tiny data: calibrate on train
            cal_idx = tr_idx

        # explicit arguments win; then the JSON config; then defaults
        tcfg = cfg["training"] if self.config is not None else {}
        epochs = self.epochs or tcfg.get("epochs", 200)
        lr = self.lr or tcfg.get("lr", 1e-3)
        bs = self.batch_size or tcfg.get("batch_size", 128)

        trainer = MWTrainer(vae, lr=lr, epochs=epochs, batch_size=bs,
                            verbose=self.verbose)
        self.loss_curve_ = trainer.fit(Xn[tr_idx])
        self.vae_ = trainer.vae
        self.calibration_ = calibrate(self.vae_, Xn[cal_idx],
                                      self.calibration_percentile, self.mc_samples)
        self.is_fitted_ = True
        return self

    # ── raw MW quantities ────────────────────────────────────────────────────
    def mw_distance(self, X) -> np.ndarray:
        """Raw MW distance (higher = further from the normal manifold)."""
        return self._distance(X)[0]

    def uncertainty(self, X) -> np.ndarray:
        """MC-dropout std of the MW distance. High = model unsure; get more data."""
        return self._distance(X)[1]

    van_vleck_uncertainty = uncertainty     # name used in the original research code

    def risk_score(self, X) -> np.ndarray:
        """0-100. 50 = at the calibrated threshold."""
        return risk_score(self.mw_distance(X), self.calibration_)

    def status(self, X) -> np.ndarray:
        return risk_status(self.risk_score(X))

    def feature_attribution(self, X) -> pd.DataFrame:
        check_is_fitted(self, "is_fitted_")
        c = self.vae_.feature_contributions(self._tensor(X)).cpu().numpy()
        return pd.DataFrame(c, columns=self.feature_names_,
                            index=X.index if isinstance(X, pd.DataFrame) else None)

    def explain(self, X, top_k: int = 5) -> List[Dict]:
        """
        Per-row: risk, status, uncertainty, and the top-k driving features.
        `drivers` = each feature's share (%) of the off-manifold error.
        """
        d, u = self._distance(X)
        r = risk_score(d, self.calibration_)
        attr = self.feature_attribution(X).values
        out = []
        for i in range(len(d)):
            top = np.argsort(attr[i])[::-1][:top_k]
            share = 100.0 * attr[i] / max(attr[i].sum(), 1e-12)
            out.append({"risk": round(float(r[i]), 1),
                        "status": str(risk_status(r[i])[0]),
                        "mw_distance": round(float(d[i]), 4),
                        "uncertainty": round(float(u[i]), 4),
                        "drivers": {self.feature_names_[j]: f"{share[j]:.0f}%"
                                    for j in top}})
        return out

    # ── sklearn interface ────────────────────────────────────────────────────
    def score_samples(self, X) -> np.ndarray:
        return -self.mw_distance(X)

    def decision_function(self, X) -> np.ndarray:
        return self.calibration_.threshold - self.mw_distance(X)

    def predict(self, X) -> np.ndarray:
        return np.where(self.decision_function(X) < 0, -1, 1)

    # ── persistence ──────────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        check_is_fitted(self, "is_fitted_")
        save_checkpoint(path, self.vae_, self.norm_, self.calibration_, self.domain_)

    @classmethod
    def load(cls, path: str, mc_samples: int = 10) -> "MWDetector":
        vae, norm, cal, meta = load_checkpoint(path)
        det = cls(mc_samples=mc_samples)
        det.vae_ = vae.to(get_device())
        det.norm_, det.calibration_ = norm, cal
        det.feature_names_ = list(norm.feature_names)
        det.domain_ = meta["domain"]
        det.is_fitted_ = True
        return det

    # ── helpers ──────────────────────────────────────────────────────────────
    @staticmethod
    def _to_numpy(X, names) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            missing = [n for n in names if n not in X.columns]
            if missing:
                raise ValueError(f"input is missing columns: {missing}")
            return X[names].to_numpy(dtype=np.float32)
        return np.asarray(X, dtype=np.float32)

    def _tensor(self, X) -> torch.Tensor:
        Xn = self.norm_.transform(self._to_numpy(X, self.feature_names_))
        return torch.as_tensor(Xn, dtype=torch.float32,
                               device=next(self.vae_.parameters()).device)

    def _distance(self, X):
        check_is_fitted(self, "is_fitted_")
        m, s = self.vae_.mw_distance(self._tensor(X), self.mc_samples)
        return m.cpu().numpy(), s.cpu().numpy()
