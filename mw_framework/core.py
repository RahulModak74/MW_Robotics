"""
mw_framework.core
=================
The heart of the Modak-Walawalkar (MW) Framework, in one file.

    1. MWNormaliser  - puts every feature on a "natural variability" scale
    2. MWVAE         - Bayesian VAE whose decoder is bound by physics laws
    3. MWTrainer     - stochastic variational inference (Pyro SVI) loop
    4. calibrate()   - learns what "normal distance" looks like
    5. risk_score()  - maps MW distance to a 0-100 risk number

WHY a VAE?
    Physics confines real systems to a thin, low-dimensional surface (a
    manifold) inside the high-dimensional sensor space. A VAE learns that
    surface: the encoder maps a reading onto it, the decoder maps back.
    A reading that is OFF the surface cannot be reconstructed well.
    That reconstruction gap, measured in noise units, is the MW distance.

WHY Bayesian (Pyro)?
    Physics laws are written as likelihood terms ("this residual should be
    ~0, give or take a tolerance"). Unknown physical constants become
    latent variables with priors. Pyro turns both into one ELBO objective.
    The model learns a surface that fits the data AND obeys the laws.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam

log = logging.getLogger("mw_framework")

ArrayLike = Union[np.ndarray, Sequence[Sequence[float]]]

# Observation-noise tiers, in normalised units.
#   Tier 1: directly measured, trusted sensor  -> manifold must pass close
#   Tier 2: primary derived quantity
#   Tier 3: inferred / soft quantity           -> manifold may pass loosely
TIER_SIGMA = {1: 0.05, 2: 0.10, 3: 0.30}


def get_device() -> torch.device:
    """CUDA if available. Set MW_DEVICE=cpu (or cuda:1, ...) to override."""
    import os
    forced = os.environ.get("MW_DEVICE")
    if forced:
        return torch.device(forced)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def module_device(module: nn.Module) -> torch.device:
    """Device holding a module's weights."""
    return next(module.parameters()).device


def set_seed(seed: int = 0) -> None:
    """Make numpy, torch and pyro reproducible."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)


# ─────────────────────────────────────────────────────────────────────────────
# 1. NORMALISER
# ─────────────────────────────────────────────────────────────────────────────

class MWNormaliser:
    """
    x_norm = (x - baseline) / sigma

    WHY physics baselines instead of data mean/std?
        If you train only on healthy data, the data std is the *healthy*
        spread. That is fine. But when the engineer already knows the
        nominal value and natural variability (e.g. LFP cell = 3.3 V +- 0.1 V),
        using those numbers makes the scale meaningful and stable across
        fleets. Use `from_physics` when you know them, `fit` when you don't.
    """

    def __init__(self, mean: ArrayLike, std: ArrayLike,
                 feature_names: Optional[List[str]] = None):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.maximum(np.asarray(std, dtype=np.float32), 1e-8)
        self.feature_names = list(feature_names) if feature_names else None

    @classmethod
    def from_physics(cls, baselines: Dict[str, float],
                     sigmas: Dict[str, float]) -> "MWNormaliser":
        names = list(baselines.keys())
        return cls([baselines[n] for n in names], [sigmas[n] for n in names], names)

    @classmethod
    def fit(cls, X: ArrayLike, feature_names: Optional[List[str]] = None,
            robust: bool = False) -> "MWNormaliser":
        """robust=True uses median / (IQR/1.349) - good for fat-tailed data."""
        X = np.asarray(X, dtype=np.float64)
        if robust:
            med = np.median(X, axis=0)
            q75, q25 = np.percentile(X, [75, 25], axis=0)
            return cls(med, (q75 - q25) / 1.349 + 1e-6, feature_names)
        return cls(X.mean(0), X.std(0) + 1e-6, feature_names)

    def transform(self, X: ArrayLike, clip: Optional[float] = 8.0) -> np.ndarray:
        Z = (np.asarray(X, dtype=np.float32) - self.mean) / self.std
        return np.clip(Z, -clip, clip) if clip else Z

    def inverse(self, Z: ArrayLike) -> np.ndarray:
        return np.asarray(Z, dtype=np.float32) * self.std + self.mean

    def to_dict(self) -> Dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist(),
                "feature_names": self.feature_names}

    @classmethod
    def from_dict(cls, d: Dict) -> "MWNormaliser":
        return cls(d["mean"], d["std"], d.get("feature_names"))


# ─────────────────────────────────────────────────────────────────────────────
# 2. PHYSICS AS LIKELIHOOD TERMS
# ─────────────────────────────────────────────────────────────────────────────

def constraint(name: str, residual: torch.Tensor, tolerance: float,
               kind: str = "normal", df: float = 2.0) -> None:
    """
    Declare a soft physics law:  residual ~ D(0, tolerance).

    Example (Ohm's law):  constraint("ohm", V - I*R, tolerance=0.05)

    WHY soft, not hard?
        Real sensors are noisy and laws are approximations. A tolerance says
        how much violation is physically plausible. Pyro adds
        log p(residual) to the ELBO, so violations cost training "energy".

    kind:
        "normal"   - ordinary laws, Gaussian tolerance
        "laplace"  - sparse, sharp violations allowed (shocks, switching)
        "studentt" - heavy tails: rare, violent events are *physical*
                     (disruptions, vortex cores). df controls tail weight.
    """
    tol = torch.as_tensor(tolerance, dtype=residual.dtype, device=residual.device)
    zero = torch.zeros((), dtype=residual.dtype, device=residual.device)
    if kind == "normal":
        d = dist.Normal(zero, tol)
    elif kind == "laplace":
        d = dist.Laplace(zero, tol)
    elif kind == "studentt":
        d = dist.StudentT(torch.as_tensor(df, dtype=residual.dtype,
                                          device=residual.device), zero, tol)
    else:
        raise ValueError(f"unknown constraint kind '{kind}'")
    pyro.sample(name, d.expand(residual.shape), obs=residual)


def bound(name: str, value: torch.Tensor, low: Optional[float] = None,
          high: Optional[float] = None, tolerance: float = 0.01) -> None:
    """
    Declare a physical range:  low <= value <= high  (softly).
    Example: bound("soh_range", soh, 0.0, 1.0)   # health is a fraction
    Zero cost inside the range; Gaussian cost outside it.
    """
    viol = torch.zeros_like(value)
    if low is not None:
        viol = viol + F.relu(low - value)
    if high is not None:
        viol = viol + F.relu(value - high)
    constraint(name, viol, tolerance)


def data_scale(module: nn.Module, batch_size: int) -> float:
    """
    Mini-batch correction N / B. Makes the ELBO an unbiased estimate of the
    full-data ELBO, so posteriors over GLOBAL physics constants (e.g. an
    activation energy) get the right width. The trainer sets `_n_total`.
    """
    return float(getattr(module, "_n_total", batch_size)) / float(batch_size)


# ─────────────────────────────────────────────────────────────────────────────
# 3. THE MW VAE
# ─────────────────────────────────────────────────────────────────────────────

class MWVAE(nn.Module):
    """
    Physics-constrained Bayesian VAE.

    Subclass it and override:
        physics(self, x)          -> call constraint()/bound() on decoded,
                                     PHYSICAL-unit features  (inside data plate)
        physics_globals(self)     -> optional: pyro.sample() unknown constants
        physics_globals_guide(self) -> optional: their variational posterior

    `x` passed to physics() is a dict {feature_name: tensor (B,)} in real
    units (Volts, deg C, ...), so laws are written exactly as in a textbook.
    """

    def __init__(self, n_features: int, latent_dim: int = 4, hidden_dim: int = 64,
                 dropout: float = 0.1, obs_sigma: Optional[Sequence[float]] = None,
                 feature_names: Optional[List[str]] = None,
                 norm_mean: Optional[Sequence[float]] = None,
                 norm_std: Optional[Sequence[float]] = None,
                 kl_weight: float = 0.1):
        super().__init__()
        self.n_features = n_features
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.kl_weight = kl_weight
        self.feature_names = list(feature_names) if feature_names else \
            [f"f{i}" for i in range(n_features)]
        self._pyro_name = f"mw_vae_{id(self)}"

        h, h2 = hidden_dim, max(hidden_dim // 2, latent_dim * 2)
        self.drop = nn.Dropout(dropout)
        self.enc1, self.enc2 = nn.Linear(n_features, h), nn.Linear(h, h2)
        self.enc_mu, self.enc_sig = nn.Linear(h2, latent_dim), nn.Linear(h2, latent_dim)
        self.dec1, self.dec2 = nn.Linear(latent_dim, h2), nn.Linear(h2, h)
        self.dec_out = nn.Linear(h, n_features)

        obs = obs_sigma if obs_sigma is not None else [TIER_SIGMA[2]] * n_features
        self.register_buffer("obs_sigma", torch.as_tensor(obs, dtype=torch.float32))
        self.register_buffer("norm_mean", torch.as_tensor(
            norm_mean if norm_mean is not None else [0.0] * n_features, dtype=torch.float32))
        self.register_buffer("norm_std", torch.as_tensor(
            norm_std if norm_std is not None else [1.0] * n_features, dtype=torch.float32))

    # ── bookkeeping ──────────────────────────────────────────────────────────
    def get_config(self) -> Dict:
        """Constructor kwargs - used to rebuild the model from a checkpoint."""
        return dict(n_features=self.n_features, latent_dim=self.latent_dim,
                    hidden_dim=self.hidden_dim, dropout=self.dropout,
                    obs_sigma=self.obs_sigma.tolist(), feature_names=self.feature_names,
                    norm_mean=self.norm_mean.tolist(), norm_std=self.norm_std.tolist(),
                    kl_weight=self.kl_weight)

    def set_normaliser(self, norm: MWNormaliser) -> None:
        self.norm_mean.copy_(torch.as_tensor(norm.mean))
        self.norm_std.copy_(torch.as_tensor(norm.std))

    def to_physical(self, x_norm: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = x_norm.to(self.norm_std.device) * self.norm_std + self.norm_mean
        x = x.to(x_norm.device)
        return {n: x[..., i] for i, n in enumerate(self.feature_names)}

    # ── networks ─────────────────────────────────────────────────────────────
    # DEVICE RULE: every public method accepts tensors on ANY device (CPU or
    # GPU), runs on the model's device, and returns results on the CALLER's
    # device. So a model trained on CUDA can be fed plain CPU tensors.
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        src = x.device
        x = x.to(self.enc1.weight.device)
        h = self.drop(F.selu(self.enc1(x)))
        h = self.drop(F.selu(self.enc2(h)))
        return self.enc_mu(h).to(src), (F.softplus(self.enc_sig(h)) + 1e-4).to(src)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        src = z.device
        z = z.to(self.dec1.weight.device)
        h = self.drop(F.selu(self.dec1(z)))
        h = self.drop(F.selu(self.dec2(h)))
        return self.dec_out(h).to(src)

    # ── physics hooks (override in subclasses) ───────────────────────────────
    def physics(self, x: Dict[str, torch.Tensor]) -> None:
        """Override: declare laws on decoded physical features."""

    def physics_globals(self) -> None:
        """Override: sample unknown physical constants (model side)."""

    def physics_globals_guide(self) -> None:
        """Override: variational posterior for those constants (guide side)."""

    # ── Pyro generative model and guide ──────────────────────────────────────
    def model(self, x: torch.Tensor) -> None:
        """
        p(z) = N(0, I)                      latent position on the manifold
        p(x | z) = N(decoder(z), obs_sigma) the manifold itself
        + physics(decoder(z))               the laws the manifold must obey
        """
        pyro.module(self._pyro_name, self)
        self.physics_globals()
        B = x.shape[0]
        with pyro.plate("data", B), poutine.scale(scale=data_scale(self, B)):
            z = pyro.sample("latent", dist.Normal(
                x.new_zeros(B, self.latent_dim), x.new_ones(B, self.latent_dim)).to_event(1))
            x_hat = self.decode(z)
            pyro.sample("obs", dist.Normal(x_hat, self.obs_sigma).to_event(1), obs=x)
            self.physics(self.to_physical(x_hat))

    def guide(self, x: torch.Tensor) -> None:
        """q(z | x) = N(encoder_mu(x), encoder_sigma(x))"""
        pyro.module(self._pyro_name, self)
        self.physics_globals_guide()
        B = x.shape[0]
        with pyro.plate("data", B), poutine.scale(scale=data_scale(self, B)):
            mu, sig = self.encode(x)
            pyro.sample("latent", dist.Normal(mu, sig).to_event(1))

    # ── MW distance ──────────────────────────────────────────────────────────
    def _distance_once(self, x: torch.Tensor) -> torch.Tensor:
        mu, sig = self.encode(x)
        x_hat = self.decode(mu)
        recon = (((x - x_hat) / self.obs_sigma) ** 2).mean(-1)
        kl = 0.5 * (mu ** 2 + sig ** 2 - 2 * torch.log(sig) - 1).sum(-1)
        return recon + self.kl_weight * kl

    def mw_distance(self, x: torch.Tensor, mc_samples: int = 10
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        MW distance = mean over features of ((x - x_hat)/obs_sigma)^2
                      + kl_weight * KL(q(z|x) || N(0,I))

        Term 1: how far the point sits OFF the manifold (normal direction).
        Term 2: how far ALONG the manifold it sits from the typical region.

        Returns (mean, std) over `mc_samples` passes with dropout ON.
        The std is the model's own uncertainty about the score (MC dropout).
        """
        src, x = x.device, x.to(self.obs_sigma.device)
        was_training = self.training
        self.train(self.dropout > 0)
        with torch.no_grad():
            s = torch.stack([self._distance_once(x) for _ in range(max(mc_samples, 1))])
        self.train(was_training)
        return s.mean(0).to(src), s.std(0, unbiased=False).to(src)

    def feature_contributions(self, x: torch.Tensor) -> torch.Tensor:
        """((x - x_hat)/obs_sigma)^2 per feature: WHICH sensor drove the score."""
        src, x = x.device, x.to(self.obs_sigma.device)
        was_training = self.training
        self.eval()
        with torch.no_grad():
            mu, _ = self.encode(x)
            out = ((x - self.decode(mu)) / self.obs_sigma) ** 2
        self.train(was_training)
        return out.to(src)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Project x onto the learned manifold (normalised units)."""
        was_training = self.training
        self.eval()
        with torch.no_grad():
            out = self.decode(self.encode(x)[0])
        self.train(was_training)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 4. TRAINER
# ─────────────────────────────────────────────────────────────────────────────

class MWTrainer:
    """
    Generic SVI loop. Works for anything with .model(x) and .guide(x)
    (MWVAE, its subclasses, and ManifoldAlgebra).
    """

    def __init__(self, vae: nn.Module, lr: float = 1e-3, epochs: int = 200,
                 batch_size: int = 128, clip_norm: float = 10.0,
                 verbose: bool = True, log_every: int = 50,
                 device: Optional[torch.device] = None):
        self.device = device or get_device()
        self.vae = vae.to(self.device)
        self.epochs, self.batch_size = epochs, batch_size
        self.verbose, self.log_every = verbose, log_every
        self.svi = SVI(vae.model, vae.guide,
                       ClippedAdam({"lr": lr, "clip_norm": clip_norm}),
                       loss=Trace_ELBO())

    def fit(self, X_norm: Union[np.ndarray, torch.Tensor]) -> List[float]:
        """X_norm: normalised NORMAL data only. Returns per-epoch loss."""
        X = torch.as_tensor(X_norm, dtype=torch.float32).to(self.device)
        n, losses, t0 = len(X), [], time.time()
        self.vae._n_total = n
        self.vae.train()
        for epoch in range(self.epochs):
            perm, tot, nb = torch.randperm(n, device=self.device), 0.0, 0
            for s in range(0, n, self.batch_size):
                batch = X[perm[s:s + self.batch_size]]
                if len(batch) < 2:
                    continue
                tot += self.svi.step(batch) / n      # loss per data point
                nb += 1
            losses.append(tot / max(nb, 1))
            if self.verbose and ((epoch + 1) % self.log_every == 0 or epoch == 0):
                print(f"    epoch {epoch + 1:>4}/{self.epochs}  "
                      f"loss/sample={losses[-1]:10.3f}  ({time.time() - t0:.1f}s)")
        self.vae.eval()
        return losses


# ─────────────────────────────────────────────────────────────────────────────
# 5. CALIBRATION AND RISK
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Calibration:
    """Distribution of MW distance on held-out NORMAL data."""
    mu: float
    sigma: float
    threshold: float          # raw distance at `percentile`
    percentile: float

    def to_dict(self) -> Dict:
        return asdict(self)


def calibrate(model: nn.Module, X_norm: Union[np.ndarray, torch.Tensor],
              percentile: float = 99.0, mc_samples: int = 10) -> Calibration:
    """
    WHY calibrate on HELD-OUT normal data?
        Training data is reconstructed better than unseen data (the model
        has seen it). Calibrating on it sets the threshold too low and
        floods you with false alarms. Hold out 20% of normal data.
    """
    X = torch.as_tensor(X_norm, dtype=torch.float32).to(module_device(model))
    d, _ = model.mw_distance(X, mc_samples=mc_samples)
    d = d.cpu().numpy()
    return Calibration(float(d.mean()), float(d.std() + 1e-8),
                       float(np.percentile(d, percentile)), percentile)


def risk_score(raw: Union[float, np.ndarray], cal: Calibration) -> np.ndarray:
    """
    0-100 risk.  50 = exactly at the calibrated threshold.
        risk = 100 * sigmoid((raw - threshold) / sigma_normal)
    So by construction ~(100 - percentile)% of normal points exceed 50.
    """
    z = (np.asarray(raw, dtype=np.float64) - cal.threshold) / max(cal.sigma, 1e-8)
    return 100.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


def risk_status(risk: Union[float, np.ndarray]) -> np.ndarray:
    """NORMAL < 50 <= MONITOR < 80 <= THREAT"""
    r = np.atleast_1d(risk)
    return np.where(r < 50, "NORMAL", np.where(r < 80, "MONITOR", "THREAT"))


def roc_auc(scores_normal: np.ndarray, scores_anomaly: np.ndarray) -> float:
    """AUC without sklearn: P(anomaly score > normal score)."""
    a = np.asarray(scores_anomaly)[:, None]
    n = np.asarray(scores_normal)[None, :]
    return float((a > n).mean() + 0.5 * (a == n).mean())
