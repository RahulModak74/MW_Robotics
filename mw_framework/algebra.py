"""
mw_framework.algebra
====================
Manifold algebra: build one big manifold from small, physically meaningful
pieces - instead of one monolithic VAE over every sensor.

Four operations (MW Region 2):

  PRODUCT       M = A x B
                A and B are independent. Separate latents, separate VAEs.
                p(x_A, x_B) = p(x_A) p(x_B)

  FIBRE BUNDLE  M = F -> B   ("F is a fibre over base B")
                The shape of F depends on where you are on B.
                p(x_F | z_B): the fibre's encoder AND decoder see z_B.
                e.g. cell temperature (fibre) depends on electrical load (base).

  GLUING MAP    A ~ B along a physical law
                A law that couples features owned by different pieces.
                e.g. Joule heating: temp_rise = k * I^2 R / cooling
                Added as a soft constraint on the decoded features.

  MARGINALISE   integrate out pieces you cannot observe
                Score only the sub-manifolds you have sensors for.
                Missing bases are integrated over their prior N(0, I).

WHY decompose?
    * Each piece is small: cost is sum_i O(n_i) = O(n).
    * Each piece is interpretable: "the THERMAL fibre is off" beats
      "something is off".
    * Pieces can be trained, reused or swapped independently.
    * Partial sensor sets still work (marginalisation).

Usage
-----
    alg = ManifoldAlgebra(feature_names, normaliser)
    alg.product("electrical", ["current_a", "voltage_v", ...], latent_dim=2)
    alg.fibre("thermal", ["temp_cell_c", ...], latent_dim=1, over="electrical")
    alg.product("aging", ["cycle_count", "soh"], latent_dim=1)
    alg.glue("joule", lambda x: x["temp_cell_c"] - ..., tolerance=1.0)
    MWTrainer(alg, ...).fit(X_norm)
    total, std = alg.mw_distance(X)
    parts      = alg.distance_breakdown(X)        # per sub-manifold
    thermal    = alg.marginal_distance(X, keep=["thermal"])
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine

from .core import MWNormaliser, TIER_SIGMA, constraint, data_scale


def _mlp(dims: Sequence[int], dropout: float) -> nn.Sequential:
    layers: List[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers += [nn.LayerNorm(dims[i + 1]), nn.SiLU(), nn.Dropout(dropout)]
    return nn.Sequential(*layers)


class SubVAE(nn.Module):
    """A small VAE for one sub-manifold. hidden defaults to max(32, 4 * n_in); pass `hidden` to override."""

    def __init__(self, n_in: int, latent_dim: int, cond_dim: int = 0, dropout: float = 0.1,
                 hidden: Optional[int] = None):
        super().__init__()
        h = hidden or max(32, 4 * n_in)
        self.n_in, self.latent_dim, self.cond_dim = n_in, latent_dim, cond_dim
        self.enc = _mlp([n_in + cond_dim, h, h], dropout)
        self.mu, self.sig = nn.Linear(h, latent_dim), nn.Linear(h, latent_dim)
        self.dec = _mlp([latent_dim + cond_dim, h, h, n_in], dropout)

    def encode(self, x, cond=None):
        src, dev = x.device, self.mu.weight.device
        x = x.to(dev) if cond is None else torch.cat([x.to(dev), cond.to(dev)], -1)
        h = self.enc(x)
        return self.mu(h).to(src), (F.softplus(self.sig(h)) + 1e-4).to(src)

    def decode(self, z, cond=None):
        src, dev = z.device, self.mu.weight.device
        z = z.to(dev) if cond is None else torch.cat([z.to(dev), cond.to(dev)], -1)
        return self.dec(z).to(src)


@dataclass
class _Piece:
    name: str
    features: List[str]
    latent_dim: int
    over: Optional[str]
    idx: List[int] = field(default_factory=list)


class ManifoldAlgebra(nn.Module):

    def __init__(self, feature_names: List[str], normaliser: Optional[MWNormaliser] = None,
                 obs_sigma: Optional[Sequence[float]] = None, dropout: float = 0.1,
                 kl_weight: float = 0.1):
        super().__init__()
        self.feature_names = list(feature_names)
        self.kl_weight, self.dropout = kl_weight, dropout
        n = len(feature_names)
        mean = normaliser.mean if normaliser is not None else [0.0] * n
        std = normaliser.std if normaliser is not None else [1.0] * n
        self.register_buffer("norm_mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("norm_std", torch.as_tensor(std, dtype=torch.float32))
        self.register_buffer("obs_sigma", torch.as_tensor(
            obs_sigma if obs_sigma is not None else [TIER_SIGMA[2]] * n, dtype=torch.float32))
        self.pieces: Dict[str, _Piece] = {}
        self.vaes = nn.ModuleDict()
        self.glues: List[Tuple[str, Callable, float, str]] = []
        self._pyro_name = f"mw_algebra_{id(self)}"

    # ── building the algebra ─────────────────────────────────────────────────
    def _add(self, name, features, latent_dim, over, hidden=None):
        if name in self.pieces:
            raise ValueError(f"sub-manifold '{name}' already exists")
        if over is not None and over not in self.pieces:
            raise ValueError(f"base '{over}' must be declared before fibre '{name}'")
        owned = {f for p in self.pieces.values() for f in p.features}
        clash = [f for f in features if f in owned]
        if clash:
            raise ValueError(f"features {clash} already belong to another sub-manifold")
        idx = [self.feature_names.index(f) for f in features]
        cond = self.pieces[over].latent_dim if over else 0
        self.pieces[name] = _Piece(name, list(features), latent_dim, over, idx)
        self.vaes[name] = SubVAE(len(features), latent_dim, cond, self.dropout, hidden)
        return self

    def product(self, name: str, features: List[str], latent_dim: int, hidden: Optional[int] = None):
        """Independent factor:  M = ... x name"""
        return self._add(name, features, latent_dim, None, hidden)

    def fibre(self, name: str, features: List[str], latent_dim: int, over: str, hidden: Optional[int] = None):
        """Fibre over a base: the fibre's geometry depends on the base point."""
        return self._add(name, features, latent_dim, over, hidden)

    def glue(self, name: str, fn: Callable[[Dict[str, torch.Tensor]], torch.Tensor],
             tolerance: float, kind: str = "normal"):
        """Gluing map: a physical law across sub-manifolds, in physical units."""
        self.glues.append((name, fn, tolerance, kind))
        return self

    def check_complete(self) -> None:
        owned = [f for p in self.pieces.values() for f in p.features]
        missing = [f for f in self.feature_names if f not in owned]
        if missing:
            raise ValueError(f"features not assigned to any sub-manifold: {missing}")

    def describe(self) -> str:
        lines = []
        for p in self.pieces.values():
            kind = f"FIBRE over {p.over}" if p.over else "PRODUCT factor"
            lines.append(f"  {p.name:<12} {kind:<24} latent={p.latent_dim}  "
                         f"features={p.features}")
        for g in self.glues:
            lines.append(f"  {g[0]:<12} GLUING MAP               tol={g[2]}")
        return "\n".join(lines)

    # ── helpers ──────────────────────────────────────────────────────────────
    def to_physical(self, x_norm: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = (x_norm.to(self.norm_std.device) * self.norm_std + self.norm_mean).to(x_norm.device)
        return {n: x[..., i] for i, n in enumerate(self.feature_names)}

    # ── Pyro model / guide ───────────────────────────────────────────────────
    def model(self, x: torch.Tensor) -> None:
        self.check_complete()
        pyro.module(self._pyro_name, self)
        B = x.shape[0]
        x_hat = torch.zeros_like(x)
        z: Dict[str, torch.Tensor] = {}
        with pyro.plate("data", B), poutine.scale(scale=data_scale(self, B)):
            for p in self.pieces.values():
                cond = z[p.over] if p.over else None
                z[p.name] = pyro.sample(f"z_{p.name}", dist.Normal(
                    x.new_zeros(B, p.latent_dim), x.new_ones(B, p.latent_dim)).to_event(1))
                xh = self.vaes[p.name].decode(z[p.name], cond)
                pyro.sample(f"x_{p.name}", dist.Normal(xh, self.obs_sigma[p.idx]).to_event(1),
                            obs=x[:, p.idx])
                x_hat[:, p.idx] = xh
            phys = self.to_physical(x_hat)
            for name, fn, tol, kind in self.glues:
                constraint(f"glue_{name}", fn(phys), tol, kind)

    def guide(self, x: torch.Tensor) -> None:
        pyro.module(self._pyro_name, self)
        z: Dict[str, torch.Tensor] = {}
        B = x.shape[0]
        with pyro.plate("data", B), poutine.scale(scale=data_scale(self, B)):
            for p in self.pieces.values():
                cond = z[p.over] if p.over else None
                mu, sig = self.vaes[p.name].encode(x[:, p.idx], cond)
                z[p.name] = pyro.sample(f"z_{p.name}", dist.Normal(mu, sig).to_event(1))

    # ── MW distance ──────────────────────────────────────────────────────────
    def _piece_distance(self, p: _Piece, x: torch.Tensor, cond) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (distance, mu) for one sub-manifold."""
        xp = x[:, p.idx]
        mu, sig = self.vaes[p.name].encode(xp, cond)
        xh = self.vaes[p.name].decode(mu, cond)
        recon = (((xp - xh) / self.obs_sigma[p.idx]) ** 2).mean(-1)
        kl = 0.5 * (mu ** 2 + sig ** 2 - 2 * torch.log(sig) - 1).sum(-1)
        return recon + self.kl_weight * kl, mu

    def _breakdown_once(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z, out = {}, {}
        for p in self.pieces.values():
            out[p.name], z[p.name] = self._piece_distance(p, x, z[p.over] if p.over else None)
        return out

    def distance_breakdown(self, x: torch.Tensor, mc_samples: int = 10) -> Dict[str, torch.Tensor]:
        """Mean MW distance of each sub-manifold separately."""
        src, x = x.device, x.to(self.obs_sigma.device)
        was = self.training
        self.train(self.dropout > 0)
        with torch.no_grad():
            runs = [self._breakdown_once(x) for _ in range(max(mc_samples, 1))]
        self.train(was)
        return {k: torch.stack([r[k] for r in runs]).mean(0).to(src) for k in runs[0]}

    def mw_distance(self, x: torch.Tensor, mc_samples: int = 10):
        """
        Total = feature-weighted sum of sub-manifold distances, so it is on
        the same scale as a monolithic MW VAE. Returns (mean, std).
        """
        n = len(self.feature_names)
        src, x = x.device, x.to(self.obs_sigma.device)
        was = self.training
        self.train(self.dropout > 0)
        with torch.no_grad():
            tots = []
            for _ in range(max(mc_samples, 1)):
                b = self._breakdown_once(x)
                tots.append(sum(b[p.name] * len(p.idx) / n for p in self.pieces.values()))
            s = torch.stack(tots)
        self.train(was)
        return s.mean(0).to(src), s.std(0, unbiased=False).to(src)

    def marginal_distance(self, x: torch.Tensor, keep: List[str],
                          n_base_samples: int = 64) -> torch.Tensor:
        """
        MARGINALISATION. Score using only the sub-manifolds in `keep`.
        Columns of other sub-manifolds may be NaN (sensors absent).

        If a kept fibre's base is not kept, the base latent is unknown.
        We integrate it out:  d = -log E_{z_B ~ N(0,I)} exp(-d(x_F | z_B))
        (a soft-minimum over plausible base states, by Monte Carlo).
        """
        for k in keep:
            if k not in self.pieces:
                raise ValueError(f"unknown sub-manifold '{k}'")
        src = x.device
        x = torch.nan_to_num(x, nan=0.0).to(self.obs_sigma.device)
        was = self.training
        self.eval()
        B, total, nfeat = x.shape[0], torch.zeros(x.shape[0], device=x.device), 0
        z: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for p in self.pieces.values():          # declaration order = base first
                if p.name not in keep:
                    continue
                nfeat += len(p.idx)
                if p.over is None:
                    d, z[p.name] = self._piece_distance(p, x, None)
                elif p.over in z:
                    d, z[p.name] = self._piece_distance(p, x, z[p.over])
                else:
                    zb = torch.randn(n_base_samples, B, self.pieces[p.over].latent_dim,
                                     device=x.device)
                    ds, mus = [], []
                    for k in range(n_base_samples):
                        dk, mk = self._piece_distance(p, x, zb[k])
                        ds.append(dk); mus.append(mk)
                    ds = torch.stack(ds)
                    d = -(torch.logsumexp(-ds, 0) - math.log(n_base_samples))
                    z[p.name] = torch.stack(mus)[ds.argmin(0), torch.arange(B)]
                total = total + d * len(p.idx)
        self.train(was)
        return (total / max(nfeat, 1)).to(src)
