"""
policy_3d.py - shared code for the 3D (MuJoCo) policy pipeline.

  features(data)       generic 3D state features (qpos[2:] + qvel; x,y world position dropped)
  LinearPolicy         a = hi * tanh(W f + b)   (the CEM teacher, same family as cem_policy_search.py)
  CVAE                 conditional VAE p(a | z, s) with prior p(z | s); the policy object that app_3d serves
  make_env(name)       Humanoid-v5 / Ant-v5 with a fixed config
"""
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_env(name="Humanoid-v5", **kw):
    return gym.make(name, **kw)


def features(data):
    return np.concatenate([data.qpos[2:], data.qvel]).astype(np.float64)


def feat_dim(env):
    d = env.unwrapped.data
    return len(features(d))


class LinearPolicy:
    def __init__(self, theta, fdim, adim, hi):
        self.W = theta[:adim * fdim].reshape(adim, fdim)
        self.b = theta[adim * fdim:adim * fdim + adim]
        self.hi = hi

    def __call__(self, f):
        return self.hi * np.tanh(self.W @ f + self.b)


class CVAE(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim=8, hidden=128):
        super().__init__()
        self.state_dim, self.action_dim, self.latent_dim = state_dim, action_dim, latent_dim
        mlp = lambda i: nn.Sequential(nn.Linear(i, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.enc, self.prior = mlp(state_dim + action_dim), mlp(state_dim)
        self.enc_mu, self.enc_sig = nn.Linear(hidden, latent_dim), nn.Linear(hidden, latent_dim)
        self.prior_mu, self.prior_sig = nn.Linear(hidden, latent_dim), nn.Linear(hidden, latent_dim)
        self.dec = nn.Sequential(nn.Linear(latent_dim + state_dim, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, action_dim), nn.Tanh())
        # state normalisation (set at train time, saved with the model)
        self.register_buffer("s_mean", torch.zeros(state_dim))
        self.register_buffer("s_std", torch.ones(state_dim))

    def _n(self, s):
        return (s - self.s_mean) / self.s_std

    def encode(self, s, a):
        h = self.enc(torch.cat([self._n(s), a], -1))
        return self.enc_mu(h), F.softplus(self.enc_sig(h)) + 1e-3

    def prior_params(self, s):
        h = self.prior(self._n(s))
        return self.prior_mu(h), F.softplus(self.prior_sig(h)) + 1e-3

    def decode(self, z, s):
        return self.dec(torch.cat([z, self._n(s)], -1))

    def forward(self, s, a, beta=0.01):
        mq, sq = self.encode(s, a)
        z = mq + sq * torch.randn_like(sq)
        recon = ((a - self.decode(z, s)) ** 2).sum(-1).mean()
        mp, sp = self.prior_params(s)
        kl = 0.5 * (sq**2 / sp**2 + (mq - mp)**2 / sp**2 - 2 * torch.log(sq / sp) - 1).sum(-1).mean()
        return recon + beta * kl, recon, kl

    @torch.no_grad()
    def act(self, s, stochastic=False):
        mp, sp = self.prior_params(s)
        z = mp + sp * torch.randn_like(sp) if stochastic else mp
        return self.decode(z, s)
