"""student_net.py - plain-PyTorch inference modules for the G1 tracker students (no Pyro / mw_framework needed at run time).

MLPStudent     standardised input -> 3 hidden layers -> action
BundleStudent  the MW fibre bundle at inference: state base z = mu(enc(x)); action fibre a = dec([0, z])  (one forward pass)
The BundleStudent layers mirror mw_framework.algebra.SubVAE so trained weights load directly.
"""
import torch
import torch.nn as nn


def mlp(dims):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers += [nn.LayerNorm(dims[i + 1]), nn.SiLU(), nn.Dropout(0.0)]
    return nn.Sequential(*layers)


class MLPStudent(nn.Module):
    kind = "mlp"

    def __init__(self, n_in=349, n_out=29, hidden=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, n_out))
        for n, d in (("xm", n_in), ("xs", n_in), ("ym", n_out), ("ys", n_out)):
            self.register_buffer(n, torch.zeros(d) if n.endswith("m") else torch.ones(d))

    def forward(self, x):
        return self.net(((x - self.xm) / self.xs).clamp(-8, 8)) * self.ys + self.ym


class BundleStudent(nn.Module):
    kind = "bundle"

    def __init__(self, n_in=349, n_out=29, latent_s=64, latent_a=2, hidden=512):
        super().__init__()
        self.latent_a = latent_a
        self.enc = mlp([n_in, hidden, hidden]); self.mu = nn.Linear(hidden, latent_s)
        self.dec = mlp([latent_a + latent_s, hidden, hidden, n_out])
        for n, d in (("xm", n_in), ("xs", n_in), ("ym", n_out), ("ys", n_out)):
            self.register_buffer(n, torch.zeros(d) if n.endswith("m") else torch.ones(d))

    def forward(self, x):
        z = self.mu(self.enc(((x - self.xm) / self.xs).clamp(-8, 8)))
        a = self.dec(torch.cat([torch.zeros(x.shape[0], self.latent_a, device=x.device, dtype=x.dtype), z], -1))
        return a * self.ys + self.ym


def build(kind, **kw):
    return {"mlp": MLPStudent, "bundle": BundleStudent}[kind](**kw)


def load(path, device="cpu"):
    ck = torch.load(path, map_location="cpu")
    m = build(ck["kind"], **ck["cfg"])
    m.load_state_dict(ck["state"])
    return m.to(device).eval()
