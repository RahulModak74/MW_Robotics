"""student_prob.py - probabilistic G1 tracker students (Retry_MW_distillation.md), plain PyTorch at run time (no Pyro).

    GaussMLP     R1  trunk -> mean action + log-variance (aleatoric)
    Ensemble     R2  K GaussMLPs; epistemic = spread of their means
    MWBundleV2   R4  encoder q(z|o) -> state manifold (obs decoder, MW distance) ; action head on [o, z] (skip path)
    LastLayer    R3/R5 wrapper: exact last-layer Laplace on the mean head (Gaussian likelihood => closed form), one pass

Every model: forward(x) -> action (denormalised), used for control; uq(x) -> dict of per-step signals:
    action, sig_ale (29), sig_epi (29), mw_dist (scalar or nan), law_range, law_rate
Load with load(path); `kind` in the checkpoint picks the class.
"""
import math
import torch
import torch.nn as nn

N_IN, N_OUT, PREV0 = 349, 29, 64 + 256


def trunk(n_in, hidden, depth=3):
    layers, d = [], n_in
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), nn.SiLU()]
        d = hidden
    return nn.Sequential(*layers)


class _Norm(nn.Module):
    def __init__(self, n_in=N_IN, n_out=N_OUT):
        super().__init__()
        for n, d in (("xm", n_in), ("xs", n_in), ("ym", n_out), ("ys", n_out)):
            self.register_buffer(n, torch.zeros(d) if n.endswith("m") else torch.ones(d))
        self.register_buffer("amax", torch.tensor(float("inf")))        # action-range law (raw units)
        self.register_buffer("tol_rate", torch.tensor(1.0))            # action-rate law tolerance (raw units)

    def set_norm(self, xm, xs, ym, ys, amax=None, tol_rate=None):
        for n, v in (("xm", xm), ("xs", xs), ("ym", ym), ("ys", ys)):
            getattr(self, n).copy_(torch.as_tensor(v, dtype=torch.float32))
        if amax is not None:
            self.amax.fill_(float(amax))
        if tol_rate is not None:
            self.tol_rate.fill_(float(tol_rate))

    def nx(self, x):
        return ((x - self.xm) / self.xs).clamp(-8, 8)

    def laws(self, x, a):
        rng = torch.relu(a.abs() - self.amax).sum(-1)
        rate = (a - x[:, PREV0:PREV0 + N_OUT]).abs().mean(-1) / self.tol_rate
        return rng, rate


class GaussMLP(_Norm):
    kind = "gauss"

    def __init__(self, n_in=N_IN, n_out=N_OUT, hidden=512):
        super().__init__(n_in, n_out)
        self.body = trunk(n_in, hidden)
        self.mean = nn.Linear(hidden, n_out)
        self.logvar = nn.Linear(hidden, n_out)

    def features(self, x):                     # x already normalised
        return self.body(x)

    def heads(self, xn):
        h = self.features(xn)
        return self.mean(h), self.logvar(h).clamp(-12, 4), h

    def forward(self, x):
        return self.heads(self.nx(x))[0] * self.ys + self.ym

    @torch.no_grad()
    def uq(self, x):
        mu, lv, _ = self.heads(self.nx(x))
        a = mu * self.ys + self.ym
        rng, rate = self.laws(x, a)
        nan = torch.full((x.shape[0],), float("nan"))
        return dict(action=a, sig_ale=(0.5 * lv).exp() * self.ys, sig_epi=torch.zeros_like(a), mw_dist=nan,
                    law_range=rng, law_rate=rate)


class MWBundleV2(_Norm):
    """State manifold + policy. z ~ q(z|o) (latent_s dims); obs decoder p(o|z) with learned per-dim sigma (the manifold);
    action head on [o_n, z] (skip path keeps precision). MW distance = mean standardised obs residual + KL(q(z|o) || N(0, I)) / n_in."""
    kind = "mwv2"

    def __init__(self, n_in=N_IN, n_out=N_OUT, hidden=512, latent_s=16):
        super().__init__(n_in, n_out)
        self.latent_s = latent_s
        self.enc = trunk(n_in, hidden, 2)
        self.z_mu, self.z_lv = nn.Linear(hidden, latent_s), nn.Linear(hidden, latent_s)
        self.dec_o = nn.Sequential(nn.Linear(latent_s, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, n_in))
        self.o_logsig = nn.Parameter(torch.full((n_in,), math.log(0.3)))
        self.body = trunk(n_in + latent_s, hidden)
        self.mean = nn.Linear(hidden, n_out)
        self.logvar = nn.Linear(hidden, n_out)

    def encode(self, xn):
        h = self.enc(xn)
        return self.z_mu(h), self.z_lv(h).clamp(-10, 4)

    def act(self, xn, z):
        h = self.body(torch.cat([xn, z], -1))
        return self.mean(h), self.logvar(h).clamp(-12, 4), h

    def features(self, xn):
        mu_z, _ = self.encode(xn)
        return self.body(torch.cat([xn, mu_z], -1))

    def heads(self, xn):
        mu_z, _ = self.encode(xn)
        return self.act(xn, mu_z)

    def forward(self, x):
        return self.heads(self.nx(x))[0] * self.ys + self.ym

    def mw_distance(self, xn):
        mu_z, lv_z = self.encode(xn)
        r = (xn - self.dec_o(mu_z)) / self.o_logsig.exp()
        kl = 0.5 * (mu_z ** 2 + lv_z.exp() - 1 - lv_z).sum(-1)
        return (r ** 2).mean(-1) + kl / xn.shape[-1]

    @torch.no_grad()
    def uq(self, x):
        xn = self.nx(x)
        mu, lv, _ = self.heads(xn)
        a = mu * self.ys + self.ym
        rng, rate = self.laws(x, a)
        return dict(action=a, sig_ale=(0.5 * lv).exp() * self.ys, sig_epi=torch.zeros_like(a), mw_dist=self.mw_distance(xn),
                    law_range=rng, law_rate=rate)


class MWBundleV3(MWBundleV2):
    """MW v2 + a next-state (dynamics) head: predicts the next proprioception (64) from [o_n, z, a_n]. Used in training to put the
    physical priors (priors.py) on the consequences of the student's own action; at run time it can forecast law violations."""
    kind = "mwv3"

    def __init__(self, n_in=N_IN, n_out=N_OUT, hidden=512, latent_s=16, dyn_hidden=256):
        super().__init__(n_in, n_out, hidden, latent_s)
        self.dyn = nn.Sequential(nn.Linear(n_in + latent_s + n_out, dyn_hidden), nn.SiLU(), nn.Linear(dyn_hidden, dyn_hidden), nn.SiLU(),
                                 nn.Linear(dyn_hidden, 64))
        self.register_buffer("dm", torch.zeros(64)); self.register_buffer("ds", torch.ones(64))

    def predict_next(self, x, xn, z, a_raw, params=None):
        """next proprioception (raw units) for action a_raw; params: optional detached dyn parameters (laws must not train dyn)."""
        an = (a_raw - self.ym) / self.ys
        inp = torch.cat([xn, z, an], -1)
        if params is None:
            d = self.dyn(inp)
        else:
            d = torch.func.functional_call(self.dyn, params, (inp,))
        return x[:, :64] + self.dm + self.ds * d


class Ensemble(nn.Module):
    kind = "ensemble"

    def __init__(self, members):
        super().__init__()
        self.members = nn.ModuleList(members)

    def forward(self, x):
        return torch.stack([m(x) for m in self.members]).mean(0)

    @torch.no_grad()
    def uq(self, x):
        outs = [m.uq(x) for m in self.members]
        A = torch.stack([o["action"] for o in outs])
        a = A.mean(0)
        rng, rate = self.members[0].laws(x, a)
        ale = torch.stack([o["sig_ale"] ** 2 for o in outs]).mean(0).sqrt()
        mw = torch.stack([o["mw_dist"] for o in outs]).mean(0)
        return dict(action=a, sig_ale=ale, sig_epi=A.std(0), mw_dist=mw, law_range=rng, law_rate=rate)


class LastLayer(nn.Module):
    """Exact last-layer Laplace around the trained mean head. With a Gaussian likelihood the posterior of a linear last layer is
    Gaussian: Sigma_j = s_j^2 (Phi'Phi + lam I)^-1 (shared factor). Epistemic sd_j(x) = s_j * ||L^-1 h(x)||, one triangular solve."""
    kind = "lastlayer"

    def __init__(self, base, hidden=512, n_out=N_OUT):
        super().__init__()
        self.base = base
        self.register_buffer("Linv", torch.eye(hidden + 1))          # inverse Cholesky factor of (Phi'Phi + lam I), with bias column
        self.register_buffer("s", torch.ones(n_out))                # per-joint residual sd of the mean head (normalised units)

    @torch.no_grad()
    def fit(self, Xn_batches, Yn_batches, lam=1.0):
        d = self.Linv.shape[0]
        P = lam * torch.eye(d, dtype=torch.float64)
        sse, n = torch.zeros(self.s.shape[0], dtype=torch.float64), 0
        for xn, yn in zip(Xn_batches, Yn_batches):
            h = self.base.features(xn)
            phi = torch.cat([h, torch.ones(h.shape[0], 1, device=h.device)], -1).double().cpu()
            P += phi.T @ phi
            sse += ((self.base.mean(h) - yn) ** 2).double().cpu().sum(0); n += len(xn)
        L = torch.linalg.cholesky(P)
        self.Linv.copy_(torch.linalg.solve_triangular(L, torch.eye(d, dtype=torch.float64), upper=False).float())
        self.s.copy_((sse / n).sqrt().float())

    def forward(self, x):
        return self.base(x)

    @torch.no_grad()
    def uq(self, x):
        out = self.base.uq(x)
        h = self.base.features(self.base.nx(x))
        phi = torch.cat([h, torch.ones(h.shape[0], 1)], -1)
        q = ((phi @ self.Linv.T) ** 2).sum(-1, keepdim=True).sqrt()
        out["sig_epi"] = q * self.s * self.base.ys
        return out


def build(kind, cfg):
    if kind == "gauss":
        return GaussMLP(**cfg)
    if kind == "mwv2":
        return MWBundleV2(**cfg)
    if kind == "mwv3":
        return MWBundleV3(**cfg)
    if kind == "ensemble":
        return Ensemble([build(c["kind"], c["cfg"]) for c in cfg["members"]])
    if kind == "lastlayer":
        return LastLayer(build(cfg["base"]["kind"], cfg["base"]["cfg"]), hidden=cfg["base"]["cfg"].get("hidden", 512))
    raise ValueError(kind)


def save(model, cfg, path, extra=None):
    torch.save({"kind": model.kind, "cfg": cfg, "state": model.state_dict(), **(extra or {})}, path)


def load(path, device="cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = build(ck["kind"], ck["cfg"])
    m.load_state_dict(ck["state"])
    return m.to(device).eval()
