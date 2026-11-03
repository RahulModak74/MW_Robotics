"""
mw_bundle_policy.py - the MW policy as a FIBRE BUNDLE (mw_framework.ManifoldAlgebra).

    base   "state"   : the robot's posture/velocity features,         latent z_s   (product factor)
    fibre  "action"  : the 17 actuator torques OVER the state base,   latent z_a   (the fibre's encoder AND decoder see z_s)
    glue   physics   : torso height envelope, unit quaternion, joint limits, torque limits (+ optional angular-speed and
                       left/right-hip laws) as soft laws on the decoded features

Why a bundle rather than one flat VAE over (s, a): a flat model has to be INVERTED to get an action from a state (find a
latent point whose decoded state matches, then read the action block), and that inversion loses information. In the bundle
the state is the base, so the policy is a single forward pass -
    z_s = encode_state(s);   a = decode_action(z_a = 0, z_s)
- while keeping what makes it MW: a generative manifold trained by Pyro SVI with physics laws in the objective, a per-step
MW distance for any (s, a), and per-piece distances (is the STATE off-manifold, or is the ACTION?).
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
from mw_framework import ManifoldAlgebra, MWNormaliser
from policy_3d import features


def _viol(v, lo=None, hi=None):
    r = torch.zeros_like(v)
    if lo is not None:
        r = r + F.relu(lo - v)
    if hi is not None:
        r = r + F.relu(v - hi)
    return r


def build_bundle(spec, norm, latent_s=32, latent_a=2, hidden_s=256, hidden_a=256, sigma_s=0.15, sigma_a=0.2,
                 priors=True, tol=0.05, sym_pair=None, sym_tol=0.4):
    """ManifoldAlgebra with state base + action fibre. `spec` is mw_policy_3d.spec_from_env(env, "legacy")."""
    sd, ad, hi = spec["s_dim"], spec["a_dim"], spec["hi"]
    names = [f"s{i}" for i in range(sd)] + [f"a{i}" for i in range(ad)]
    alg = ManifoldAlgebra(names, norm, obs_sigma=[sigma_s] * sd + [sigma_a] * ad, dropout=0.0)
    alg.product("state", names[:sd], latent_s, hidden=hidden_s)
    alg.fibre("action", names[sd:], latent_a, over="state", hidden=hidden_a)
    if priors:
        alg.glue("p1_height", lambda x: _viol(x["s0"], spec["z_lo"], spec["z_hi"]), tol)
        alg.glue("p2_unit_quat", lambda x: sum(x[f"s{i}"] ** 2 for i in range(1, 5)) - 1.0, tol)
        for k, (lo, up) in enumerate(zip(spec["joint_lo"], spec["joint_hi"])):
            j = spec["joint_start"] + k
            alg.glue(f"p3_joint{k}", lambda x, j=j, lo=lo, up=up: _viol(x[f"s{j}"], lo, up), tol)
        for j in range(ad):
            alg.glue(f"p4_torque{j}", lambda x, j=j: _viol(x[f"a{j}"], -hi, hi), tol)
        if sym_pair:
            i, j = sym_pair
            alg.glue("p6_sym_hip", lambda x: x[f"a{i}"] + x[f"a{j}"], sym_tol * hi, kind="laplace")
    return alg


class BundlePolicy:
    """act(features) -> torques: one forward pass through the state base and the action fibre."""
    def __init__(self, alg, norm, spec, cfg):
        self.alg, self.norm, self.spec, self.cfg = alg.eval(), norm, spec, cfg
        self.sd, self.ad, self.hi = spec["s_dim"], spec["a_dim"], float(spec["hi"])
        self.mean = torch.as_tensor(norm.mean, dtype=torch.float32)
        self.std = torch.as_tensor(norm.std, dtype=torch.float32)
        self.features = features

    def reset(self):
        pass

    def observe(self, data):
        return self.features(data)

    @torch.no_grad()
    def act(self, f):
        s = ((torch.as_tensor(f, dtype=torch.float32) - self.mean[:self.sd]) / self.std[:self.sd]).clamp(-8, 8)[None]
        zs, _ = self.alg.vaes["state"].encode(s)                                   # base point
        la = self.alg.pieces["action"].latent_dim
        a_n = self.alg.vaes["action"].decode(torch.zeros(1, la), zs)               # fibre point at its prior mean
        a = a_n[0] * self.std[self.sd:] + self.mean[self.sd:]
        return a.clamp(-self.hi, self.hi).numpy().astype(np.float64)

    @torch.no_grad()
    def distances(self, f, a):
        """MW distance of (s, a) and its per-piece breakdown {'state': .., 'action': ..}."""
        x = torch.as_tensor(self.norm.transform(np.concatenate([f, a])[None].astype(np.float32)), dtype=torch.float32)
        tot, _ = self.alg.mw_distance(x, mc_samples=1)
        br = self.alg.distance_breakdown(x, mc_samples=1)
        return float(tot[0]), {k: float(v[0]) for k, v in br.items()}

    def save(self, path):
        torch.save({"state": {k: v.cpu() for k, v in self.alg.state_dict().items()}, "spec": self.spec, "cfg": self.cfg,
                    "norm": self.norm.to_dict()}, path)

    @classmethod
    def load(cls, path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        norm = MWNormaliser.from_dict(ck["norm"])
        alg = build_bundle(ck["spec"], norm, **ck["cfg"])
        alg.load_state_dict(ck["state"])
        return cls(alg, norm, ck["spec"], ck["cfg"])


# ---------------------------------------------------------------- held-out evaluation (CPU workers)
def _cpu_only():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(1)


def _eval_one(args):
    path, seed, max_steps, env_name = args
    torch.set_num_threads(1)
    from policy_3d import make_env
    pol = BundlePolicy.load(path)
    env = make_env(env_name); env.reset(seed=seed); d = env.unwrapped.data
    n = 0
    for _ in range(max_steps):
        _, _, term, trunc, _ = env.step(pol.act(pol.observe(d))); n += 1
        if term or trunc:
            break
    return n, float(d.qpos[0])


def evaluate_bundle(path, env_name="Humanoid-v5", n_eval=10, max_steps=500, workers=10, first_seed=5000):
    import multiprocessing as mp
    jobs = [(path, s, max_steps, env_name) for s in range(first_seed, first_seed + n_eval)]
    with mp.get_context("spawn").Pool(min(workers, n_eval), initializer=_cpu_only) as pool:
        ev = pool.map(_eval_one, jobs)
    return {"steps": float(np.mean([e[0] for e in ev])), "x": float(np.mean([e[1] for e in ev])),
            "per_seed_steps": [e[0] for e in ev], "n_eval": n_eval}
