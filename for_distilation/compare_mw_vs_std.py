#!/usr/bin/env python3
"""
compare_mw_vs_std.py - MW policy manifold vs standard learners, on the SAME teacher data and held-out seeds.

  A. Policy quality (held-out steps / forward distance):
       std MLP behaviour cloning | std CVAE (plain PyTorch) | MW manifold, z-query (current) |
       MW manifold, action chosen by MW distance (CEM, z-query warm start) | same from a zero start
  B. Distance as an early-warning signal: how well does each score, computed on the student's own rollouts, separate
     "about to fall (next 20 steps)" from "healthy"?   MW distance of (s, a)  vs  Mahalanobis(s)  vs  kNN distance(s)
     (AUROC; 'step index' is reported as a trivial baseline so a time trend is not mistaken for skill).

    python compare_mw_vs_std.py --data_cache data_cache/ppo_humanoid.npz --mw_ckpt checkpoints_3d/humanoid_ppo_e0_legacy_mw.pth
"""
import os, sys, json, argparse, time
os.environ.setdefault("MW_DEVICE", "cpu")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, torch, torch.nn as nn
import multiprocessing as mp
from policy_3d import make_env, features, CVAE
from mw_policy_3d import ManifoldPolicy

HERE = os.path.dirname(os.path.abspath(__file__))
HI = 0.4


class MLPPolicy(nn.Module):
    def __init__(self, fdim, adim, hidden=256):
        super().__init__()
        self.register_buffer("mean", torch.zeros(fdim)); self.register_buffer("std", torch.ones(fdim))
        self.net = nn.Sequential(nn.Linear(fdim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, adim), nn.Tanh())

    def forward(self, s):
        return self.net((s - self.mean) / self.std)


class MWDistanceQuery(ManifoldPolicy):
    """Pick the action that makes (s, a) most manifold-consistent: CEM over the 17 torques, state clamped, scored by the
    MW distance (reconstruction in sigma units + KL) - the 'inverse query' idea of mw_framework/query/inverse.py."""
    def __init__(self, det, pop=64, gens=4, std0=0.5, zero_init=False, seed=0):
        super().__init__(det)
        self.pop, self.gens, self.std0, self.zero_init = pop, gens, std0, zero_init
        self.rng = np.random.default_rng(seed)

    def act(self, f):
        sd, hi, adim = self.sd, float(self.vae.hi_), self.vae.a_dim_
        s = ((torch.as_tensor(f, dtype=torch.float32) - self.mean[:sd]) / self.std[:sd]).clamp(-8, 8)
        mean_a = np.zeros(adim) if self.zero_init else super().act(f)
        sd_a = np.full(adim, self.std0 * hi)
        n_el = max(2, self.pop // 6)
        for _ in range(self.gens):
            cand = np.clip(mean_a + sd_a * self.rng.standard_normal((self.pop, adim)), -hi, hi)
            cand[0] = mean_a
            a_n = (torch.as_tensor(cand, dtype=torch.float32) - self.mean[sd:]) / self.std[sd:]
            x = torch.cat([s.expand(self.pop, -1), a_n], 1)
            with torch.no_grad():
                d = self.vae._distance_once(x)
            el = cand[torch.argsort(d)[:n_el].numpy()]
            mean_a, sd_a = el.mean(0), el.std(0) + 1e-3
        return np.clip(mean_a, -hi, hi)


# ---------------------------------------------------------------- workers
_ref = {}


def _init_worker(cache):
    torch.set_num_threads(1)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    z = np.load(cache)
    S = z["S_legacy"]
    m, sdv = S.mean(0), S.std(0) + 1e-6
    Sn = ((S - m) / sdv).astype(np.float32)
    cov = np.cov(Sn[:50000].T) + 1e-3 * np.eye(Sn.shape[1])
    _ref.update(mean=m, std=sdv, prec=np.linalg.inv(cov), knn=torch.as_tensor(Sn[np.random.default_rng(0).choice(len(Sn), 20000, replace=False)]))


def _rollout(args):
    kind, path, seed, max_steps, record = args
    from mw_framework import MWDetector
    env = make_env("Humanoid-v5"); env.reset(seed=seed); d = env.unwrapped.data
    pol = det = None
    if kind in ("mlp", "cvae"):
        ck = torch.load(path, map_location="cpu")
        if kind == "mlp":
            model = MLPPolicy(ck["fdim"], ck["adim"], ck["hidden"]); model.load_state_dict(ck["state"]); model.eval()
            pol = lambda f: HI * model(torch.as_tensor(f[None], dtype=torch.float32))[0].detach().numpy()
        else:
            model = CVAE(ck["fdim"], ck["adim"], hidden=ck["hidden"]); model.load_state_dict(ck["state"]); model.eval()
            pol = lambda f: HI * model.act(torch.as_tensor(f[None], dtype=torch.float32)).numpy()[0]
    else:
        det = MWDetector.load(path)
        pol = {"mw_z": lambda: ManifoldPolicy(det), "mw_mwd": lambda: MWDistanceQuery(det),
               "mw_mwd0": lambda: MWDistanceQuery(det, zero_init=True)}[kind]().act
    trace, n = [], 0
    for _ in range(max_steps):
        f = features(d); a = pol(f)
        if record:
            x = torch.as_tensor(det.norm_.transform(np.concatenate([f, a])[None].astype(np.float32)), dtype=torch.float32)
            with torch.no_grad():
                md = float(det.vae_._distance_once(x)[0])
            zs = (f - _ref["mean"]) / _ref["std"]
            maha = float(np.sqrt(max(zs @ _ref["prec"] @ zs, 0)))
            knn = float(torch.cdist(torch.as_tensor(zs[None], dtype=torch.float32), _ref["knn"]).topk(5, largest=False).values[0, -1])
            trace.append((md, maha, knn))
        _, r, term, trunc, _ = env.step(a); n += 1
        if term or trunc:
            break
    return {"steps": n, "x": float(d.qpos[0]), "fell": bool(term), "trace": trace}


def _pool(cache, jobs, workers):
    with mp.get_context("spawn").Pool(min(workers, len(jobs)), initializer=_init_worker, initargs=(cache,)) as p:
        return p.map(_rollout, jobs)


# ---------------------------------------------------------------- training of the standard models
def train_std(cache, rows, epochs, hidden, out_dir):
    z = np.load(cache)
    S_all, A_all = z["S_legacy"], z["A"]                         # A is a / hi, in [-1, 1]
    idx = pd.Series(np.arange(len(S_all))).sample(rows, random_state=0).values     # the same rows train_mw_3d used
    S, A = S_all[idx], A_all[idx]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    St, At = torch.as_tensor(S, dtype=torch.float32, device=dev), torch.as_tensor(A, dtype=torch.float32, device=dev)
    mean, std = St.mean(0), St.std(0) + 1e-6
    fdim, adim = S.shape[1], A.shape[1]
    res = {}
    mlp = MLPPolicy(fdim, adim, hidden).to(dev); mlp.mean.copy_(mean); mlp.std.copy_(std)
    cv = CVAE(fdim, adim, hidden=hidden).to(dev); cv.s_mean.copy_(mean); cv.s_std.copy_(std)
    for name, model, step in (("mlp", mlp, lambda m, s, a: ((m(s) - a) ** 2).sum(-1).mean()),
                              ("cvae", cv, lambda m, s, a: m(s, a, beta=0.01)[0])):
        opt = torch.optim.Adam(model.parameters(), lr=1e-3); t0 = time.time()
        for ep in range(epochs):
            perm = torch.randperm(len(St), device=dev)
            for i in range(0, len(St), 512):
                b = perm[i:i + 512]
                loss = step(model, St[b], At[b]); opt.zero_grad(); loss.backward(); opt.step()
        path = os.path.join(out_dir, f"std_{name}.pt")
        torch.save({"state": {k: v.cpu() for k, v in model.state_dict().items()}, "fdim": fdim, "adim": adim,
                    "hidden": hidden}, path)
        res[name] = path
        print(f"trained std {name} on {dev} in {time.time() - t0:.0f}s -> {path}", flush=True)
    return res


def auroc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)) if len(set(y)) == 2 else float("nan")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_cache", required=True); ap.add_argument("--mw_ckpt", required=True)
    ap.add_argument("--rows", type=int, default=100000); ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--hidden", type=int, default=256); ap.add_argument("--n_eval", type=int, default=10)
    ap.add_argument("--n_warn", type=int, default=20); ap.add_argument("--skip_policy", action="store_true"); ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(HERE, "logs_3d", "compare_mw_vs_std.json"))
    a = ap.parse_args()
    out_dir = os.path.join(HERE, "data_cache"); os.makedirs(out_dir, exist_ok=True)
    std = {"mlp": os.path.join(out_dir, "std_mlp.pt"), "cvae": os.path.join(out_dir, "std_cvae.pt")} if a.skip_policy \
        else train_std(a.data_cache, a.rows, a.epochs, a.hidden, out_dir)
    seeds = range(5000, 5000 + a.n_eval)
    results = {}
    for label, kind, path in (("std MLP (behaviour cloning)", "mlp", std["mlp"]), ("std CVAE (plain PyTorch)", "cvae", std["cvae"]),
                              ("MW manifold, z-query (current)", "mw_z", a.mw_ckpt),
                              ("MW manifold, action by MW distance (z warm start)", "mw_mwd", a.mw_ckpt),
                              ("MW manifold, action by MW distance (zero start)", "mw_mwd0", a.mw_ckpt))[:0 if a.skip_policy else None]:
        t0 = time.time()
        r = _pool(a.data_cache, [(kind, path, s, 500, False) for s in seeds], a.workers)
        results[label] = {"steps": float(np.mean([x["steps"] for x in r])), "x": float(np.mean([x["x"] for x in r])),
                          "per_seed_steps": [x["steps"] for x in r]}
        print(f"{label:55s} steps={results[label]['steps']:6.1f}  x={results[label]['x']:5.2f}  ({time.time() - t0:.0f}s)", flush=True)

    # B. early-warning comparison on the MW student's own rollouts
    r = _pool(a.data_cache, [("mw_z", a.mw_ckpt, s, 500, True) for s in range(6000, 6000 + a.n_warn)], a.workers)
    def collect(neg_lo, neg_hi):
        """positives: last 20 steps before a fall; negatives: steps in [n-neg_hi, n-neg_lo) of the same rollouts
        (a matched time window, so 'later in the episode' cannot explain the score)."""
        Y, M, H, K, T = [], [], [], [], []
        for x in r:
            n = x["steps"]
            for t, (md, ma, kn) in enumerate(x["trace"]):
                if x["fell"] and t >= n - 20:
                    y = 1
                elif x["fell"] and n - neg_hi <= t < n - neg_lo:
                    y = 0
                else:
                    continue
                Y.append(y); M.append(md); H.append(ma); K.append(kn); T.append(t)
        return {"n_pos": int(sum(Y)), "n_neg": int(len(Y) - sum(Y)), "auroc_mw_distance": auroc(Y, M),
                "auroc_mahalanobis": auroc(Y, H), "auroc_knn": auroc(Y, K), "auroc_step_index_baseline": auroc(Y, T)}
    warn = {"n_rollouts": len(r), "n_fell": sum(x["fell"] for x in r),
            "matched_window_40_to_60_steps_before_fall": collect(40, 60),
            "matched_window_30_to_50_steps_before_fall": collect(30, 50)}
    print("early-warning (positive = within 20 steps of a fall):", json.dumps(warn), flush=True)
    json.dump({"policy": results, "warning": warn, "settings": vars(a)}, open(a.out, "w"), indent=1)
