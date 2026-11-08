#!/usr/bin/env python3
"""
train_student.py - distil the ProtoMotions G1 tracker into a small student, on the GPU (main venv: torch+cuda, pyro, mw_framework).

    python train_student.py --kind bundle --laws --name bundle_laws --train_motions walk,wave,high5,jumpjack
    python train_student.py --kind mlp --name mlp
Data: data_pm/<motion>_s<seed>.npz written by pm_collect.py (deployable observation 349 -> teacher action 29).
Exports a plain-PyTorch checkpoint (students/<name>.pt) that the ProtoMotions CPU environment can load without Pyro.
"""
import os, sys, glob, json, time, argparse
import numpy as np, torch, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
for p in (os.environ.get("MW_ROBOTICS", os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))),):
    sys.path.insert(0, p)
from student_net import MLPStudent, BundleStudent

N_IN, N_OUT, PREV0 = 349, 29, 64 + 256


def data_dir():
    """PM_DATA env var, else data_pm/ next to ours/ (original Windows layout), else data_pm/ at the repo root."""
    for d in (os.environ.get("PM_DATA"), os.path.join(ROOT, "data_pm"), os.path.join(os.path.dirname(ROOT), "data_pm")):
        if d and glob.glob(os.path.join(d, "*_s*.npz")):
            return d
    raise SystemExit("no data_pm/*.npz found; set PM_DATA")


def load_data(motions, fraction, seed):
    """fraction < 1: with motion-library data (npz has mid + motion_files) keep the training episodes of a fixed random subset of
    MOTIONS (nested: the 10 % set is inside the 40 % set; same subset for every training seed); otherwise a subset of episodes.
    The validation episodes are the same for every fraction."""
    X, A, E, K = [], [], [], []
    ddir = data_dir()
    off = 0
    for m in motions:
        for f in sorted(glob.glob(os.path.join(ddir, f"{m}_s*.npz"))):
            z = np.load(f)
            keep = np.ones(len(z["X"]), bool)
            if "root_z" in z.files:                       # drop whole episodes in which the TEACHER fell (bad labels)
                fell = (z["root_z"] < 0.45) | (z["upright"] < 0.5)
                bad = np.unique(z["ep"][fell]); keep = ~np.isin(z["ep"], bad)
                print(f"  {os.path.basename(f)}: dropped {len(bad)} of {z['ep'].max() + 1} episodes where the teacher fell", flush=True)
            if "mid" in z.files and len(z["motion_files"]):
                names = np.array([os.path.basename(str(p)) for p in z["motion_files"]])
                K.append(names[z["mid"][keep]])
            else:
                K.append(np.full(int(keep.sum()), f"{m}:{os.path.basename(f)}"))
            X.append(z["X"][keep]); A.append(z["A"][keep]); E.append(z["ep"][keep] + off); off += int(z["ep"].max()) + 1
    X, A, E, K = np.concatenate(X), np.concatenate(A), np.concatenate(E), np.concatenate(K)
    rng = np.random.RandomState(seed)
    eps = np.unique(E); rng.shuffle(eps)
    n_val = max(2, int(0.05 * len(eps)))
    val_eps, tr_eps = eps[:n_val], eps[n_val:]
    if fraction < 1.0:
        uniq = np.unique(K)
        if len(uniq) > 20:                                  # motion-level subset (library data)
            order = np.random.RandomState(12345).permutation(len(uniq))
            chosen = set(uniq[order[:max(1, int(round(fraction * len(uniq))))]])
            ep_motion = {e: k for e, k in zip(E, K)}
            tr_eps = np.array([e for e in tr_eps if ep_motion[e] in chosen])
            print(f"  fraction {fraction}: {len(chosen)} of {len(uniq)} motions, {len(tr_eps)} training episodes", flush=True)
        else:
            tr_eps = tr_eps[:max(2, int(round(fraction * len(tr_eps))))]
    tr, va = np.isin(E, tr_eps), np.isin(E, val_eps)
    return X[tr], A[tr], X[va], A[va], len(tr_eps), len(val_eps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["mlp", "bundle"], required=True)
    ap.add_argument("--laws", action="store_true")
    ap.add_argument("--name", required=True)
    ap.add_argument("--train_motions", default="walk,wave,high5,jumpjack")
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--latent_s", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--sigma_a", type=float, default=0.02)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)
    Xt, At, Xv, Av, n_tr, n_va = load_data(a.train_motions.split(","), a.fraction, a.seed)
    print(f"data: {len(Xt)} train pairs ({n_tr} episodes), {len(Xv)} val pairs ({n_va} episodes); device {dev}", flush=True)
    xm, xs = Xt.mean(0), Xt.std(0) + 1e-6
    ym, ys = At.mean(0), At.std(0) + 1e-6
    xt, at = torch.tensor(Xt, device=dev), torch.tensor(At, device=dev)
    xv, av = torch.tensor(Xv, device=dev), torch.tensor(Av, device=dev)
    ysd = torch.tensor(ys, device=dev)
    cfg = {"n_in": N_IN, "n_out": N_OUT, "hidden": a.hidden}
    t0 = time.time()
    best, best_state = float("inf"), None

    def val_err(model):
        with torch.no_grad():
            return float((((model(xv) - av) / ysd) ** 2).mean())

    if a.kind == "mlp":
        model = MLPStudent(**cfg).to(dev)
        for n, v in (("xm", xm), ("xs", xs), ("ym", ym), ("ys", ys)):
            getattr(model, n).copy_(torch.tensor(v))
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps, eta_min=1e-4)
        for step in range(a.steps):
            b = torch.randint(0, len(xt), (a.bs,), device=dev)
            loss = (((model(xt[b]) - at[b]) / ysd) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            if (step + 1) % 500 == 0 or step + 1 == a.steps:
                v = val_err(model)
                if v < best:
                    best, best_state = v, {k: x.detach().cpu().clone() for k, x in model.state_dict().items()}
                print(f"  mlp step {step + 1}/{a.steps} loss {float(loss):.4f} val {v:.4f}", flush=True)
        export_cfg = cfg
    else:
        import pyro
        from pyro.infer import SVI, Trace_ELBO
        from pyro.optim import ClippedAdam
        from mw_framework import ManifoldAlgebra, MWNormaliser    # --kind bundle only: mw_framework is in the archive/pre-fbw tag
        pyro.set_rng_seed(a.seed); pyro.clear_param_store()
        names = [f"o{i}" for i in range(N_IN)] + [f"a{i}" for i in range(N_OUT)]
        X = np.concatenate([Xt, At], 1)
        norm = MWNormaliser.fit(X, names)
        alg = ManifoldAlgebra(names, norm, obs_sigma=[0.15] * N_IN + [a.sigma_a] * N_OUT, dropout=0.0)
        alg.product("obs", names[:N_IN], a.latent_s, hidden=a.hidden)
        alg.fibre("act", names[N_IN:], 2, over="obs", hidden=a.hidden)
        if a.laws:
            amax = float(np.percentile(np.abs(At), 99.9) * 1.2)
            rate = np.abs(At - Xt[:, PREV0:PREV0 + N_OUT]).mean(1)
            tol_rate = float(rate.mean() * 1.5)
            print(f"laws: |action| <= {amax:.2f} ; action-rate |a - previous a| (mean per joint) ~ N(0, {tol_rate:.3f})", flush=True)

            def acts(x): return torch.stack([x[f"a{i}"] for i in range(N_OUT)], -1)
            alg.glue("p1_range", lambda x: F.relu(acts(x).abs() - amax).sum(-1), 0.05)
            alg.glue("p2_rate", lambda x: (acts(x) - torch.stack([x[f"o{PREV0 + i}"] for i in range(N_OUT)], -1)).abs().mean(-1), tol_rate)
        alg = alg.to(dev)
        Xn = torch.tensor(norm.transform(X), dtype=torch.float32, device=dev)
        alg._n_total = len(Xn)
        svi = SVI(alg.model, alg.guide, ClippedAdam({"lr": 1e-3, "lrd": 0.1 ** (1.0 / a.steps), "clip_norm": 10.0}), Trace_ELBO())
        model = BundleStudent(n_in=N_IN, n_out=N_OUT, latent_s=a.latent_s, latent_a=2, hidden=a.hidden).to(dev)

        def sync():                                           # copy the trained weights into the plain inference module
            v_o, v_a = alg.vaes["obs"], alg.vaes["act"]
            model.enc.load_state_dict(v_o.enc.state_dict()); model.mu.load_state_dict(v_o.mu.state_dict())
            model.dec.load_state_dict(v_a.dec.state_dict())
            for n, v in (("xm", norm.mean[:N_IN]), ("xs", norm.std[:N_IN]), ("ym", norm.mean[N_IN:]), ("ys", norm.std[N_IN:])):
                getattr(model, n).copy_(torch.tensor(v))
        for step in range(a.steps):
            alg.train()
            b = torch.randint(0, len(Xn), (a.bs,), device=dev)
            loss = svi.step(Xn[b]) / a.bs
            if (step + 1) % 500 == 0 or step + 1 == a.steps:
                alg.eval(); sync(); model.eval()
                v = val_err(model)
                if v < best:
                    best, best_state = v, {k: x.detach().cpu().clone() for k, x in model.state_dict().items()}
                print(f"  bundle{'+laws' if a.laws else ''} step {step + 1}/{a.steps} elbo/sample {loss:.2f} val {v:.4f}", flush=True)
        export_cfg = {"n_in": N_IN, "n_out": N_OUT, "latent_s": a.latent_s, "latent_a": 2, "hidden": a.hidden}
    os.makedirs(os.path.join(ROOT, "students"), exist_ok=True)
    out = os.path.join(ROOT, "students", f"{a.name}.pt")
    torch.save({"kind": a.kind, "cfg": export_cfg, "state": best_state}, out)
    n_par = sum(p.numel() for p in model.parameters())
    meta = dict(name=a.name, kind=a.kind, laws=a.laws, fraction=a.fraction, steps=a.steps, train_pairs=int(len(Xt)), n_episodes=int(n_tr),
                best_val=best, params=int(n_par), size_mb=round(os.path.getsize(out) / 1e6, 2), train_s=round(time.time() - t0), seed=a.seed)
    json.dump(meta, open(out.replace(".pt", ".json"), "w"), indent=1)
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
