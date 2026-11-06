#!/usr/bin/env python3
"""train_student_prob.py - probabilistic students for Retry_MW_distillation.md (system python: torch + pyro, GPU if available).

    python train_student_prob.py --kind gauss    --name r1_gauss  [--lastlayer]          # R1 (R3 with --lastlayer)
    python train_student_prob.py --kind ensemble --name r2_ens --k 5                      # R2
    python train_student_prob.py --kind mwv2     --name r4_mwv2  [--lastlayer]           # R4 (R5 with --lastlayer), Pyro SVI
Common: --train_motions walk,wave,high5,jumpjack  --fraction 1.0  --steps 6000  --seed 0
Data: data_pm/<motion>_s*.npz (pm_collect.py). Output: students/<name>.pt + .json (plain PyTorch; load with student_prob.load).

Mean action and its variance are trained so the variance cannot cost accuracy: the mean sees a fixed-sigma Gaussian (MSE-like) term,
the log-variance head sees a Gaussian NLL with the mean detached.
"""
import os, sys, json, time, argparse
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from train_student import load_data                         # same episodes, split and validation rule as the first study
import student_prob as SP

N_IN, N_OUT, PREV0 = SP.N_IN, SP.N_OUT, SP.PREV0


def law_constants(Xt, At):
    amax = float(np.percentile(np.abs(At), 99.9) * 1.2)
    tol_rate = float(np.abs(At - Xt[:, PREV0:PREV0 + N_OUT]).mean(1).mean() * 1.5)
    return amax, tol_rate


def train_gauss(Xt, At, Xv, Av, a, dev, seed):
    torch.manual_seed(seed)
    net = SP.GaussMLP(hidden=a.hidden).to(dev)
    xm, xs, ym, ys = Xt.mean(0), Xt.std(0) + 1e-6, At.mean(0), At.std(0) + 1e-6
    net.set_norm(xm, xs, ym, ys, *law_constants(Xt, At))
    xt, yt = torch.tensor(Xt, device=dev), torch.tensor((At - ym) / ys, device=dev)
    xv, av = torch.tensor(Xv, device=dev), torch.tensor(Av, device=dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps, eta_min=1e-4)
    best, best_state = float("inf"), None
    for step in range(a.steps):
        b = torch.randint(0, len(xt), (a.bs,), device=dev)
        mu, lv, _ = net.heads(net.nx(xt[b]))
        y = yt[b]
        loss = ((mu - y) ** 2).mean() + 0.5 * (lv + (y - mu.detach()) ** 2 / lv.exp()).mean() * a.w_var
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if (step + 1) % 500 == 0 or step + 1 == a.steps:
            v = val_err(net, xv, av)
            if v < best:
                best, best_state = v, {k: t.detach().clone() for k, t in net.state_dict().items()}
            print(f"  gauss[{seed}] step {step + 1}/{a.steps} loss {float(loss.detach()):.4f} val {v:.4f}", flush=True)
    net.load_state_dict(best_state)
    return net, {"n_in": N_IN, "n_out": N_OUT, "hidden": a.hidden}, best


def train_mwv2(Xt, At, Xv, Av, a, dev, seed):
    import pyro
    import pyro.distributions as dist
    from pyro import poutine
    from pyro.infer import SVI, TraceMeanField_ELBO
    from pyro.optim import ClippedAdam
    pyro.set_rng_seed(seed); pyro.clear_param_store(); torch.manual_seed(seed)
    net = SP.MWBundleV2(hidden=a.hidden, latent_s=a.latent_s).to(dev)
    xm, xs, ym, ys = Xt.mean(0), Xt.std(0) + 1e-6, At.mean(0), At.std(0) + 1e-6
    amax, tol_rate = law_constants(Xt, At)
    net.set_norm(xm, xs, ym, ys, amax, tol_rate)
    xt, yt = torch.tensor(Xt, device=dev), torch.tensor((At - ym) / ys, device=dev)
    xv, av = torch.tensor(Xv, device=dev), torch.tensor(Av, device=dev)
    state = {"beta": 0.0}
    sig_a = a.sigma_a

    def model(x, y):
        pyro.module("mw", net)
        xn = net.nx(x)
        with pyro.plate("data", x.shape[0]):
            with poutine.scale(scale=max(state["beta"], 1e-4)):
                z = pyro.sample("z", dist.Normal(torch.zeros(x.shape[0], net.latent_s, device=dev), 1.0).to_event(1))
            with poutine.scale(scale=a.w_obs):                                       # the state manifold, down-weighted
                pyro.sample("o", dist.Normal(net.dec_o(z), net.o_logsig.exp()).to_event(1), obs=xn)
            mu, lv, _ = net.act(xn, z)
            pyro.sample("a_mean", dist.Normal(mu, sig_a).to_event(1), obs=y)         # accuracy term on the mean
            with poutine.scale(scale=a.w_var):
                pyro.sample("a_var", dist.Normal(mu.detach(), (0.5 * lv).exp()).to_event(1), obs=y)   # aleatoric head
            if a.laws:                                                                # MW glue: physical priors on the output
                rng, rate = net.laws(x, mu * net.ys + net.ym)
                pyro.factor("law_range", -rng / 0.05)
                pyro.factor("law_rate", -0.5 * rate ** 2)

    def guide(x, y):
        pyro.module("mw", net)
        mu_z, lv_z = net.encode(net.nx(x))
        with pyro.plate("data", x.shape[0]):
            with poutine.scale(scale=max(state["beta"], 1e-4)):
                pyro.sample("z", dist.Normal(mu_z, (0.5 * lv_z).exp()).to_event(1))

    svi = SVI(model, guide, ClippedAdam({"lr": 1e-3, "lrd": 0.1 ** (1.0 / a.steps), "clip_norm": 10.0}), TraceMeanField_ELBO())
    best, best_state, warm = float("inf"), None, max(1, int(0.3 * a.steps))
    for step in range(a.steps):
        state["beta"] = min(1.0, (step + 1) / warm)
        net.train()
        b = torch.randint(0, len(xt), (a.bs,), device=dev)
        loss = svi.step(xt[b], yt[b]) / a.bs
        if (step + 1) % 500 == 0 or step + 1 == a.steps:
            net.eval()
            v = val_err(net, xv, av)
            with torch.no_grad():
                mu_z, lv_z = net.encode(net.nx(xv[:4096]))
                kl_dim = (0.5 * (mu_z ** 2 + lv_z.exp() - 1 - lv_z)).mean(0)
            if v < best:
                best, best_state = v, {k: t.detach().clone() for k, t in net.state_dict().items()}
            print(f"  mwv2 step {step + 1}/{a.steps} beta {state['beta']:.2f} elbo/sample {loss:.2f} val {v:.4f} "
                  f"KL/dim mean {float(kl_dim.mean()):.3f} live dims (>0.05) {int((kl_dim > 0.05).sum())}/{net.latent_s}", flush=True)
    net.load_state_dict(best_state)
    return net, {"n_in": N_IN, "n_out": N_OUT, "hidden": a.hidden, "latent_s": a.latent_s}, best


@torch.no_grad()
def val_err(net, xv, av):
    net.eval()
    return float((((net(xv) - av) / net.ys) ** 2).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["gauss", "ensemble", "mwv2"], required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--lastlayer", action="store_true", help="wrap with exact last-layer Laplace (R3 / R5)")
    ap.add_argument("--laws", action="store_true", default=True)
    ap.add_argument("--no-laws", dest="laws", action="store_false")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--train_motions", default="walk,wave,high5,jumpjack")
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--latent_s", type=int, default=16)
    ap.add_argument("--sigma_a", type=float, default=0.1, help="fixed sigma of the mean-action term (normalised units)")
    ap.add_argument("--w_obs", type=float, default=0.05, help="weight of the observation (state-manifold) likelihood")
    ap.add_argument("--w_var", type=float, default=0.1, help="weight of the aleatoric-variance term")
    ap.add_argument("--lam", type=float, default=1.0, help="last-layer prior precision")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xt, At, Xv, Av, n_tr, n_va = load_data(a.train_motions.split(","), a.fraction, a.seed)
    print(f"data: {len(Xt)} train pairs ({n_tr} episodes), {len(Xv)} val pairs ({n_va} episodes); device {dev}", flush=True)
    t0 = time.time()
    if a.kind == "gauss":
        net, cfg, best = train_gauss(Xt, At, Xv, Av, a, dev, a.seed)
    elif a.kind == "mwv2":
        net, cfg, best = train_mwv2(Xt, At, Xv, Av, a, dev, a.seed)
    else:
        members, cfgs, bests = [], [], []
        for k in range(a.k):
            m, c, b = train_gauss(Xt, At, Xv, Av, a, dev, a.seed * 100 + k)
            members.append(m.cpu()); cfgs.append({"kind": "gauss", "cfg": c}); bests.append(b)
        net, cfg, best = SP.Ensemble(members), {"members": cfgs}, float(np.mean(bests))
    if a.lastlayer:
        assert a.kind in ("gauss", "mwv2"), "--lastlayer wraps a single model"
        base = net.to(dev).eval()
        ll = SP.LastLayer(base, hidden=a.hidden).to(dev)
        xt = torch.tensor(Xt, device=dev)
        yt = torch.tensor((At - base.ym.cpu().numpy()) / base.ys.cpu().numpy(), device=dev)
        bs = 8192
        ll.fit([base.nx(xt[i:i + bs]) for i in range(0, len(xt), bs)], [yt[i:i + bs] for i in range(0, len(xt), bs)], lam=a.lam)
        net, cfg = ll, {"base": {"kind": a.kind, "cfg": cfg}}
    net = net.cpu().eval()
    os.makedirs(os.path.join(ROOT, "students"), exist_ok=True)
    out = os.path.join(ROOT, "students", f"{a.name}.pt")
    SP.save(net, cfg, out)
    xv = torch.tensor(Xv[:2048])
    t1 = time.perf_counter()
    with torch.no_grad():
        for i in range(200):
            net.uq(xv[i:i + 1])
    lat_ms = (time.perf_counter() - t1) / 200 * 1e3
    meta = dict(name=a.name, kind=net.kind, base=a.kind, lastlayer=a.lastlayer, laws=a.laws, train_motions=a.train_motions,
                fraction=a.fraction, steps=a.steps, seed=a.seed, train_pairs=int(len(Xt)), best_val=best,
                params=int(sum(p.numel() for p in net.parameters())), size_mb=round(os.path.getsize(out) / 1e6, 2),
                train_s=round(time.time() - t0), uq_ms_cpu_torch=round(lat_ms, 3))
    json.dump(meta, open(out.replace(".pt", ".json"), "w"), indent=1)
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
