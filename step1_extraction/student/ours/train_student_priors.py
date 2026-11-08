#!/usr/bin/env python3
"""train_student_priors.py - MW v3 student with physical (balance / dance) priors, Pyro SVI, CPU-friendly.

    CUDA_VISIBLE_DEVICES= python3 train_student_priors.py --name r6_mwpri_q100 --train_motions walk,wave,high5,jumpjack,uncond --lam 5
    ... --lam 0  -> the control arm: identical model and training, priors off
Same objective as MW v2 (state manifold + accuracy-first action head + range/rate laws) plus:
  * a dynamics head trained on the teacher's transitions (next proprioception);
  * priors.py laws (capture-point balance, torso upright, joint limits, follow-the-reference) evaluated on the predicted next state
    of the student's OWN action, with the dynamics head frozen inside the law (laws shape the policy, not the dynamics model);
  * the laws are also enforced on COLLOCATION states the data never covers: perturbed tilt / angular velocity / joints, and
    swapped reference motions (a new motion requested from the current state). That is where a prior can add information.
Tolerances: the teacher's own 95th percentile on the training transitions (priors.calibrate).
"""
import os, sys, json, time, glob, argparse
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import student_prob as SP
from priors import Priors, transitions, LAWS, pd_map, next_obs
from train_student import data_dir

N_IN, N_OUT, PREV0 = SP.N_IN, SP.N_OUT, SP.PREV0


def collocate(x, rng_t):
    """States the data does not cover: new reference motion from the current state + perturbed body state."""
    B = x.shape[0]; xc = x.clone(); dv = x.device
    swap = torch.rand(B, device=dv) < 0.5
    perm = torch.randperm(B, device=dv)
    xc[swap, 64:320] = x[perm][swap, 64:320]                    # request a different motion
    xc[:, 0:29] += 0.05 * torch.randn(B, 29, device=dv)                   # joint angles
    xc[:, 29:58] += 0.5 * torch.randn(B, 29, device=dv)                   # joint velocities
    xc[:, 58:61] += 0.3 * torch.randn(B, 3, device=dv)                    # torso angular velocity
    g = xc[:, 61:64] + 0.1 * torch.randn(B, 3, device=dv)                 # tilt
    xc[:, 61:64] = torch.nn.functional.normalize(g, dim=-1)
    return xc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--train_motions", default="walk,wave,high5,jumpjack,uncond")
    ap.add_argument("--steps", type=int, default=6000); ap.add_argument("--bs", type=int, default=1024); ap.add_argument("--bs_col", type=int, default=256)
    ap.add_argument("--lam", type=float, default=5.0, help="weight of the priors on data states (0 = control arm)")
    ap.add_argument("--lam_col", type=float, default=None, help="weight on collocation states (default = lam)")
    ap.add_argument("--law_weights", default="1,1,1,1", help=",".join(LAWS))
    ap.add_argument("--horizon", type=int, default=1, help="look-ahead steps: the laws are applied along a rolled-forward path")
    ap.add_argument("--tol_q", type=float, default=0.95, help="tolerance = this quantile of the teacher's own law values")
    ap.add_argument("--w_obs", type=float, default=0.05); ap.add_argument("--w_var", type=float, default=0.1); ap.add_argument("--w_dyn", type=float, default=1.0)
    ap.add_argument("--sigma_a", type=float, default=0.1); ap.add_argument("--latent_s", type=int, default=16); ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--threads", type=int, default=4); ap.add_argument("--lastlayer", action="store_true")
    ap.add_argument("--motion_subset", default="", help="file with motion names (one per line): keep only these motions' episodes")
    ap.add_argument("--device", default="cpu", help="cpu | cuda"); ap.add_argument("--ckpt_every", type=int, default=500,
                    help="save a resumable checkpoint (students/<name>.ckpt) every N steps; a restart resumes from it")
    a = ap.parse_args()
    dev = torch.device(a.device)
    lam_col = a.lam if a.lam_col is None else a.lam_col
    lw = [float(v) for v in a.law_weights.split(",")]
    torch.set_num_threads(a.threads)
    import pyro, pyro.distributions as dist
    from pyro import poutine
    from pyro.infer import SVI, TraceMeanField_ELBO
    from pyro.optim import ClippedAdam
    pyro.set_rng_seed(a.seed); pyro.clear_param_store(); torch.manual_seed(a.seed)

    files = [f for m in a.train_motions.split(",") for f in sorted(glob.glob(os.path.join(data_dir(), f"{m}_s*.npz")))]
    keep = set(l.strip() for l in open(a.motion_subset) if l.strip()) if a.motion_subset else None
    X, A, P = transitions(files, keep_motions=keep)
    rng = np.random.RandomState(a.seed); idx = rng.permutation(len(X)); nv = max(1000, len(X) // 20)
    va, tr = idx[:nv], idx[nv:]
    print(f"data: {len(tr)} train / {len(va)} val transitions from {len(files)} files", flush=True)
    xm, xs = X[tr].mean(0), X[tr].std(0) + 1e-6; ym, ys = A[tr].mean(0), A[tr].std(0) + 1e-6
    amax = float(np.percentile(np.abs(A[tr]), 99.9) * 1.2); tol_rate = float(np.abs(A[tr] - X[tr][:, PREV0:PREV0 + N_OUT]).mean(1).mean() * 1.5)
    net = SP.MWBundleV3(hidden=a.hidden, latent_s=a.latent_s)
    net.set_norm(xm, xs, ym, ys, amax, tol_rate)
    D = P[tr] - X[tr][:, :64]; net.dm.copy_(torch.tensor(D.mean(0))); net.ds.copy_(torch.tensor(D.std(0) + 1e-6))
    pri = Priors()
    cal = rng.choice(tr, min(40000, len(tr)), replace=False)
    pri.calibrate(torch.tensor(X[cal]), torch.tensor(P[cal]), q=a.tol_q)
    print(f"prior tolerances (teacher {100 * a.tol_q:.0f}th pct):", dict(zip(LAWS, pri.tol.numpy().round(3))), flush=True)
    net, pri = net.to(dev), pri.to(dev)
    Xt, At, Pt = torch.tensor(X).to(dev), torch.tensor(A).to(dev), torch.tensor(P).to(dev)
    state = {"beta": 0.0}

    OFF, SC = (t.to(dev) for t in pd_map())

    def law_terms(x, xn, z, mu_a_n):
        """Laws along a rolled-forward path of `horizon` steps: the student acts at every step (deterministic latent), the frozen
        dynamics head predicts the next state, the observation is rolled with next_obs. Mean penalty over the path, per start state."""
        params = {k: v.detach() for k, v in net.dyn.named_parameters()}
        tot = 0
        for k in range(a.horizon):
            if k > 0:
                xn = net.nx(x); z, _ = net.encode(xn); mu_a_n, _, _ = net.act(xn, z)
            act = mu_a_n * net.ys + net.ym
            p_hat = net.predict_next(x, xn, z, act, params)
            tot = tot + pri.penalty(pri.values(x, p_hat), lw)
            if k + 1 < a.horizon:
                x = next_obs(x, p_hat, act, OFF, SC)
        return tot / a.horizon

    def model(idx_b, idx_c):
        pyro.module("mw", net)
        x, y = Xt[idx_b], (At[idx_b] - net.ym) / net.ys
        xn = net.nx(x)
        with pyro.plate("data", x.shape[0]):
            with poutine.scale(scale=max(state["beta"], 1e-4)):
                z = pyro.sample("z", dist.Normal(torch.zeros(x.shape[0], net.latent_s, device=x.device), 1.0).to_event(1))
            with poutine.scale(scale=a.w_obs):
                pyro.sample("o", dist.Normal(net.dec_o(z), net.o_logsig.exp()).to_event(1), obs=xn)
            mu, lv, _ = net.act(xn, z)
            pyro.sample("a_mean", dist.Normal(mu, a.sigma_a).to_event(1), obs=y)
            with poutine.scale(scale=a.w_var):
                pyro.sample("a_var", dist.Normal(mu.detach(), (0.5 * lv).exp()).to_event(1), obs=y)
            # dynamics head on the teacher's own transition
            d_true = (Pt[idx_b] - x[:, :64] - net.dm) / net.ds
            d_pred = net.dyn(torch.cat([xn, z, y], -1))
            with poutine.scale(scale=a.w_dyn):
                pyro.sample("dyn", dist.Normal(d_pred, 1.0).to_event(1), obs=d_true)
            rng_l, rate_l = net.laws(x, mu * net.ys + net.ym)
            pyro.factor("law_range", -rng_l / 0.05); pyro.factor("law_rate", -0.5 * rate_l ** 2)

        if a.lam > 0:                                                      # data states (subset), rolled forward
            xd = Xt[idx_c]; xdn = net.nx(xd); zd, _ = net.encode(xdn); mud, _, _ = net.act(xdn, zd)
            pyro.factor("priors_data", -a.lam * law_terms(xd, xdn, zd, mud).sum() * (x.shape[0] / xd.shape[0]))
        if lam_col > 0:                                                    # collocation states (no labels)
            xc = collocate(Xt[idx_c], None); xcn = net.nx(xc)
            zc, _ = net.encode(xcn)
            muc, _, _ = net.act(xcn, zc)
            pyro.factor("priors_col", -lam_col * law_terms(xc, xcn, zc, muc).sum() * (x.shape[0] / xc.shape[0]))

    def guide(idx_b, idx_c):
        pyro.module("mw", net)
        mu_z, lv_z = net.encode(net.nx(Xt[idx_b]))
        with pyro.plate("data", len(idx_b)):
            with poutine.scale(scale=max(state["beta"], 1e-4)):
                pyro.sample("z", dist.Normal(mu_z, (0.5 * lv_z).exp()).to_event(1))

    svi = SVI(model, guide, ClippedAdam({"lr": 1e-3, "lrd": 0.1 ** (1.0 / a.steps), "clip_norm": 10.0}), TraceMeanField_ELBO())
    trn = torch.tensor(tr, device=dev); xv, av = Xt[va], At[va]
    best, best_state, warm, t0 = float("inf"), None, max(1, int(0.3 * a.steps)), time.time()
    ckpt = os.path.join(os.path.dirname(HERE), "students", f"{a.name}.ckpt"); start = 0
    if os.path.exists(ckpt):                                               # resume
        c = torch.load(ckpt, map_location=dev, weights_only=False)
        net.load_state_dict(c["net"]); svi.optim.set_state(c["optim"]); torch.set_rng_state(c["rng"])
        if dev.type == "cuda" and c.get("rng_cuda") is not None: torch.cuda.set_rng_state(c["rng_cuda"])
        start, best, best_state = c["step"], c["best"], c["best_state"]
        print(f"resumed from {ckpt} at step {start} (best val {best:.4f})", flush=True)
    for step in range(start, a.steps):
        state["beta"] = min(1.0, (step + 1) / warm)
        net.train()
        ib = trn[torch.randint(0, len(trn), (a.bs,), device=dev)]; ic = trn[torch.randint(0, len(trn), (a.bs_col,), device=dev)]
        loss = svi.step(ib, ic) / a.bs
        if (step + 1) % 250 == 0 or step + 1 == a.steps:
            net.eval()
            with torch.no_grad():
                v = float((((net(xv) - av) / net.ys) ** 2).mean())
                sub = va[:4000]; xs_ = Xt[sub]; xn_ = net.nx(xs_); z_, _ = net.encode(xn_); mu_, _, _ = net.act(xn_, z_)
                p_hat = net.predict_next(xs_, xn_, z_, mu_ * net.ys + net.ym)
                dyn_err = float(((p_hat - Pt[sub]) / net.ds).pow(2).mean())
                xc = collocate(xs_[:2000], None); xcn = net.nx(xc); zc, _ = net.encode(xcn); muc, _, _ = net.act(xcn, zc)
                viol = (pri.values(xc, net.predict_next(xc, xcn, zc, muc * net.ys + net.ym)) > pri.tol).float().mean(0)
            if v < best:
                best, best_state = v, {k: t.detach().clone() for k, t in net.state_dict().items()}
            print(f"  step {step + 1}/{a.steps} beta {state['beta']:.2f} loss/sample {loss:.2f} val {v:.4f} dyn {dyn_err:.3f} "
                  f"collocation beyond-teacher {dict(zip(LAWS, viol.cpu().numpy().round(2)))} ({time.time() - t0:.0f}s)", flush=True)
        if a.ckpt_every and (step + 1) % a.ckpt_every == 0 and step + 1 < a.steps:
            os.makedirs(os.path.dirname(ckpt), exist_ok=True)
            torch.save({"net": net.state_dict(), "optim": svi.optim.get_state(), "step": step + 1, "best": best, "best_state": best_state,
                        "rng": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state() if dev.type == "cuda" else None}, ckpt + ".tmp")
            os.replace(ckpt + ".tmp", ckpt); print(f"  checkpoint -> {ckpt} (step {step + 1})", flush=True)
    net.load_state_dict(best_state); net = net.cpu().eval(); pri = pri.cpu()
    Xt, At = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)), torch.from_numpy(np.ascontiguousarray(A, dtype=np.float32))
    cfg = {"n_in": N_IN, "n_out": N_OUT, "hidden": a.hidden, "latent_s": a.latent_s}
    out_net, out_cfg = net, cfg
    if a.lastlayer:
        ll = SP.LastLayer(net, hidden=a.hidden)
        bs = 8192; trs = torch.tensor(tr)
        ll.fit([net.nx(Xt[trs[i:i + bs]]) for i in range(0, len(trs), bs)], [(At[trs[i:i + bs]] - net.ym) / net.ys for i in range(0, len(trs), bs)])
        out_net, out_cfg = ll, {"base": {"kind": "mwv3", "cfg": cfg}}
    os.makedirs(os.path.join(os.path.dirname(HERE), "students"), exist_ok=True)
    path = os.path.join(os.path.dirname(HERE), "students", f"{a.name}.pt")
    SP.save(out_net, out_cfg, path, extra={"prior_tol": pri.tol.numpy().tolist(), "lam": a.lam, "lam_col": lam_col})
    meta = dict(name=a.name, lam=a.lam, lam_col=lam_col, law_weights=lw, steps=a.steps, best_val=best, train_s=round(time.time() - t0),
                params=int(sum(p.numel() for p in out_net.parameters())), size_mb=round(os.path.getsize(path) / 1e6, 2))
    json.dump(meta, open(path.replace(".pt", ".json"), "w"), indent=1); print(json.dumps(meta), flush=True)
    if os.path.exists(ckpt): os.remove(ckpt)


if __name__ == "__main__":
    main()
