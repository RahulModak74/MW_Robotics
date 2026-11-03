#!/usr/bin/env python3
"""
EXP_03 / E1 — Train a conditional VAE on (s, a) pairs from the proven
920-step balance policy. Measure whether the VAE can act as that policy
from state alone.

Loads cem_best_theta.npy, generates training + held-out data, trains a
CVAE, evaluates:
  1. Reconstruction error (does the VAE learn (s,a)?)
  2. Policy replacement error (can it act from s alone?)
  3. Rollout survival (is it functionally equivalent to the original?)
  4. Stochastic vs deterministic inference (does sample() help or hurt?)
"""
import sys, os, time
sys.path.insert(0, os.getcwd())
import numpy as np
import gymnasium as gym
import torch
torch.set_num_threads(8)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ── data collection ──────────────────────────────────────────────────────────
def collect(theta, seeds, max_steps=1000):
    """Runs the linear policy in Walker2d, returns (S, A) arrays."""
    from mw_to_protomotions.cem_policy_search import get_features, policy_action
    S, A = [], []
    for seed in seeds:
        env = gym.make("Walker2d-v5", exclude_current_positions_from_observation=False)
        obs, info = env.reset(seed=seed)
        data = env.unwrapped.data
        for _ in range(max_steps):
            feat = get_features(data)
            act = policy_action(theta, feat)
            S.append(feat); A.append(act)
            obs, r, term, trunc, info = env.step(act)
            if term:
                break
        env.close()
    return np.array(S, dtype=np.float32), np.array(A, dtype=np.float32)


# ── CVAE ─────────────────────────────────────────────────────────────────────
class CVAE(nn.Module):
    def __init__(self, state_dim=15, action_dim=6, latent_dim=8, hidden=64):
        super().__init__()
        self.state_dim, self.action_dim, self.latent_dim = state_dim, action_dim, latent_dim

        # q(z | s, a)
        self.enc = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.enc_mu  = nn.Linear(hidden, latent_dim)
        self.enc_sig = nn.Linear(hidden, latent_dim)

        # p(z | s)
        self.prior = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.prior_mu  = nn.Linear(hidden, latent_dim)
        self.prior_sig = nn.Linear(hidden, latent_dim)

        # p(a | z, s)
        self.dec = nn.Sequential(
            nn.Linear(latent_dim + state_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, action_dim), nn.Tanh(),
        )

    def encode(self, s, a):
        h = self.enc(torch.cat([s, a], -1))
        return self.enc_mu(h), F.softplus(self.enc_sig(h)) + 1e-3

    def prior_params(self, s):
        h = self.prior(s)
        return self.prior_mu(h), F.softplus(self.prior_sig(h)) + 1e-3

    def decode(self, z, s):
        return self.dec(torch.cat([z, s], -1))

    def forward(self, s, a, beta=0.01):
        mu_q, sig_q = self.encode(s, a)
        z = mu_q + sig_q * torch.randn_like(sig_q)
        a_hat = self.decode(z, s)
        recon = ((a - a_hat) ** 2).sum(-1).mean()

        mu_p, sig_p = self.prior_params(s)
        kl = 0.5 * (sig_q**2/sig_p**2 + (mu_q - mu_p)**2/sig_p**2
                    - 2*torch.log(sig_q/sig_p) - 1).sum(-1).mean()

        return recon + beta * kl, recon, kl

    @torch.no_grad()
    def act_deterministic(self, s):
        mu_p, _ = self.prior_params(s)
        return self.decode(mu_p, s)

    @torch.no_grad()
    def act_stochastic(self, s):
        mu_p, sig_p = self.prior_params(s)
        z = mu_p + sig_p * torch.randn_like(sig_p)
        return self.decode(z, s)


# ── training ─────────────────────────────────────────────────────────────────
def train_cvae(S, A, latent_dim=8, hidden=64, beta=0.01, epochs=200,
               batch_size=512, lr=1e-3):
    S_t = torch.as_tensor(S); A_t = torch.as_tensor(A)
    model = CVAE(S.shape[1], A.shape[1], latent_dim, hidden)
    opt = optim.Adam(model.parameters(), lr=lr)
    n = len(S)
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot, tot_recon, tot_kl = 0.0, 0.0, 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start+batch_size]
            loss, recon, kl = model(S_t[idx], A_t[idx], beta)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); tot_recon += recon.item(); tot_kl += kl.item()
        if (ep+1) % 50 == 0 or ep == 0:
            print(f"    ep {ep+1:>3}/{epochs}  loss={tot:.3f}  "
                  f"recon={tot_recon:.4f}  kl={tot_kl:.3f}", flush=True)
    model.eval()
    return model


# ── evaluation ───────────────────────────────────────────────────────────────
def eval_reconstruction(model, S, A):
    with torch.no_grad():
        S_t = torch.as_tensor(S); A_t = torch.as_tensor(A)
        mu_q, _ = model.encode(S_t, A_t)
        A_hat = model.decode(mu_q, S_t)
        return float(((A_t - A_hat) ** 2).sum(-1).mean())


def eval_policy_replacement(model, S, A_actual):
    with torch.no_grad():
        S_t = torch.as_tensor(S)
        A_vae = model.act_deterministic(S_t).numpy()
        return float(((A_actual - A_vae) ** 2).sum(-1).mean()), A_vae


def eval_rollout(model, seeds, max_steps=1000, stochastic=False):
    from mw_to_protomotions.cem_policy_search import get_features
    results = []
    for seed in seeds:
        env = gym.make("Walker2d-v5", exclude_current_positions_from_observation=False)
        obs, info = env.reset(seed=seed)
        data = env.unwrapped.data
        steps = 0
        for _ in range(max_steps):
            s = torch.as_tensor(get_features(data)[None], dtype=torch.float32)
            act = model.act_stochastic(s) if stochastic else model.act_deterministic(s)
            act = act.numpy()[0]
            obs, r, term, trunc, info = env.step(act)
            steps += 1
            if term:
                break
        env.close()
        results.append(steps)
    return np.array(results)


# ── main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("EXP_03 / E1 — VAE as Policy, Compression Test")
    print("=" * 68)

    # Load ground-truth policy
    theta_path = "cem_best_theta.npy"
    if not os.path.exists(theta_path):
        sys.exit(f"ERROR: {theta_path} not found — run ./run_plan_a.sh first")
    theta = np.load(theta_path)
    print(f"loaded {theta_path}, |theta| = {np.linalg.norm(theta):.3f}\n")

    # Generate training + held-out data
    print("[1/5] collecting training data (100 rollouts)…")
    t0 = time.time()
    S_train, A_train = collect(theta, seeds=list(range(0, 100)))
    print(f"  {len(S_train)} (s, a) pairs, {time.time()-t0:.1f}s\n")

    print("[2/5] collecting held-out data (20 rollouts)…")
    S_held, A_held = collect(theta, seeds=list(range(1000, 1020)))
    print(f"  {len(S_held)} (s, a) pairs\n")

    # Train CVAE
    print(f"[3/5] training CVAE (latent_dim=8, hidden=64, beta=0.01)…")
    t0 = time.time()
    model = train_cvae(S_train, A_train, latent_dim=8, hidden=64, beta=0.01)
    print(f"  trained in {time.time()-t0:.1f}s\n")

    # Evaluate
    print("[4/5] evaluating compression…")
    recon_err = eval_reconstruction(model, S_held, A_held)
    print(f"  reconstruction error (held-out):  {recon_err:.4f}")
    print(f"    (target: < 0.05 for 'VAE sees the data')")

    repl_err, A_vae = eval_policy_replacement(model, S_held, A_held)
    print(f"  policy replacement error:         {repl_err:.4f}")
    print(f"    (target: < 0.10 for 'VAE IS the policy')")

    print("\n[5/5] evaluating rollouts…")
    det_steps = eval_rollout(model, list(range(2000, 2020)), stochastic=False)
    sto_steps = eval_rollout(model, list(range(2000, 2020)), stochastic=True)
    print(f"  deterministic (z = prior_mu):  mean {det_steps.mean():.1f} / 1000")
    print(f"  stochastic    (z ~ prior):     mean {sto_steps.mean():.1f} / 1000")
    print(f"  original policy reference:     920.9 / 1000")

    # Save
    torch.save({
        "state_dict": model.state_dict(),
        "config": dict(state_dim=15, action_dim=6, latent_dim=8, hidden=64),
    }, "exp_03_cvae_balance.pt")
    print(f"\nsaved model to exp_03_cvae_balance.pt")

    # Verdict
    print("\n" + "=" * 68)
    print("VERDICT")
    print("=" * 68)
    if recon_err < 0.05 and repl_err < 0.10 and det_steps.mean() >= 900:
        print("STRONG — VAE is a valid policy object.")
        print("  → proceed to E2 (latent search)")
    elif recon_err < 0.05 and repl_err < 0.30 and det_steps.mean() >= 500:
        print("MODERATE — VAE represents the policy, but lossily.")
        print("  → sweep latent_dim (4, 8, 16) and beta (0.01, 0.1, 0.5)")
    elif recon_err < 0.05 and repl_err >= 0.30:
        print("AMBIGUOUS — VAE encodes (s,a) but cannot act from s alone.")
        print("  → diagnose: which failure mode from EXP_03 §6?")
        print("    Likely β too low. Re-run with β = 0.1, 0.5, 1.0.")
    elif recon_err >= 0.10:
        print("WEAK — VAE capacity is insufficient.")
        print("  → increase hidden (128) and latent_dim (16)")
    else:
        print("FAIL — VAE cannot represent this policy.")
        print("  → the policy-as-manifold idea requires a different architecture")
    print("=" * 68)
