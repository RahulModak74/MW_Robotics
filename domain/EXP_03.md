
# EXP_03.md — VAE as Policy, Minimal Form

**One-sentence claim.** A Variational Autoencoder trained on `(state, action)`
pairs from a working robot policy can act as that policy: given a state,
it generates the action. If this works, the VAE *is* a policy object — a
small, generative, manifold-constrained one — and the entire "policy as
trained manifold" framing becomes testable without fibre bundles, algebra,
or conditional dynamics.

**Everything else in the project is downstream of this test.**

---

## 0. Why this experiment, and not another spec

Three prior experiments are relevant, and none of them answered this question:

- **Plan B (joint VAE over `(s, a, s')`)** failed at the E-gate because
  the joint formulation ignores the action. It tested "manifold of
  transitions", not "VAE of a policy".
- **`run_plan_a.sh` (CEM over linear policy)** proved a linear policy
  works. It did not test whether a VAE could represent that policy.
- **The "prior family" reframe** proposed a family of VAEs indexed by
  priors. It has not been tested at all.

The question these leave open, and which is cheapest to answer, is:

> **Can a VAE compress a working policy into a small generative model
> that produces the same actions?**

If yes: we have a new policy object. It has ~8–16 latent dimensions instead
of 96+ parameters. Searching over the latent is cheaper than searching
over the parameters. The family idea becomes plausible.

If no: the VAE cannot represent a policy, and everything downstream is
premature. Better to learn that today for 1 hour of CPU than in a month
of building.

---

## 1. The Minimal VAE-as-Policy

Forget fibre bundles. Forget `ManifoldAlgebra`. Forget `(s, a, s')`.

The simplest conditional VAE for a policy is:
Encoder q(z | s, a) — latent from state-action pair
Prior p(z | s) — latent from state alone
Decoder p(a | z, s) — action from latent and state

text

Training objective (standard ELBO):
L = E_{z ~ q(z|s,a)} [ ||a - decode(z, s)||² ] + β · KL( q(z|s,a) || p(z|s) )

text

At inference, the policy is:

```python
def policy(s):
    z = prior_mu(s)        # or: sample z ~ prior(s) for stochastic policy
    a = decode(z, s)
    return a
This is called a conditional VAE or CVAE, and it is the correct
architecture for "VAE of a policy". It is not the same as the joint VAE
that failed in Plan B. The key difference: a is the decoder's output
and only that. There is no path for the decoder to ignore a and
memorise s', because s' is not an output at all.

Why this specific architecture is the right first test:

It is well-known (CVAE is standard in imitation learning, e.g., "CVAE-BC").

It has one knob (latent_dim) you can sweep.

It has one training signal (reconstruction + KL), which you can
diagnose separately.

It has one inference mode (z = prior_mu(s)), which is deterministic.

If it works, it is obviously a policy object. No ambiguity.

2. The Load-Bearing Question
Can a CVAE trained on (s, a) from a good policy reproduce that
policy's actions?

Not "does it reduce loss on a validation set" — that is a proxy. The
literal question: for every state s in a held-out rollout, is
decode(prior_mu(s), s) close to π_original(s)?

If the CVAE matches π_original on ≥ 80% of held-out states (with a
tolerance appropriate to the action space), the policy has been
compressed. If it matches on < 50%, it hasn't.

This is a direct test of the policy object, and it is
unambiguous — the original policy is a ground truth we already have.

3. Experiment E1 — Compression Test
Setup.

Ground-truth policy: cem_best_theta.npy (the 920-step balance policy).

Training data: 100 rollouts × 1000 steps = ~100,000 (s, a) pairs from
this policy. Collected by running the policy in Walker2d.

Held-out data: 20 fresh rollouts × 1000 steps = ~20,000 pairs, seeds disjoint.

Action space: 6 torques, range [-1, 1].

State space: 15 features (get_features from cem_policy_search.py).

Training.

Conditional VAE with latent_dim = 8, hidden = 64, KL weight β = 0.01.

200 epochs, batch size 512.

Target: reconstruction loss (MSE on a) below 0.05.

Evaluation.

Reconstruction error on held-out pairs. Encode (s, a_actual),
decode, measure ||a_actual - decode(z, s)||. This is the "does it
have enough capacity" check.

Policy replacement error. For each held-out state s, compute
a_vae = decode(prior_mu(s), s) and a_policy = π_original(s).
Measure ||a_vae - a_policy||. This is the load-bearing metric.

Rollout divergence. Run a_vae policy in Walker2d for 1000 steps.
Compare survival steps to the original policy's 920.9. If the CVAE
survives 900+, it is functionally equivalent.

Deterministic vs stochastic. Also try z = sample(prior(s)) (a
stochastic policy). Measure whether stochastic is better or worse than
deterministic. If stochastic is much worse, the latent doesn't
capture the policy well — it means the policy is one specific action
per state, and the VAE is averaging.

Interpretation:

Reconstruction error	Policy replacement error	Survival	Verdict
< 0.05	< 0.1	≥ 900	STRONG: VAE is a valid policy object.
< 0.05	0.1–0.3	500–900	MODERATE: compression is lossy but usable.
< 0.05	> 0.3	< 500	AMBIGUOUS: VAE encodes (s,a) but cannot act from s alone. Fixable (see §6).
> 0.1	any	any	WEAK: VAE capacity is insufficient. Fixable (bigger VAE).
any	> 0.5	< 100	FAIL: this VAE cannot represent a policy of this complexity.
Wall-clock: ~10 min training + ~2 min eval.

4. Experiment E2 — Latent Search (only if E1 passes)
Only run this if E1's verdict is STRONG or MODERATE. If E1 fails,
the "VAE as policy" idea is dead and there is nothing to search.

Question. If the VAE is a valid policy representation, is searching
over its latent space cheaper than searching over the original parameters?

Setup.

Fix the trained CVAE from E1.

Give it a new task: walk forward (Walker2d, forward-distance fitness
with survival weight 0.02 — same as run_demo_walker.sh).

Search dimension: instead of searching θ ∈ ℝ^96 (the linear policy),
search a small per-policy latent z_policy ∈ ℝ^16 (a global
latent that parametrises the whole VAE).

At each step, the resulting action is decode(z_policy, s) where the
state s varies. z_policy is fixed per rollout.

Wait — this requires the CVAE's decoder to be conditioned on both a
per-step z_state (from the prior) and a per-policy z_policy (the
search variable). That's not the CVAE from §1. Let me reframe.

Two ways to make this work:

Option A (per-policy latent, simplest). Train a hypernetwork-style
VAE where the encoder sees a full rollout (all (s, a) pairs from
one policy), outputs a single z_policy, and the decoder uses
z_policy + s to produce a. This is a policy-as-latent object.
Training data: many (policy, its rollouts) pairs. We only have one
policy (the balance one), so this needs Phase 2 data — but we can
generate it by re-running CEM and saving multiple restarts' thetas.

Option B (reuse E1's CVAE, search over the per-state prior mean).
At each control step, we search for z_t such that decode(z_t, s_t)
produces a good action. This is a per-step search — like MPC. Search
dimension is small (8 per step) but inference cost is high (CEM over 8
dims at 50 Hz).

Option C (encode, don't search). Just use the CVAE from E1 as the
policy. No search. Measure performance on the walk task. If it's
zero, the balance-trained CVAE can't walk — that's expected and fine.
If it's non-zero, the CVAE learned something general about "good Walker2d
actions", not just "balanced Walker2d actions".

E2 should test Option C first — it's free and answers whether the
CVAE generalises at all. Option B and A are follow-ups if C shows promise.

E2 test:

Load the CVAE from E1.

Run it as a policy on Walker2d with forward-walk fitness.

Measure: mean forward distance, survival steps.

Compare to: the linear policy's 1.33 m forward, the balance policy's
near-zero forward.

Wall-clock: 30 min.

Interpretation:

Result	Verdict
≥ 1.0 m forward	The balance-trained CVAE walks. Strong generalisation.
0.2–1.0 m forward	Partial transfer. The CVAE has some general Walker2d prior.
< 0.2 m forward, survives 500+ steps	It stands but doesn't walk — expected. Not a failure.
< 0.2 m forward, falls quickly	The CVAE is overfit to balance states. Diagnose.
Critical: if the CVAE stands but doesn't walk, this is not evidence
against the VAE-as-policy idea — it is evidence that a balance-trained
policy doesn't walk. The right follow-up is to train a second CVAE on
the walk policy's data and test whether that CVAE walks. If yes,
the VAE-as-policy idea holds; only the training data was the limit.

5. What This Experiment Does Not Test
Being explicit so nobody conflates the results later:

Not testing the "policy family" reframe. That's a separate question
(does a family of priors give a family of policies?). This experiment
tests one VAE, on one task, with one prior configuration.

Not testing "manifold as physics constraint." The MW manifold
(validator) is a separate object. The CVAE here is a policy, not a
validator. They are not the same thing.

Not testing "scale to 17 or 29 DoF." Everything here is Walker2d
(6 actions). Scaling is a follow-up only if E1 and E2 both work.

Not testing the fibre-bundle or ManifoldAlgebra infrastructure. That
is reserved for later. The CVAE here is a single torch.nn.Module.

6. Known Failure Modes and Fixes
If E1 fails, the diagnosis determines the fix. Do not skip the diagnosis;
do not "just make the VAE bigger" without knowing which failure mode you
are in.

Failure A: Reconstruction is fine, but action-from-state is bad
Symptom: q(z|s,a) reconstructs (s,a) well, but decode(prior_mu(s), s)
gives nonsense actions.

Cause: The prior p(z|s) has not learned to concentrate around the
latent that the encoder produces for the actual action. The β is too
low, so the model essentially ignores the prior and behaves like an
autoencoder. At inference, prior_mu(s) gives an arbitrary latent, and
the decoder produces arbitrary actions.

Fix: Increase β from 0.01 to 0.1, 0.5, or 1.0. Sweep it.
Higher β forces the prior and posterior to agree.

Failure B: The VAE averages
Symptom: Reconstruction is fine, action-from-state is sometimes
fine, but individual actions are never the same as the original —
they're always close to the mean.

Cause: The policy is deterministic (one action per state), but
the VAE is stochastic (a distribution of actions per state). It learns
the mean of the distribution, which may be a compromise that satisfies
none of the states well.

Fix: Reduce latent_dim from 8 to 4 or 2. Smaller latent → less
capacity to model the "spread", more capacity to model the mean. This is
counterintuitive but it's the correct fix for a deterministic target
policy.

Failure C: Under-capacity
Symptom: Reconstruction error is high. The VAE cannot represent
(s, a) pairs even in training.

Cause: Hidden layer too small, latent too small, or the state isn't
informative enough.

Fix: Increase hidden from 64 to 128. Increase latent_dim from 8
to 16. Add more training data.

Failure D: Generalisation gap
Symptom: Reconstruction is fine on training data but degrades badly
on held-out data.

Cause: The VAE memorised the training rollouts.

Fix: More training rollouts. Or dropout. Or stronger KL regularisation.
Or, most likely: the state vector doesn't capture all the info the
original policy uses (check get_features).

Each failure mode has a specific, cheap fix. None of them requires
re-architecting. The experiment is designed to distinguish these cases,
which is why the metrics are separate (reconstruction error, policy
replacement error, rollout survival).

7. Files to Build
Three files, in this order. Each is a self-contained script.

domain/exp_e1_vae_policy_compress.py
python
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
domain/exp_e2_vae_policy_transfer.py
Only run if E1's verdict is STRONG or MODERATE.

python
#!/usr/bin/env python3
"""
EXP_03 / E2 — Does the balance-trained CVAE generalise to a walk task?
Tests whether "VAE as policy" generalises when the task changes.

Loads exp_03_cvae_balance.pt, runs it as a policy on Walker2d with the
forward-walk task. Expected outcomes:
  - ≥ 1.0 m forward: the CVAE learned general Walker2d behavior
  - 0.2–1.0 m: partial transfer, the manifold has a "good action" prior
  - < 0.2 m but survives 500+: it stands, doesn't walk (expected, not a failure)
  - < 0.2 m and falls: overfit to balance, diagnose
"""
import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
import gymnasium as gym
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_e1_vae_policy_compress import CVAE, eval_rollout
from mw_to_protomotions.cem_policy_search import get_features


def eval_walk(model, seeds, max_steps=600):
    """Runs the CVAE as a policy with forward-distance measurement."""
    fwd, steps = [], []
    for seed in seeds:
        env = gym.make("Walker2d-v5", exclude_current_positions_from_observation=False)
        obs, info = env.reset(seed=seed)
        data = env.unwrapped.data
        x0 = data.qpos[0]
        s_count = 0
        for _ in range(max_steps):
            s = torch.as_tensor(get_features(data)[None], dtype=torch.float32)
            act = model.act_deterministic(s).numpy()[0]
            obs, r, term, trunc, info = env.step(act)
            s_count += 1
            if term:
                break
        fwd.append(float(data.qpos[0] - x0))
        steps.append(s_count)
        env.close()
    return np.array(fwd), np.array(steps)


if __name__ == "__main__":
    print("EXP_03 / E2 — VAE Policy Transfer: balance-trained CVAE on walk task")
    print("=" * 68)

    # Load the CVAE from E1
    path = "exp_03_cvae_balance.pt"
    if not os.path.exists(path):
        sys.exit(f"ERROR: {path} not found — run exp_e1_vae_policy_compress.py first")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = CVAE(**ck["config"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    print(f"loaded CVAE from {path}\n")

    # Run the CVAE as a policy on the walk task
    print("[1/2] running balance-trained CVAE on walk task (20 fresh seeds)…")
    fwd, steps = eval_walk(model, seeds=list(range(3000, 3020)))
    print(f"  mean forward: {fwd.mean():.2f} m")
    print(f"  mean steps:   {steps.mean():.1f} / 600")
    print(f"  reference:    linear walk policy = 1.33 m, "
          f"linear balance policy ≈ -0.75 m\n")

    # Verdict
    print("[2/2] verdict")
    print("=" * 68)
    if fwd.mean() >= 1.0:
        print("TRANSFERS — CVAE walks without walk-specific training.")
        print("  → the balance-trained manifold contains general walker2d behavior")
        print("  → strong evidence for 'VAE as policy, trained on any good data'")
    elif fwd.mean() >= 0.2:
        print("PARTIAL — CVAE moves forward somewhat without being trained to.")
        print("  → likely the CVAE encodes 'standing upright' behavior which "
              "translates to slow forward lean")
        print("  → train a walk-trained CVAE next for a real comparison")
    elif steps.mean() >= 500:
        print("STANDS — CVAE holds balance but doesn't walk.")
        print("  → this is expected; the balance CVAE was never trained to walk")
        print("  → the interesting follow-up: train CVAE on walk data, "
              "does it walk?")
    else:
        print("FAILS — CVAE cannot act as a walk policy.")
        print("  → check whether act_deterministic(s) produces "
              "physically-reasonable actions on walk-task states")
        print("  → likely the balance-trained distribution is too narrow")
    print("=" * 68)
domain/run_exp_03.sh
bash
#!/usr/bin/env bash
# run_exp_03.sh — VAE as Policy, minimal form.
# Runs E1 (compression) and optionally E2 (transfer).
#
#   ./run_exp_03.sh              # E1 only, ~15 min
#   ./run_exp_03.sh --with-e2    # E1 + E2, ~45 min
#   ./run_exp_03.sh --dry-run

set -euo pipefail
if [[ -t 1 ]]; then GRN=$'\033[32m'; RED=$'\033[31m'; YEL=$'\033[33m'; BLU=$'\033[34m'; RST=$'\033[0m'
else GRN=""; RED=""; YEL=""; BLU=""; RST=""; fi
say(){ printf "%s [%s] %s\n" "$(date -u +%H:%M:%SZ)" "${BLU}exp03${RST}" "$*"; }
ok(){  printf "%s [%s] %s\n" "$(date -u +%H:%M:%SZ)" "${GRN}exp03${RST}" "$*"; }
die(){ printf "%s [%s] %s\n" "$(date -u +%H:%M:%SZ)" "${RED}exp03${RST}" "$*" >&2; exit 1; }

WITH_E2=0; DRY=0
for a in "$@"; do case "$a" in
    --with-e2) WITH_E2=1 ;;
    --dry-run) DRY=1 ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) die "unknown arg: $a" ;;
esac; done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p logs

[[ -f ../cem_best_theta.npy ]] || die "cem_best_theta.npy missing — run ./run_plan_a.sh first"
[[ -f exp_e1_vae_policy_compress.py ]] || die "exp_e1 file missing — paste it from EXP_03.md §7"
[[ $WITH_E2 -eq 1 ]] && { [[ -f exp_e2_vae_policy_transfer.py ]] || die "exp_e2 file missing"; }

PY="${PYTHON:-python3}"
"$PY" -c "import gymnasium, torch, numpy" || die "missing deps"
export MW_DEVICE=cpu CUDA_VISIBLE_DEVICES=""

if [[ $DRY -eq 1 ]]; then
    say "dry-run: would run E1 ($([[ $WITH_E2 -eq 1 ]] && echo 'then E2'))"
    exit 0
fi

STAMP=$(date -u +%Y%m%d_%H%M%S)

say "E1 — VAE as Policy, compression test (~15 min)"
"$PY" exp_e1_vae_policy_compress.py 2>&1 | tee "logs/exp03_e1_${STAMP}.log"

if [[ $WITH_E2 -eq 1 ]]; then
    say "E2 — VAE transfer to walk task (~5 min)"
    "$PY" exp_e2_vae_policy_transfer.py 2>&1 | tee "logs/exp03_e2_${STAMP}.log"
fi

say "done. logs: logs/exp03_*_${STAMP}.log"
8. Order of Operations, Wall-Clock
Step	Time	Output
E1 data collection	5 min	100k (s, a) pairs
E1 training	8–10 min	CVAE checkpoint
E1 evaluation	2 min	reconstruction + replacement + rollout
E1 total	~15 min	verdict
E2 transfer	5 min	walk performance
E1 + E2 total	~20 min	full picture
Run E1 first. Do not proceed to E2 until you've read the E1 verdict
and, if it's AMBIGUOUS or WEAK, run at least one sweep (latent_dim,
beta) to diagnose which failure mode you're in.

9. What to Do With Each Outcome
E1 = STRONG (recon < 0.05, replacement < 0.1, rollout ≥ 900)
The VAE is a valid policy object. This is the win.

Next steps, in order:

E2: does it transfer? (Free, 5 min.)

Train a CVAE on the walk policy (same recipe, walk-task data).
Does that CVAE also compress cleanly? If yes, we have two policy
objects and a real dataset of "policies as latents".

Train a hypernetwork-style CVAE (encoder sees the whole rollout,
outputs a single z_policy; decoder uses (z_policy, s)). This is
the "policy as latent" version that makes search possible.

Latent search: CEM over z_policy in ℝ^8 or ℝ^16. Does it
reach the linear policy's performance in less time?

E1 = MODERATE (recon < 0.05, replacement < 0.3, rollout ≥ 500)
The VAE is a policy object but lossy. Before scaling:

Sweep latent_dim in {4, 8, 16, 32}. Plot reconstruction and
replacement errors. Find the elbow.

Sweep beta in {0.01, 0.1, 0.5, 1.0}. Higher beta → better
prior agreement → better action-from-state.

Re-run E1 with the best pair. If it becomes STRONG, proceed. If it
stays MODERATE, note the ceiling and decide whether a lossy policy
object is still useful.

E1 = AMBIGUOUS (recon < 0.05, replacement ≥ 0.3)
The VAE encodes (s, a) jointly but cannot act from s alone. This is
the "prior didn't learn" failure mode.

Fix by increasing beta to 0.1, then 0.5, then 1.0. Re-run. Report
which beta fixes it (or that none does).

E1 = WEAK (recon ≥ 0.10)
VAE capacity is insufficient. Increase hidden to 128, latent_dim to

Re-run. If still weak, the policy is too complex for this VAE
architecture; that's an informative negative result.

E1 = FAIL (replacement ≥ 0.5, rollout < 100)
The VAE cannot represent this policy. Either:

The architecture is wrong (try a normalizing flow instead of a VAE)

The task is wrong (try a simpler policy to compress first)

The idea is wrong (VAE-as-policy doesn't work at this complexity)

This is the "Plan B failed" outcome for the simpler question. Publishable.

10. What "Manifold as Policy" Actually Means Here
The phrase has been ambiguous. Let me pin it down for this experiment:

Not: "the MW validator manifold acts as the policy" (that was Plan B, failed)

Not: "a family of prior-varied manifolds is a policy family" (untested, separate)

Yes: "a conditional VAE trained on (s, a) is a policy object that
represents a learned policy compactly."

This is what "manifold as policy" reduces to in the minimal case:

The VAE learns a low-dimensional latent manifold of (s, a) pairs.

The decoder is a map from latent + state to action.

Given a state, the VAE "acts" by decoding a latent.

The latent space is the "policy manifold" — a small parameterisation
of a full policy.

The interesting question is not "does this work for one policy" (it
should, for a simple enough policy) but "does the compressed policy
generalise across tasks, and is it cheaper to search over?" E1 tests
the first half (compression). E2 and the follow-ups test the second
half (searchability).

The end state, if all goes well:

"A trained CVAE is a policy object. A small number of these (5–20),
each trained on a different task/policy, forms a policy library.
At runtime, retrieve the closest and refine its latent with CEM. This
is the CPU-first, manifold-first alternative to gradient-trained
policies at deployment."

That is the version of "manifold as policy" that is defensible, honest,
and cheap to test. EXP_03 tests whether the first sentence is even true.

11. Honest Probability
Outcome	Probability	Why
E1 STRONG (recon < 0.05, replacement < 0.1, rollout ≥ 900)	40–50%	CVAE is a standard imitation learning architecture; it works on similar problems in the literature. The 920-step balance policy is a relatively simple, deterministic policy, which is the easier case.
E1 MODERATE (recon < 0.05, replacement < 0.3, rollout ≥ 500)	25–30%	Likely outcome if the VAE compresses but loses some fidelity.
E1 AMBIGUOUS (encodes but cannot act)	15–20%	Likely if beta = 0.01 is too low. Fixable by sweeping.
E1 WEAK (recon ≥ 0.10)	5–10%	The linear policy is fairly simple; a 64-hidden CVAE should be able to reconstruct it.
E1 FAIL (replacement ≥ 0.5)	< 5%	Only if the architecture is fundamentally wrong for this task.
If E1 reaches STRONG or MODERATE, the follow-up questions (E2, latent
search, hypernetwork VAE) become worth pursuing. The probabilities there
are lower — the transfer test is 30–40%, and the latent search is 20–30%
— but each is conditional on the previous, and each is cheap.

The floor: even if every phase after E1 fails, we learn "VAE can
compress a policy" — which is itself a useful result, and the basis for
a small "policy library" pattern regardless of the search story.

The ceiling: if everything works, "VAE as policy, searchable latent,
CPU-deployable" becomes a defensible product direction.

Either way, E1 is worth 15 minutes.

12. What This Experiment Does Not Include
Explicitly not in this spec:

ManifoldAlgebra — the product/fibre/glue infrastructure is out of scope

Transition manifolds (s, a, s') — this is (s, a) only

Prior sweeps / policy family — one prior, one task

17 or 29 DoF — Walker2d (6 actions) only

ONNX export — torch checkpoint only

Flask demo — headless eval only

Physics priors / validator role — no MW validator is used here; the
CVAE is a policy, not a validator

These are the correct next steps once E1 succeeds. Running E1 first,
and only then adding complexity, is the whole point of this spec.

13. Where This Fits
The project's experiment history:

Experiment	Question	Answer
Walker2d linear + CEM	Can CEM learn a policy?	✅ 920.9 steps
Walker2d rank-4 + CEM	Does rank-reduction work?	✅ 896.2 steps
Humanoid-v5 flat/rank-4	Does it scale to 3D?	⚠️ 97–113 steps (EXP_02 in flight)
Plan B joint VAE	Manifold of transitions as policy?	❌ Failed E-gate
EXP_03 (this file)	Can a VAE compress a policy?	Pending
EXP_03 follow-up	Is the compressed policy searchable?	Blocked on E1
EXP_03 follow-up	Does the policy family exist?	Blocked on E1
EXP_03 is the minimal version of "manifold as policy". It is the cheapest
experiment that answers whether the phrase means anything at all. Run it
before writing the conditional VAE spec, before the prior-sweep spec,
before anything else. The answer will either open the follow-up work with
confidence or save a month of speculation.
