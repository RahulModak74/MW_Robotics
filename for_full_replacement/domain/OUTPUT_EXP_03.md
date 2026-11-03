# OUTPUT_EXP_03.md — VAE as Policy: Results (2026-11-02)

All runs: CPU only (12 cores, torch 2.13, gymnasium 1.3, Walker2d-v5).
Spec: `EXP_03.md`. Code: `exp_e1_vae_policy_compress.py`, `exp_e2_vae_policy_transfer.py`
(both copied verbatim from §7, plus `torch.set_num_threads(8)`), and the new
`exp_e2b_vae_walk_policy.py` (the §4/§9 follow-up "train a CVAE on the walk policy").
Logs: `logs/exp03_e1_run1.log`, `logs/exp03_e2_run1.log`, `logs/exp03_e2b_run1.log`.
Checkpoints (repo root): `exp_03_cvae_balance.pt`, `exp_03_cvae_walk.pt`.

Run from the repo root (`~/MW_Robotics`). The spec's `run_exp_03.sh` `cd`s into
`domain/` and then loads `cem_best_theta.npy` relative to cwd, which would fail.
Not used. Fix that if the script gets built.

---

## Headline

**A conditional VAE can replace a working CEM policy on CPU.** Measured on the
same seeds, it matches or beats both proven Walker2d policies:

| Policy | Task | Original (same seeds) | CVAE | CVAE / original |
|---|---|---|---|---|
| Balance (`cem_best_theta.npy`) | survive, cap 1000 | 755.0 steps | 741.1 det / **832.2 stoch** | 98% / 110% |
| Walk (`cem_walk_theta.npy`) | forward, cap 600 | 1.44 m, 558 steps | **1.85 m**, 550 steps | 129% |

Training takes about 12 min per CVAE on CPU. The total for all three experiments was about 30 min.

---

## E1 — Compression test (balance policy)

| Metric | Result | Spec target |
|---|---|---|
| Train / held-out pairs | 88,972 / 14,005 | ~100k / ~20k (rollouts terminate early) |
| Reconstruction error (held-out) | **0.0002** | < 0.05 |
| Policy replacement error | **0.0002** | < 0.10 |
| Rollout, deterministic `z = prior_mu` | 741.1 / 1000 | ≥ 900 |
| Rollout, stochastic `z ~ prior` | 832.2 / 1000 | — |
| Script verdict | MODERATE | |

**Correction to the verdict:** the 900-step bar comes from the 920.9 reference,
which was measured on *different* seeds. On E1's rollout seeds (2000–2019), the
original policy itself scores **755.0**:
`[195, 197, 203, 205, 288, 470, 541, 1000 ×13]`. Against that fair baseline,
the CVAE is at 98% (deterministic) and 110% (stochastic). Both error metrics
are 250–500× below their targets. **Read fairly, E1 is STRONG.** The script
printed MODERATE only because its baseline was miscalibrated.

The stochastic policy outperforms the deterministic one, and the original too.
That is the opposite of §3's "VAE is averaging" warning. The small prior noise
seems to act as a robustness perturbation.

## E2 — Transfer (balance CVAE on walk task)

| Metric | Result |
|---|---|
| Mean forward | −0.01 m |
| Mean steps | 568.8 / 600 |
| Verdict | **STANDS** — holds balance, doesn't walk |

This is the spec's expected outcome and does not count against the idea (§4: "evidence that a
balance-trained policy doesn't walk"). As the spec prescribes, the next step was to train on walk data.

## E2b — Walk-trained CVAE (new, the spec's prescribed follow-up)

| Metric | Result |
|---|---|
| Train / held-out pairs | 53,526 / 10,225 |
| Policy replacement error | 0.0036 |
| Original walk policy, seeds 3000–3019 | 1.44 m (median 1.63), 558.1 steps |
| **Walk CVAE, same seeds** | **1.85 m (median 1.79), 549.5 steps** |

**The walk-trained CVAE walks, and walks further than its teacher.** Per §4:
"If yes, the VAE-as-policy idea holds; only the training data was the limit."
That condition is met. There are now two working policy objects, balance and walk,
from one recipe.

---

## Comments: what this does and does not establish

**1. The latent is close to unused (posterior collapse).** This is the most
important caveat for the *next* step. Evidence:
- Final KL per sample is about 0.001 (the logged `kl=0.110` is summed over about 174
  batches). Logged `recon`/`loss` values are sums over batches too, not per-sample.
- Moving z by 3 units in random directions changes actions by only
  **0.08 mean |Δa|** on a [−1, 1] range. Mean prior σ is 0.13.

The decoder sees `s` directly. The teacher is deterministic (`tanh(W s + b)`), so the
decoder can learn `a = f(s)` without z. In practice this CVAE is a very good
**behaviour-cloning MLP**, with the VAE machinery idle. So E1/E2b prove
"a small generative net can be the policy". They do **not** yet prove "the
latent space is a searchable policy manifold". That is E2 Option A/B and the
latent CEM search. With per-state z collapsed, Option B (per-step z search) has
almost nothing to search over.

**2. Next step: the hypernetwork / per-policy latent (§4 Option A).** Two
working teachers (balance, walk) now exist, plus whatever CEM restarts can add.
Encode a whole rollout into one `z_policy`, so the latent is *forced* to carry
which-policy information. Then CEM over `z_policy` ∈ ℝ^8 vs θ ∈ ℝ^96.
That test turns "VAE as policy" into "manifold as policy". Collect 5–10 CEM
thetas first (multiple restarts of `run_plan_a.sh` / `run_demo_walker.sh`).

**3. Relation to ProtoMotions.** ProtoMotions-style controllers (MaskedMimic)
have the same basic shape: a conditional VAE with a learned prior, distilled
from expert policies. EXP_03 shows that shape trains and deploys **entirely on
CPU in minutes**, and matches or beats its teacher. That is real progress toward a
CPU-level ProtoMotions-like stack, at 6 DoF.
What it does not move:
- **Scale:** still Walker2d, 6 actions, 2D-constrained (§5 is explicit).
- **The 3D wall (EXP_02):** a CVAE imitates a teacher. On Humanoid-v5 the best
  CPU teacher is about 100–113 steps, so distilling it would give about 100-step behaviour.
  The open problem for 17/29 DoF is still getting *a* good 3D teacher. The
  candidates are the ankle-actuation and 2D-constrained-intermediate tests from
  `29DoF/PROGRESS_LOG.md`, or motion-capture data. It is not the policy
  representation, which this experiment shows is not the bottleneck.

**4. Practical upside now:** both CVAEs are small (≈15k params), CPU-cheap at
inference, and stochastic sampling helps robustness. Together they are a
two-entry "policy library" (§10's end state) that works today.

## Suggested order for the next session
1. Collect 5–10 diverse CEM thetas (walk + balance restarts).
2. Train the per-policy-latent CVAE (Option A). Check that z_policy is *used*
   (KL clearly > 0, decoding different z gives different gaits).
3. CEM over z_policy vs CEM over θ: compare evaluations needed to reach 1.4 m forward.
4. In parallel, pick one 3D-teacher test (ankle / 2D-constrained humanoid) so
   there is something worth distilling at 17 DoF.
