# agent_status_and_next_steps.md — restart brief for the next agent

**STATUS:** 2026-11-03 by the owner (Rahul). Read this first, then
`OUTPUT_EXP_03.md` for full EXP_03 numbers/comments. Older history:
`29DoF/PROGRESS_LOG.md`, `STATUS_Nov02.md`, `EXP_02_Nov02.md`, `PLAN_A_PASSED.md`.

## Goal (owner's framing)
A CPU-only, "math-first / manifold-first" route to ProtoMotions-class humanoid
control (eventually G1, 29 DoF) — CEM over small policies + compact
generative policy objects, no GPU training.

## Environment facts (save yourself the rediscovery)
- Repo root: `~/MW_Robotics`. System `python3` (3.10) has torch 2.13,
  gymnasium 1.3, mujoco. No venv needed. 12 CPU cores; CUDA is broken/unused
  (harmless "CUDA initialization" warning from torch — ignore).
- **Run scripts from the repo root**, e.g. `python3 domain/exp_e1_vae_policy_compress.py`.
  They do `sys.path.insert(0, os.getcwd())` to import `mw_to_protomotions.*`
  and load `cem_best_theta.npy` / `cem_walk_theta.npy` from cwd.
- Features/actions: `mw_to_protomotions/cem_policy_search.py` →
  `get_features(data)` (15-dim), `policy_action(theta, feat)` = `tanh(W s + b)`, 6 actions, 96 params.
- `run_exp_03.sh` from the EXP_03.md spec was **not built**; as written it `cd`s
  into `domain/` and would fail to find the theta. Not needed.
- CVAE training: ~12 min per model on CPU (200 epochs, ~90k pairs).

## What is proven (numbers to trust)
| Result | Value | Where |
|---|---|---|
| Linear CEM balance, Walker2d | 920.9 steps (seeds of PLAN_A); **755.0 on seeds 2000–2019** | `cem_best_theta.npy` |
| Rank-4 CEM, Walker2d | 896.2 steps (97% of linear) | `29DoF/rank4_walker*` |
| Linear CEM walk, Walker2d | 1.81 m held-out; 1.44 m on seeds 3000–3019 | `cem_walk_theta.npy` |
| **CVAE ← balance policy** | 741 det / **832 stoch** vs 755 original, same seeds | `exp_03_cvae_balance.pt` |
| **CVAE ← walk policy** | **1.85 m** vs 1.44 m original, same seeds | `exp_03_cvae_walk.pt` |
| Balance CVAE on walk task | −0.01 m, stands 569/600 (expected) | E2 log |
| Every Humanoid-v5 (17 DoF, 3D) method | 58–113 steps (random 26) | `29DoF/PROGRESS_LOG.md`, EXP_02 |

**Seed caveat:** always compare against the original policy *on the same seeds*
— seed variance is large (balance policy: 195 → 1000 depending on seed).

## Two open problems (in priority order)
1. **Latent is unused (posterior collapse) in the EXP_03 CVAE.** KL ≈ 0.001/sample;
   perturbing z by 3 units moves actions only 0.08. The CVAE is effectively a
   behaviour-cloning MLP. "VAE as policy" ✅; "searchable policy manifold" ❌ not yet shown.
2. **3D balance wall.** No CPU method gets Humanoid-v5 past ~113 steps. A CVAE
   only imitates its teacher, so there is nothing good to distil at 17 DoF yet.
   This — not policy representation or parameter count — blocks 17/29 DoF.

## Next steps (do in this order)
1. **Build a policy dataset:** 5–10 diverse Walker2d CEM thetas (restarts of
   `run_plan_a.sh` and `run_demo_walker.sh`, different seeds; optionally vary
   the survival weight to get gait variety). ~15 min each; parallelise across cores.
2. **Per-policy-latent CVAE (EXP_03 §4 Option A):** encoder sees a whole
   rollout (or a window) → one `z_policy` ∈ ℝ^8; decoder `a = f(z_policy, s)`.
   **Gate:** KL clearly > 0 and different `z_policy` → measurably different gaits.
   If collapsed again: raise β, shrink decoder's direct access to `s`, or use
   free-bits KL.
3. **Latent search test:** CEM over `z_policy` (8-d) vs CEM over θ (96-d) on the
   walk task; metric = rollouts needed to reach 1.4 m forward. This is the
   load-bearing "manifold as policy" result.
4. **In parallel, one 3D-teacher test** (both are ~30–60 min, listed in
   `29DoF/PROGRESS_LOG.md`): (a) humanoid with ankle actuation, or
   (b) Humanoid-v5 with lateral motion locked (2D-constrained, same 17 joints).
   If (b) recovers Walker2d-level survival, 3D balance is confirmed as the wall.

## Files added this session (all in `domain/` unless noted)
- `exp_e1_vae_policy_compress.py`, `exp_e2_vae_policy_transfer.py` — from EXP_03.md §7
- `exp_e2b_vae_walk_policy.py` — new, walk-trained CVAE + same-seed baseline
- `OUTPUT_EXP_03.md` — full results and commentary
- `logs/exp03_e1_run1.log`, `logs/exp03_e2_run1.log`, `logs/exp03_e2b_run1.log`
- repo root: `exp_03_cvae_balance.pt`, `exp_03_cvae_walk.pt`

## Notes for the agent
- Logged `loss/recon/kl` in `train_cvae` are **sums over batches**, not per-sample.
- Owner prefers confident framing that leads with what's proven, plus plain
