# SPEC: Plan B — Policy as Manifold, MPC as Bayesian Posterior Sampling (CPU-First, HW-Validated)
plan A is both MW and CEM-ES.. Dynamics replaced by MW manifold. Score candidates by manifold distance, not full rollouts.

Policy replaced by CEM/ES. Small head on a frozen encoder, searched.

Library stores both the policy distribution and the manifold checkpoint.

Runtime at T0/T1 retrieves and executes; T2 refines. but we are doing plan B now -- Rahul


This spec is the radical sibling of the Plan A spec. **Plan A** kept a small CEM-searched policy and used the manifold as a cheap scorer. **Plan B** removes the policy NN entirely: the manifold is the model, the reward is a rank function on it, and control is a Bayesian posterior sample over admissible actions. This is your stated focus — Bayesian, manifold-first, no learned controller between you and physics.

The spec is written to plug directly into your existing codebase (`mw_framework/`, `walker_domain.py`, `datagen.py`, `ManifoldAlgebra`, `LorentzianCone`, `stage3_reference_test.py`) and to be implementable as code by an agent without asking questions.

---

## 0. Scope and the honest bet

**Plan B in one sentence.** Train a single manifold `M` over physically-admissible **transitions** `(s, a, s')` from real (sim-then-hardware) physics. At runtime, the controller **samples admissible actions** from `M(s_t, ·)`, **ranks them by task reward**, and **executes the best**. There is no policy network, no CEM over policy parameters, and no learned forward model other than `M` itself.

**Why this is a genuinely different object from Plan A.**

| | Plan A (previous spec) | Plan B (this spec) |
|---|---|---|
| Manifold over | states `s` | transitions `(s, a, s')` |
| Policy | small searched network | **none** — implicit in `M` + reward |
| Search at runtime | CEM over policy parameters (96–1300-dim) | Bayesian sampling over **actions** (6–30-dim) |
| Role of MW distance | scorer inside search | **prior** on admissibility + **penalty** on reward |
| What the manifold is | validity check | **the physics + the policy distribution** |

**The bet, stated as a falsifiable hypothesis.** The optimal action at any state lies on the admissible-transition manifold `M(s, ·)` — i.e. the policy that naturally satisfies physics is (near) optimal for the task, and off-manifold actions are never needed. If true, sampling on `M` is enough. If false — if the optimal action for some task lives just off the manifold — Plan B is structurally suboptimal and no amount of search on `M` will escape it.

**This spec exists to test that hypothesis cheaply, then either harden it (if true) or abandon it in favor of Plan A (if false).**

---

## 1. Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  M — TRANSITION MANIFOLD                                            │
│  Learned from MuJoCo triples (s, a, s'), physics priors injected    │
│  M(s, ·) = fibre over state = distribution of admissible actions    │
│  M(·, ·, s') = preconditioner = which states can reach s'           │
│  Distance d_M = Riemannian, physics-grounded (existing geometry.py) │
└─────────────────────────────────────────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
       ┌─────────────────────┐   ┌─────────────────────┐
       │ RUNTIME QUERY (B1)  │   │ SHORT-HORIZON       │
       │ 1-step posterior    │   │ PLANNER (B2, MPC)   │
       │ sample actions from │   │ samples H-step      │
       │ M(s_t, ·), rank by  │   │ action sequences,   │
       │ R(s_t, a), execute  │   │ rolls out on M,     │
       │ best                │   │ ranks by cumulative │
       └─────────────────────┘   │ R, executes first   │
                                 │ action              │
                                 └─────────────────────┘
                                │
                                ▼
                    ┌─────────────────────┐
                    │  REAL ROBOT         │
                    │  (sim first, then   │
                    │  hardware; never    │
                    │  online learning)   │
                    └─────────────────────┘
```

**Two runtime modes**, both pure manifold:

- **B1 — Posterior action sample.** Cheapest. One VAE pass per control step. Assumes the optimal action is in `M(s_t, ·)` and `R(s, a)` is dense enough.
- **B2 — Short-horizon MPC on the manifold.** Uses `M` as the forward model. Sample `N` action sequences of length `H`, rollout each on `M`, rank by cumulative reward + MW distance, execute first action, re-plan. This is Plan B's honest workhorse for sparse-reward locomotion tasks.

**The spec implements B1 first (cheap to validate), then B2 (the real workhorse).**

---

## 2. The manifold object — code-level definition

This section matches your code style exactly.

### 2.1 New domain class: `TransitionVAE`

New file: `mw_framework/domains/transition.py`

```python
FEATURES = ["s_0", ..., "s_n",   # current state (n dims — posture, velocity, phase)
            "a_0", ..., "a_m",   # action (m dims — joint torques)
            "sp_0", ..., "sp_n"] # next state (n dims)

class TransitionVAE(MWVAE):
    # physics():
    #   C1: s' is reachable from s in one dt under a — approximate reachability prior
    #       (e.g. ||s' - s|| <= v_max * dt componentwise, or a soft kinematic bound)
    #   C2: |a| <= a_max — actuator limits from the robot XML (same pattern as walker_domain.py)
    #   C3: joint limits on s and s' — exact, from MuJoCo jnt_range
    #   C4 (optional, LorentzianCone): s' must be "future" of s — enforce timelike
    #       separation using geometry.LorentzianCone logic embedded as a soft prior
```

**Key design choice:** the manifold is **joint over `(s, a, s')`**, not `(s, a)` separately. This is what makes it a **dynamics manifold**, not a state manifold. It carries the physics of "what transitions are real."

### 2.2 Two derived operations — critical to Plan B

These are **computed on the trained `TransitionVAE`**, not trained separately.

**`inverse_query(M, s, s') → p(a | s, s')`** — the core primitive for B1.
- **Implementation (no new network):** solve `a* = argmin_a d_M((s, a, s'))` by a 1-step CEM over `a` only. **This is small** — `m = 6` for Walker2d, `m ≈ 30` for Humanoid. Population 32, 3 generations, ~96 VAE forward passes. On CPU, ~10–50 ms.
- **Why it's cheap:** the search dimension is actions, not policy parameters. This is the key win of Plan B over Plan A.

**`forward_query(M, s, a) → p(s' | s, a)`** — the primitive for B2 rollouts.
- **Implementation:** encode `(s, a)`, then sample `s'` from the decoder's marginal at the `s'` block. In a transition VAE, this is a single decoder pass with the `s'` block read out. ~1–3 ms per sample on CPU.

### 2.3 Reusing `ManifoldAlgebra` — the fibre that makes B1 legal

`mw_framework/algebra.py` already has `fibre(name, features, over=base)`. Plan B's manifold is **structurally a fibre bundle**:

```
base:   state s
fibre:  action a conditioned on s
outer:  next state s' conditioned on (s, a)   [second-level fibre]
```

Build it as:
```python
alg = ManifoldAlgebra(feature_names=FEATURES)
alg.product("state", state_features, latent_dim=4)
alg.fibre("action", action_features, latent_dim=3, over="state")
alg.fibre("next_state", next_state_features, latent_dim=4, over="action")
alg.glue("reachability", <reach constraint fn>, tolerance=...)
alg.check_complete()
```

**This is important:** the fibre-over-base structure encodes exactly the semantics Plan B needs — "actions admissible **given** state." The `marginal_distance(x, keep=["action"])` call gives you the fibre-only distance used in §3.3. No new infrastructure needed.

---

## 3. Data generation

### 3.1 Source — same recipe as battery PyBaMM-like generators

Use the exact recipe already established in `mujoco_experiments/03_mujoco_like_datagen.py`: **many independent "units" with continuously-varying physical parameters, densely sampled**, not a sparse grid.

For Walker2d:
- 80 independent MuJoCo instances, each with `friction ~ U[0.4, 1.3]`, `damping ~ U[0.05, 0.35]`, `height ~ U[0.8, 2.0]` (initial condition)
- Each instance rolled out for 250 steps under the existing PD controller from `datagen.py`, logging `(s, a, s')` at **every step**
- **Augmentation (Gap 1 from §MW-MuJoCo discussion):** apply the explicit prior filters before training:
  - reject `a` where `|a| > a_max` (read from `model.actuator_forcerange`)
  - reject `s'` where `jnt_range` violated
  - reject `s'` where `z < 0` (ground penetration)
  - reject transitions where `||s' - s|| > v_max · dt` (kinematic reachability)

Target: **~500k–2M valid transitions per robot body.**

### 3.2 For Humanoid-v5 (stretch)

Same recipe. Higher dimension means more data. Target **~2M–5M**. See §9 for the dose-response curve that decides whether this is worth it.

### 3.3 Hardware data (Phase 2 only — after Phase 1 passes)

The spec must not be gamed by sim-only training. Phase 2 requires **real robot transitions**, collected with the same feature schema, from:
- ≥ 20 physical units of the target robot (the "95 LFP cells" analogue)
- ≥ 5 environmental conditions (indoor, outdoor, wet floor, low friction mat, slope)
- ≥ 5 task categories (walk forward, turn, recover from push, stand, stairs)

**Blind prospective protocol** (matching the battery blind study):
- Feature extraction pipeline frozen before hardware collection
- MW predictions filed with a timestamp before the physical trial
- Scored on coverage (did the 90% credible band contain the actual outcome ≥ 85% of the time?) and discrimination (ROC-AUC on the failure classification)

---

## 4. Training

### 4.1 Phase 1 (sim only)

```python
# Pseudocode matching your codebase
set_seed(0); pyro.clear_param_store()
df = collect_transitions(n_units=80, n_steps=250, seed0=100)  # ~1M rows
df = filter_transitions(df)                                    # §3.1 priors
normal = df.sample(frac=0.8)                                    # 80/20 holdout
det = MWDetector(vae=TransitionVAE(), epochs=200, lr=2e-3).fit(normal[FEATURES])
det.save("transition_mw_walker.ckpt")
```

### 4.2 Critical training hyperparameters

| Knob | Value | Why |
|---|---|---|
| `latent_dim` (state) | 4–6 | Preserve physics structure, don't compress below intrinsic dim |
| `latent_dim` (action fibre) | 3–4 | Actions live on a low-dim manifold too |
| `latent_dim` (next-state fibre) | 4–6 | Same as state — it's the same physical quantity |
| `obs_sigma` | per-feature tier (1/2/3) | Existing `TIER_SIGMA` convention |
| `kl_weight` | 0.1 (default) | Known-good from your existing domains |
| `law_tolerance` | 0.05–0.1 | Loose enough to survive noise, tight enough to reject nonsense |

### 4.3 Validation during training

**Required metrics reported in training log** (matching your `stage1_validator_test.py` pattern):

1. **Reconstruction floor** — held-out `(s, a, s')` MW distance: `< 1.5×` training baseline.
2. **Fault detection AUC** — reuse `mw_framework.core.roc_auc`. Injected fault types:
   - `joint_limit` — `s'` outside joint limit (should hit ~1.00)
   - `teleport` — `s'` far from `s` in one step (should hit ~1.00)
   - `action_saturation` — `|a| > a_max` (should hit ~1.00)
   - `unphysical_recovery` — `s'` "heals" energy beyond damping (should hit ~0.9+ via `LorentzianCone` logic)
3. **Organic fault** — reuse the near-fall rows from `datagen.py`. Target ROC-AUC **> 0.85** (their Stage 1 achieved 1.000 on Walker2d, so this is a floor not a stretch).

---

## 5. Runtime — the query loop

### 5.1 B1 — Posterior action sample (cheapest, validate first)

```python
def b1_step(det, s_t, goal, n_samples=64, prior_tol=0.5):
    # 1. Query the action fibre
    candidates = sample_action_fibre(det, s_t, n_samples)   # 1 decoder pass, ~1ms

    # 2. Filter by MW distance (reject off-manifold)
    dists = [det.mw_distance(concat(s_t, a, decode_next(s_t, a))) for a in candidates]
    keep = [a for a, d in zip(candidates, dists) if d < prior_tol]

    # 3. Rank by task reward
    rewards = [R(s_t, a, goal) for a in keep]
    return candidates[argmax(rewards)]
```

**This is the honest version of "the manifold IS the policy."** No policy NN, no CEM over parameters, just sampling from the fibre and ranking by reward.

**Cost per control step:** ~5–20 ms on CPU. Fits the 50 Hz control rate (200 ms budget) with 10× headroom.

### 5.2 B2 — Short-horizon MPC on the manifold (the real workhorse)

```python
def b2_plan(det, s_t, goal, H=8, N=96):
    # 1. Sample N action sequences from the fibre at s_t, H steps
    #    (independent samples at each step — the manifold doesn't need
    #     consistency across the horizon for a short one)
    seqs = [sample_action_sequence(det, s_t, H) for _ in range(N)]

    # 2. Roll out each sequence on the manifold (forward_query per step)
    total_rewards = []
    for seq in seqs:
        r = 0; s = s_t
        for a in seq:
            s_next = forward_query(det, s, a)
            r += R(s, a, s_next, goal)
            s = s_next
        total_rewards.append(r)

    # 3. Rank, execute first action of best sequence
    best_seq = seqs[argmax(total_rewards)]
    return best_seq[0]
```

**Cost per control step:** `N × H` forward queries = 768 VAE passes. At ~2 ms per pass, ~1.5 s on CPU. **Too slow for 50 Hz.** Two mitigations:
- **Batch forward_query** across the `N` sequences — trivially parallel. Brings this to ~150 ms on CPU with a batched decoder.
- **Reduce `N` to 32, `H` to 4** — 128 forward queries → ~30 ms. Acceptable for `dt = 50 ms` control.

**Recommendation: implement B1 first, validate the hypothesis, then B2 for the harder tasks.** B1 passing is necessary for B2 to be worth building.

### 5.3 Integration with existing `stage3_reference_test.py`

Their existing `stage3_reference_test.py` already does "MW walk as reference trajectory, track with PD." **Plan B is the natural generalization:** instead of walking toward a *given* target `s_goal`, Plan B **chooses** the target by sampling admissible actions and ranking by reward. The `stage3` code becomes the fallback mode when the goal state is known and the "walk" is the plan.

**No rewrite of `stage3` required.** Add `b1_step` and `b2_plan` as new functions; keep `stage3` as regression baseline.

---

## 6. Controller integration

### 6.1 Control loop

```python
for t in range(episode_length):
    s_t = read_state(robot)
    goal = task_goal(t)                # e.g. target forward velocity

    if mode == "B1":
        a = b1_step(det, s_t, goal)
    elif mode == "B2":
        a = b2_plan(det, s_t, goal)

    send_torque(robot, a)
    # Optional safety: if det.risk_score(s_t) > threshold, fall back to
    # the stored fallback controller (classical_controller.py, lqr_balance.py)
```

### 6.2 Safety fallback — non-negotiable

The manifold can flag "I don't know this state" via `det.risk_score(s_t)`. When risk > a calibrated threshold:
- Fall back to `classical_controller.WalkGaitController` (already in repo)
- Log the event
- **Do not** let Plan B act in unfamiliar territory

This is the same "Boundary is part of the contribution" (MW paper §9) principle applied at runtime.

### 6.3 What "task reward" `R(s, a, goal)` looks like

Must be **dense** — Plan B cannot do sparse rewards because it has no value function to propagate.

For Walker2d forward locomotion:
```python
R(s, a, s_next, goal) = -||s_next.forward_vel - goal.v||²
                       - λ₁ · ||a||²              # control effort
                       - λ₂ · d_M(s, a, s_next)   # stay on manifold
                       - λ₃ · fall_penalty if risk > thresh
```

Defaults: `λ₁ = 0.01`, `λ₂ = 0.1`, `λ₃ = 100`. Tune `λ₂` — this is the knob that makes Plan B "manifold-first."

For sparse tasks (e.g. "reach position X"), decompose into a dense proxy: `R = -||end_effector - X||` per step. Do not use `R = +1 if reached else 0`.

---

## 7. Validation plan

### 7.1 Phase 1 — Sim validation (weeks 1–3)

**All of the following must pass before Phase 2.** Each test mirrors a pattern already in your codebase.

| Test | Method | Pass criterion | Mirrors |
|---|---|---|---|
| **V1** | Reconstruction floor | Held-out MW distance < 1.5× training baseline | `stage1_validator_test.py` |
| **V2** | Injected fault AUC | ROC-AUC > 0.90 on all four fault types | `stage1_validator_test.py` |
| **V3** | Organic near-fall AUC | ROC-AUC > 0.85 | `stage1_validator_test.py` |
| **V4** | Off-manifold degradation | B1 degrades ≤ 20% when current state is corrupted by ε = 0.05·scale | **new — critical** |
| **V5** | Dose-response on params | B1 performance vs robot DOF: 6 (Walker) → 17 (Humanoid) → **hold within 80% of Walker result up to 1000 effective dims** | **new — the gating test** |
| **V6** | Cross-seed stability | B1 held-out mean across 20 seeds, std < 15% | `stage7_lead_time_sweep.py` |
| **V7** | Cross-physics stability | B1 held-out mean under shifted friction/damping (Stage 4 conditions) — flag rate < 5% | `stage4_compare.py` |
| **V8** | Cross-solver stability | B1 under dt = 0.016, 0.032, RK4 — flag rate < 10% | `stage6_sim2sim.py` |
| **V9** | B1 vs B2 | B2 ≥ B1 by ≥ 15% on mean forward velocity at equal wall-clock | **new** |
| **V10** | B1 vs Plan A (CEM over policy) | B1 within 20% of the CEM baseline from `cem_policy_search.py` | **new — the "is Plan B worth it" test** |

**V10 is the decisive test.** If B1 matches CEM-over-policy, Plan B is real. If not, Plan B is a limited tool and Plan A is the answer.

### 7.2 Phase 2 — Sim-to-hardware bridge (weeks 4–6)

**Prerequisites:** V1–V10 green; physical robot available; E-stop in place; operator trained.

**Protocol (matching the battery blind prospective study):**

| Step | Action | Acceptance |
|---|---|---|
| H1 | Collect 200 real transitions under operator control (not Plan B) | Schema matches sim; MW distance distribution within 2× sim baseline |
| H2 | Blind prospective — Plan B **predicts** next state for 500 held-out real transitions | ≥ 85% coverage at 90% credible band; mean state RMSE within 1.5× sim |
| H3 | Plan B **executes** on hardware, PD shadow-mode (Plan B decides, PD never fires unless MW distance > threshold) | 100+ episodes, zero E-stop firings, ≥ 90% success on task |
| H4 | Hard task (stairs, unknown terrain) with safety fallback active | Fallback triggers < 10% of episodes; no falls |

**H2 is the "Sandia/HUST" analogue.** It is the honest test that cannot be gamed by the sim.

### 7.3 Phase 3 — Extensive hardware validation (weeks 7–12, "the 95-cells analogue")

**This is what the user asked for: extensive validation comparable to the battery paper.**

| Dimension | Scale |
|---|---|
| Robot units | 20 physical units (walkers, e-rickshaws, delivery bots — depends on target) |
| Sites | 3+ sites (indoor lab, outdoor paved, outdoor rough) |
| Conditions | 5+ friction levels, 3+ temperature bands, 3+ payload levels |
| Motions | 5+ (walk, turn, stand, recover, stairs) |
| Total episodes | ≥ 2000 |
| Blind prospective predictions | ≥ 500 |
| Failures analyzed | Every fall, every E-stop, every fallback trigger — categorized against a pre-registered taxonomy |

**Reports to produce:**
1. Blind prospective coverage report (target: ≥ 85% at 90% band)
2. Failure-mode breakdown (physically classified, matching battery's "which assumption the data rejected")
3. Cross-robot generalization (does a manifold trained on robot A work on robot B?)
4. Public validation report + reproducible code (matching battery's approach)

### 7.4 What "success" means (pre-registered)

Define before starting Phase 2 — this is the MW-paper discipline:

- **Minimum success**: Plan B matches or beats the PD/LQR baseline on 3 of 5 motions, with 0 falls, on 20 units, in the blind prospective test.
- **Strong success**: Plan B matches CEM-over-policy (V10 green), reduces compute vs Plan A by ≥ 5× at equal performance, and generalizes to robot B without retraining the state portion.
- **Failure**: Plan B is worse than CEM-over-policy on ≥ 3 of 5 motions, or the off-manifold degradation test (V4) fails, or hardware V4 fails on more than 20% of units.

**Publish the failure honestly if it happens.** That's the battery paper's discipline: it published that ranges are too tight (12–25% coverage vs 90% target). Plan B should publish its off-manifold failures with the same honesty.

---

## 8. The gating experiment — before ANY of the above

Before writing 500 lines of runtime code, run this one experiment. It decides whether Plan B is worth the effort.

**The experiment:**
1. Train `TransitionVAE` on 1M Walker2d transitions.
2. On a held-out set of 5000 real `(s, s')` pairs, use `inverse_query` to find `a*` for each.
3. Compare `a*` to the actual action `a` that produced each pair.
4. Metric: **mean action error `||a* - a||` and its std.**

**Decision:**
- If mean error < 0.1 × action range and std < 0.2 × action range → **Plan B is viable.** Proceed to B1.
- If mean error is larger but std is small → the manifold is systematically wrong in a recoverable way. Proceed cautiously.
- If std is large → the inverse is ill-posed. **Plan B as specified will not work.** Either fall back to Plan A, or move to a richer manifold (add history, add context).

**Cost:** 1–2 days on CPU. **Value:** decides the entire direction of a multi-month project.

---

## 9. Failure modes and mitigations

| Failure | Symptom | Mitigation |
|---|---|---|
| **Manifold too simple** | Inverse ill-posed (E-gate fails) | Add history window (s_t, s_{t-1}, a_{t-1}); rebuild as fibre over history |
| **Manifold too complex** | Doesn't train, reconstruction floor high | Reduce latent dims; increase data; check prior filter isn't too aggressive |
| **Contact discontinuities** | Manifold fragments; B1 oscillates | Condition manifold on contact mode (foot down / in air) — same trick as the earlier "contact-mode fibre" |
| **Off-manifold optimal action** | B1 consistently worse than CEM baselines | V10 fails. Fall back to Plan A. Report honestly. |
| **Sim-to-real drift** | V4 passes in sim, H1 fails | Domain randomization in data; ensure `Stage 4` conditions included in training |
| **Cost too high** | B2 misses the 50 Hz budget | Batched forward_query; reduce H, N; fall back to B1 |
| **Reward hacking** | B1 finds policy that "looks good" but doesn't move | Dense reward must include forward progress explicitly; test with the specific failure mode from their `race_demo.py` bug #2 (faceplant gains) |

**The `race_demo.py` experience is directly relevant.** They documented three reward-hacking bugs in CEM (standing still, faceplant, over-survival weight). Plan B will hit the same bugs. The spec must **pre-register a reward-hacking test**: "for a forward-locomotion goal, B1 must achieve > 70% of target forward velocity AND not fall in the first 3 seconds AND stay within the manifold."

---

## 10. Concrete deliverables

Ordered by priority. Each is a discrete unit an agent can implement and test.

| # | Deliverable | File | Test |
|---|---|---|---|
| 1 | `TransitionVAE` domain class | `mw_framework/domains/transition.py` | `test_robotics_domain_runs_one_epoch` extended |
| 2 | Transition data generator (MuJoCo, pybamm_like recipe) | `mw_to_protomotions/transition_datagen.py` | Unit test: 10 units × 10 steps runs, rows valid |
| 3 | Prior filter (joint limits, torque limits, kinematic reachability) | `mw_to_protomotions/transition_filters.py` | Unit tests per filter |
| 4 | Fibre bundle assembly + `inverse_query` | `mw_framework/query/inverse.py` | **E-gate test** (§8) |
| 5 | `forward_query` (batched) | `mw_framework/query/forward.py` | Speed test: >100 queries/s on CPU |
| 6 | `b1_step` (posterior sample) | `mw_to_protomotions/b1_runtime.py` | V4, V6, V7, V8, V10 |
| 7 | `b2_plan` (MPC on manifold) | `mw_to_protomotions/b2_mpc.py` | V9 |
| 8 | Reward-hacking test suite | `tests/test_reward_hacking.py` | The specific bugs from `race_demo.py` |
| 9 | Safety fallback integration | `mw_to_protomotions/safety.py` | Uses existing `classical_controller.py`, `lqr_balance.py` |
| 10 | Hardware adapter (ROS2 or unitree_sdk bridge) | `hardware/adapter.py` | H1 protocol |

**Items 1–6 are Phase 1. Item 7 is Phase 1B (if V9 motivates). Items 8–10 are Phase 2.**

---

## 11. Integration with existing codebase — explicit

| Existing | Reused how |
|---|---|
| `MWVAE`, `MWTrainer`, `MWDetector` | **Unchanged.** `TransitionVAE` subclasses `MWVAE`. `MWDetector` fits it. |
| `constraint`, `bound` | **Directly used** in `TransitionVAE.physics()` for prior injection |
| `ManifoldAlgebra` | **Directly used** for fibre assembly (§2.3) |
| `geodesic`, `LorentzianCone`, `path_support` | **Directly used** in B1/B2 runtime and the "unphysical recovery" fault test |
| `walker_domain.py`, `datagen.py` | **Reused** for Walker2d state, action conventions, and near-fall fault testing |
| `classical_controller.py`, `lqr_balance.py` | **Reused** as safety fallback |
| `stage3_reference_test.py` | **Regression baseline** — Plan B must not regress on it |
| `mujoco_experiments/03_mujoco_like_datagen.py` | **Data-gen recipe directly ported** to `transition_datagen.py` |
| `walker_domain.py::WalkerPostureVAE` | **Superseded** by `TransitionVAE` for Plan B, but kept for the state-only fallback and for V4 (off-manifold degradation) |
| `stretch_humanoid.py` | **Baseline** for V5 dose-response |

**No file is deleted. Plan B is additive.** If Plan B fails V10, `walker_domain.py` + `cem_policy_search.py` (Plan A) remain the production path.

---

## 12. The one-page summary for the agent

> **Build `TransitionVAE` over `(s, a, s')` from MuJoCo, using your existing `MWVAE` subclass pattern and the pybamm_like data recipe. Use `ManifoldAlgebra` to encode it as a fibre bundle (state base → action fibre → next-state outer fibre). Run the E-gate experiment (§8) first; if the inverse query is ill-posed, stop and report. If viable, implement B1 (`b1_step`, §5.1) and validate against V4, V10. If B1 passes V10, implement B2 (`b2_plan`, §5.2). Do not touch hardware until all ten V-tests are green and the pre-registered "success" criteria are written down. Phase 2 hardware validation follows the blind-prospective protocol (H1–H4) — 20 units, 5 motions, 2000 episodes, minimum 500 filed predictions. Publish the failure mode if it fails — the battery paper's honesty is the model.**

**The bet**: `M` over `(s, a, s')` is rich enough that sampling from its action-fibre + ranking by dense reward is a viable policy, without any learned controller. **The decision**: V10. If Plan B beats or matches Plan A on V10, it's the future. If not, Plan A stays.
