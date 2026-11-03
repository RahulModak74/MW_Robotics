# SPEC: MW-Validity + CEM/ES Search for ProtoMotions-Class Humanoid Control (CPU-First)

This is the architecture that closes the gap between a GPU-heavy framework like ProtoMotions and a CPU-constrained deployment reality. The core logic is the same as the earlier retrieve-and-refine spec, but the **evaluator is now a learned manifold** instead of a full physics rollout, and the **search target is now a motion-tracking policy** instead of a survival policy.

---

## 0. The problem this solves, stated precisely

ProtoMotions trains **one General Tracking Policy** on ~142K motions using **24 A100 GPUs**, and even a smaller AMASS run takes **12 hours on 4 A100s**. The deployment artifact is a **Unified ONNX model** that runs on CPU via `onnxruntime` with only PyTorch and NumPy — no simulator, no GPU required. ProtoMotions also validates policies in **MuJoCo CPU** before hardware transfer.

So the asymmetry is clear: **training is GPU-bound; deployment is already CPU-friendly.** The MW+CEM/ES strategy attacks the *training and adaptation* side, not the deployment side. The goal is to produce ProtoMotions-compatible tracking policies **without the 24-A100 prerequisite**, using a small population search scored by a physics-validity manifold.

---

## 1. Architecture overview

```
┌─────────────────────────────────────────────────────────────┐
│  MW VALIDITY LAYER (CPU)                                     │
│  Learned manifold of admissible robot states + motions       │
│  Score(θ) = tracking_error − λ · MW_distance(trajectory)     │
└─────────────────────────────────────────────────────────────┘
                              ↓ cheap score
┌─────────────────────────────────────────────────────────────┐
│  CEM/ES SEARCH LAYER (CPU, population-parallel)              │
│  Sample policy params → score → keep elites → refit → repeat │
│  Warm-started from a retrieved library entry                 │
└─────────────────────────────────────────────────────────────┘
                              ↓ best params
┌─────────────────────────────────────────────────────────────┐
│  POLICY LIBRARY (CPU storage)                                │
│  .pol files storing params + search_dist(μ,σ) + MW manifold  │
│  Retrieval by context (robot, motion clip, terrain)          │
└─────────────────────────────────────────────────────────────┘
                              ↓ export
┌─────────────────────────────────────────────────────────────┐
│  ProtoMotions-compatible artifact                            │
│  ONNX checkpoint or resolved_configs + params                │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. MW Validity Layer: the learned manifold

### 2.1 What the manifold represents

The MW manifold here is **not** a battery-ageing manifold and **not** the full physics of the robot. It is a **low-dimensional latent representation of admissible state-motion pairs** for a specific robot body.

**Priors for a humanoid tracking manifold:**

| Physical fact | Prior | Why |
|---|---|---|
| Joint angles have limits | Truncated Normal at mechanical stops | Hard constraints become soft priors |
| Torques cannot exceed actuator limits | Truncated Normal, `\|τ\| ≤ τ_max` | Prevents the search from proposing infeasible actions |
| Contact forces are non-negative | Half-Normal | Physically required |
| Base orientation is continuous | Von Mises on yaw, wrapped Normal on roll/pitch | Avoids angle-wrap artifacts |
| Motion references are smooth | Gaussian process prior on trajectory derivatives | Penalises jerky candidates |
| Tracking error should not grow unboundedly | Truncated Normal on error norm | Bounds the objective |

### 2.2 Training the manifold

Train a **Bayesian VAE** (Pyro/PyTorch) on **simulation rollouts or retargeted motion data**:

- **Encoder:** maps `(state, action, next_state)` tuples to a latent `z`.
- **Decoder:** reconstructs the state transition.
- **Metric:** `G(z) = Jᵀ W J`, where `J` is the decoder Jacobian and `W` encodes physical importance weights (joint limits, torque limits, contact feasibility).
- **Training data:** generate ~10⁵–10⁶ transitions from **MuJoCo CPU** using a simple PD controller or scripted motion playback. This is cheap; no GPU needed. The retargeted motion data from ProtoMotions' PyRoki pipeline can also be used directly.

### 2.3 Scoring a candidate policy

Given a candidate parameter vector `θ`, the MW score is:

```
J_MW(θ) = − [ w_track · mean_tracking_error(θ, motion) 
            + λ · mean_MW_distance(τ_θ) ]
```

Where:
- `mean_tracking_error` is computed against the reference motion (joint-position error, root-position error).
- `mean_MW_distance` is the average Riemannian distance of the rollout trajectory from the learned manifold.
- `λ` is a Lagrange-style weight; start at `λ = 0.1` and tune.

**The critical property:** this score does **not** require a full physics rollout per candidate. The `MW_distance` term is a forward pass through the VAE encoder plus a distance computation. The tracking error can be approximated from a **short rollout** (e.g., 0.5–1 s) rather than the full motion length. This is what makes the search cheap.

### 2.4 The forward-only time constraint (optional but powerful)

If the manifold includes the **forward-only time axis with a causal cone** from the MW framework, then states that are temporally unreachable (e.g., a joint jumping from 0° to 90° in one millisecond) are automatically flagged as inadmissible. This rules out physically impossible candidates **before** they are evaluated in simulation, reducing wasted rollouts.

---

## 3. CEM/ES Search Layer: the policy search

### 3.1 Policy parameterisation

To keep the search dimension low, use a **linear or low-rank policy** on top of a **frozen, pre-trained encoder**:

```
a_t = W · φ(s_t) + b
```

Where:
- `φ(s_t)` is a fixed feature map (e.g., the first few layers of a pre-trained ProtoMotions encoder, frozen).
- `W, b` are the searched parameters. For a humanoid with ~20 joints and a 64-dim feature, this is **~1,300 parameters** — borderline but feasible with a large population. For a smaller robot or a lower-rank `W` (rank 4), it drops to **~80 parameters**.

**Rule P1:** Never search raw pixel-to-torque policies. Always search a **low-dimensional head** on a frozen representation. This is exactly the PI-ARS pattern: gradient-trained encoder + ES policy head.

### 3.2 Search loop with MW scoring

```python
# Warm start from library
mu, sigma = retrieve(context)  # or sigma_default if no library entry

for generation in range(max_generations):
    # Sample population
    pop = [mu + sigma * randn(D) for _ in range(n)]
    
    # Score each candidate — NOT full rollouts
    scores = []
    for theta in pop:
        # Short rollout (e.g., 0.5 s) in MuJoCo CPU
        tau = short_rollout(theta, motion, horizon=0.5s)
        tracking_err = compute_tracking_error(tau, reference)
        mw_dist = compute_mw_distance(tau)
        scores.append(-(w_track * tracking_err + lambda_ * mw_dist))
    
    # CEM update
    elites = top_k(pop, scores, k)
    mu = mean(elites)
    sigma = std(elites) with smoothing and floor
    
    # Early stopping
    if sigma < sigma_floor or stall > patience:
        break

# Full evaluation of winner only
final_score = full_rollout_and_evaluate(mu, motion, metric="MPJPE")
```

**Key efficiency gains over vanilla CEM:**
1. **Short rollouts** instead of full motion-length rollouts.
2. **MW distance** as a cheap physics penalty that catches bad candidates early.
3. **Warm start** from the library, which means the search starts near a known-good region.

### 3.3 Population size guidance

Based on the dimension wall:

| Parameter count | Population size `n` | Seeds `R` | Rollout budget per generation |
|---|---|---|---|
| ~80 (rank-4 head) | 32 | 3 | ~96 short rollouts |
| ~300 (rank-8 head) | 64 | 3 | ~192 short rollouts |
| ~1,300 (full linear head) | 128 | 5 | ~640 short rollouts |

If the full linear head is too slow, **reduce rank** rather than reduce population — the rank reduction is the dimension-control lever.

---

## 4. Policy Library: the reusable asset

### 4.1 What is stored

Each library entry stores:

| Artifact | Purpose |
|---|---|
| `params` (W, b) | The policy itself |
| `search_dist` (μ, σ) | Warm-start recipe for refinement |
| `mw_manifold` checkpoint | The validity model used during search |
| `eval_metrics` (MPJPE, root error, fall rate) | Honest performance record |
| `context` (robot name, motion category, terrain class) | Retrieval key |
| `provenance` (training sim commit, MuJoCo version) | Reproducibility |

### 4.2 Retrieval context

For ProtoMotions use cases, the context vector includes:
- **Robot model** (e.g., `h1_2`, `g1`, `smpl`).
- **Motion category** (walk, run, dance, recover, gesture).
- **Terrain class** (flat, stairs, slope).
- **Control rate** (e.g., 50 Hz).

The nearest entry is retrieved by Mahalanobis distance in context space, exactly as in the earlier spec.

---

## 5. Runtime scheduler: CPU-tier vs GPU-tier

| Tier | Hardware | What runs | Budget |
|---|---|---|---|
| **T0** | CPU only, no sim | Retrieval only, execute stored params | Milliseconds |
| **T1** | CPU + MuJoCo | Retrieval + tiny CEM (`n=16`, short rollouts) | Seconds to minutes |
| **T2** | CPU + MuJoCo, more cores | Retrieval + full CEM (`n=64`, short rollouts) | Minutes |
| **T3** | GPU available (optional) | Full ProtoMotions training as a **teacher**, distilled to a library entry | Hours |

**The Global South deployment reality is T0 or T1.** T2 is for a small operator with a few CPU cores and time to spare. T3 is for the research lab that produces library entries **once**, which are then shipped to the field as small files.

---

## 6. How this maps onto ProtoMotions use cases specifically

ProtoMotions' core use cases are: **motion imitation** (DeepMimic-style), **masked motion inpainting** (MaskedMimic), **general tracking** (one policy for 142K motions), and **sim-to-real on Unitree G1**.

| ProtoMotions use case | GPU approach | MW+CEM/ES approach | Honest verdict |
|---|---|---|---|
| **Single motion imitation** | PPO on one clip, 4 A100s, hours | CEM on a low-rank head, MW-scored short rollouts, CPU, minutes | **Strong fit.** Small dimension, single motion. |
| **Masked motion inpainting** | Distillation from complex tracking policy | CEM on a goal-conditioned low-rank head | **Moderate fit.** The goal-conditioning adds dimension; keep the head small. |
| **General tracking (142K motions)** | 24 A100s, full PPO, large policy | Cannot match directly. Instead: **library of motion-category-specific small policies**, retrieved at runtime | **Architectural fit, not direct match.** You ship a library, not one giant policy. |
| **Sim-to-real transfer** | Train in sim, export ONNX, validate in MuJoCo CPU | Train in MuJoCo CPU with MW scoring, export ONNX, validate in MuJoCo CPU | **Strong fit at the deployment end.** The ONNX export path is already CPU-native. |
| **Retargeting** | PyRoki optimizer, GPU | CEM as the retargeting optimizer (already used in the literature) | **Direct fit.** CEM is a natural retargeting solver. |

**The honest trade:** you will not beat 24 A100s on **one monolithic general tracker**. You **can** build a **library of small, motion-category-specific policies** that covers common motions and retrieves at runtime. This is the "catalogue + refine" model applied to motion imitation.

---

## 7. Evaluation and acceptance tests

### 7.1 What to measure

ProtoMotions reports **MPJPE** (mean per-joint position error), **root error**, and **fall rate** against reference motions. The MW+CEM/ES pipeline must report the same metrics, plus:

- **MW distance** of the final trajectory (is the policy physically plausible?).
- **Rollout budget** used per motion.
- **Wall-clock** on the target CPU.

### 7.2 Acceptance gates

| Test | Pass criterion |
|---|---|
| **A1: Single-motion imitation** | MPJPE within 2× of a PPO baseline trained on the same clip (if available), at ≤ 10⁵ short rollouts. |
| **A2: No-fall guarantee** | Fall rate ≤ 5% on held-out seeds, using the MW distance as an early-abort signal. |
| **A3: Library retrieval** | Nearest neighbor from a 50-entry library achieves ≥ 70% of the refined policy's MPJPE without refinement. |
| **A4: Wall-clock** | A single-motion refinement completes in ≤ 30 minutes on 4 CPU cores (T1/T2 tier). |
| **A5: ONNX compatibility** | The refined params export to a ProtoMotions-compatible ONNX or `resolved_configs` format. |

### 7.3 The gating experiment before building the full stack

Run **one** motion category (e.g., "walk") on **one** robot (e.g., G1 or H1-2) with:

- MW manifold trained on 10⁵ MuJoCo transitions.
- CEM on a rank-4 head (~80 params).
- Population 32, 3 seeds, short rollouts of 0.5 s.

**If MPJPE falls below a reasonable threshold (e.g., < 50 mm) within 10⁴ short rollouts, the architecture is viable.** If not, increase rank and repeat. This is the dose-response curve that decides how far the approach scales.

---

## 8. What is genuinely new here, and what is borrowed

**Borrowed and proven:**
- ProtoMotions' **ONNX export + MuJoCo CPU validation** pipeline is already CPU-native.
- CEM is already used for **retargeting and motion planning**.
- ES is known to scale with CPU cores (11 hours on 18 cores, 10 minutes on 1,440 cores for 3D humanoid).

**New and unproven:**
- Using an **MW-style learned manifold as the cheap score inside CEM/ES** for humanoid motion imitation. This is the experiment. It is not yet validated for this domain.
- The claim that a **low-rank head + MW scoring** can produce ProtoMotions-quality tracking policies on CPU. This is the hypothesis to test.

**Honest boundary:** this spec does not claim to replace ProtoMotions' large-scale training. It claims to build a **CPU-first adaptation and library pipeline** that covers common motions, retrieves at runtime, and refines locally — which is the realistic deployment model for operators without a 24-A100 cluster.

---

## 9. The one-paragraph summary for the CxO

ProtoMotions proves that GPU-scale training can produce a single general tracking policy; it does not prove that every operator needs that training pipeline. The MW+CEM/ES architecture **flips the economics**: use a learned physics-validity manifold to make small-policy search cheap, run it on CPU, store the results as a library of motion-category-specific policies, and retrieve-and-refine at deployment. The GPU tier still exists — but it is a **one-time library factory**, not a prerequisite for every operator. The gating experiment (single motion, rank-4 head, MuJoCo CPU, MW-scored short rollouts) tells you in days whether this scales to ProtoMotions-class tasks or tops out at simpler motions. Run that experiment before committing to the full stack.
