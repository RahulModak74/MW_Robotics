# MW Robotics — Roadmap

*A pragmatic two-step path to ProtoMotions-class humanoid control, built on the Modak-Walawalkar (MW) framework.*

## Summary

The goal is a humanoid that walks, runs, balances, recovers from pushes and, in time, retargets human motion to real hardware
— the capability shown by NVIDIA's [ProtoMotions](https://nvlabs.github.io/ProtoMotions/tutorials/) — with a policy layer that
**trains on consumer hardware and deploys on a commodity CPU**.

ProtoMotions gets there with brute-force GPU reinforcement learning over millions of parallel simulations. We take two steps:

- **Step 1 — GPU teacher, CPU student** ([`for_distilation/`](for_distilation/)). A GPU-trained RL policy is used *only as a data
  generator*. Its high-quality `(state, action)` pairs are distilled into an MW Pyro VAE constrained by hard physics priors.
  Training uses a consumer GPU; **deployment is on a plain CPU.**
- **Step 2 — CPU-native policy learning** ([`for_full_replacement/`](for_full_replacement/)). Remove the teacher. The physics
  priors (joint limits, ZMP, symmetry, energy) become the inductive bias and the MW VAE is the primary learner via
  CPU-native stochastic variational inference — the MCMC → SVI analogy applied to policy learning. Goal: **a GPU optional for
  seed data, and CPU for all training and deployment.**

The approach is pragmatic, not dogmatic: use the GPU where it helps today, and keep moving the work onto the CPU.

## The foundation already in place

1. **Six physics domains** with validators — beam, motor joint, planar arm, wheeled robot, legged contact, actuator heat —
   all on CPU with one unmodified core method.
2. **Forecasting** the next state faster than re-simulating, measured against MuJoCo on the hopping leg.
3. **Stable chaining:** multi-step forecasts stay anchored to the physics and do not drift.
4. **Cross-domain use:** the same "encode target, walk the latent geodesic, decode" mechanism runs in production for battery
   degradation, with no RL and no GPU.
5. **Classical baselines:** working PD/LQR balance controllers, ready to act as the safety net.
6. **A strong 3D teacher:** PPO on Humanoid-v5 reaches ~578 held-out steps (8.3 m) after 3M steps, with more headroom from longer
   training.

## MW vs. a conventional GPU stack

| | ProtoMotions (NVIDIA) | MW Robotics (target) |
|---|---|---|
| Policy generation | GPU RL (PPO/SAC) over millions of environment steps | **Step 1:** GPU teacher → MW student. **Step 2:** direct CPU-native MW learning |
| Compute | High-end GPUs for training and inference | **Step 1:** GPU to train, CPU to run. **Step 2:** CPU for both |
| Safety | Black-box network | Physics-constrained latent space, with a per-step uncertainty signal |
| Data | Millions of environment steps | Thousands of high-quality `(s, a)` pairs (Step 1), sparse physics-guided exploration (Step 2) |
| Deployment | Vendor runtime (Omniverse / Isaac / Jetson) | Agnostic: edge CPUs, air-gapped systems, low-cost hardware |

## Stages

Every stage extends something already working in this repo. Chances of success are rated High / Med / Med-with-effort.

### Step 1 — GPU teacher → MW student *(status: teacher working; student under active development)*

**Stage 1 — GPU teacher baseline.** Train PPO on `Humanoid-v5` (shaped reward: alive bonus + forward velocity, entropy
regularization, learning-rate anneal) and collect ~300k `(s, a)` pairs, including action-noise rollouts for coverage.
*Success: High.* A 3M-step teacher is included; longer runs (5M–10M steps) push survival toward the 900+ range.
See [for_distilation/docs/NEW_APPROACH.md](for_distilation/docs/NEW_APPROACH.md).

**Stage 2 — MW distillation (the core).** Train the MW `PolicyManifold3DVAE` by Pyro SVI on the teacher data, with 3D physics
priors the teacher never saw: torso-height envelope, unit quaternion, joint and torque limits, then angular momentum and
left/right symmetry. *Success: High on the mechanism; in tuning for the full humanoid.*
The planned improvements are in [STUDENT_COLLAPSE_FIXES.md](STUDENT_COLLAPSE_FIXES.md).

**Stage 3 — DAgger self-improvement.** Roll out the student, find states where its uncertainty is high (MC-dropout), query the
teacher only there, retrain a few epochs, repeat 3–4 times. The student only needs to see the edge cases (a stumble, a push).
*Success: Med-with-effort.*

**Stage 4 — Real-time CPU inference with a classical fallback.** Run the VAE on plain PyTorch CPU or ONNX (target < 10 ms per
step for 50 Hz control). A runtime monitor hands over to the PD/LQR controller when MW uncertainty crosses a calibrated
threshold. *Success: High.*

### Step 2 — CPU-native policy learning *(status: in progress)*

**Stage 5 — Teacher-free MW policy learning.** Formulate policy learning as physics-constrained manifold optimization: the VAE
is optimized directly by Pyro SVI, with physics priors as the inductive bias that keeps exploration inside the valid action
space. This is the active research frontier; see "Step 2 design" below. *Success: Med-with-effort.*

### Beyond locomotion

**Stage 6 — Motion-capture integration.** Retarget AMASS-style data to the MuJoCo humanoid and train the VAE on it. The MW
priors filter out physically impossible mocap artifacts (torque-limit violations, etc.). *Success: Med.*

**Stage 7 — Sim2sim and sim2real.** Validate across physics engines and solver settings, then retarget to a specific robot
(e.g. Unitree G1/H1) and deploy with the classical fallback armed and no cloud connection. Because the policy is constrained
by universal physics rather than one engine's quirks, it should transfer more robustly than a raw PPO policy.
*Success: Med / Med-with-effort — the most engineering-heavy stage.*

## Planning estimates *(targets, not measurements)*

Where we aim to land relative to ProtoMotions' utility, by capability, if the plan goes well:

| Capability | Target | Main lever |
|---|---|---|
| Core locomotion (walk, run, balance, turn) | ~95% | Distillation + ZMP / symmetry / torque priors; smoother, lower-energy gaits |
| Push recovery and robustness | ~90% | Uncertainty signal + classical fallback beyond the training distribution |
| Zero-shot sim2real | 70–80% | Physics priors + fallback; real-world motor latency still needs empirical tuning |
| Broad mocap skills (dance, parkour) | 50–60%, up to ~80% | Hierarchical latent space (below) |

Overall: roughly 80–85% of ProtoMotions' utility, with a decisive advantage in deployability — a $500-class CPU, 5–15 W,
air-gapped operation and an auditable, physics-constrained policy.

## Design notes

### A. Where the GPU cost in conventional stacks comes from

The policy network is small; the cost is **sample count**. PPO is on-policy, so it discards each batch after one update and
needs billions of steps, which needs thousands of parallel simulators. Reference points from published setups: Isaac Lab PPO on
Unitree H1 rough terrain, ~147M environment steps in ~64 minutes on an L40S at 4096 environments; ProtoMotions' general motion
tracker on 24 A100s; ~72M timesteps to reach a prior-best running speed.

The MW policy attacks this on three axes:

1. **The ELBO carries physics.** Joint limits, ZMP, torque limits and symmetry are terms in the loss rather than behaviour the
   network must discover — hence thousands of `(s, a)` pairs rather than billions of steps.
2. **Priors prune the search space before training** — invalid torques and asymmetric gaits are excluded from the start.
3. **Uncertainty is an output.** The MW distance reports, per step, how far the current state is from the learned manifold,
   which is what makes the classical-fallback contract auditable.

On a consumer card the same PPO budget runs roughly 4–10× slower in wall-clock than on a data-center GPU, so "train a walking
humanoid overnight on your laptop" is realistic and "match a frontier lab's iteration speed" is not the aim.

### B. Step 2 design — how reward enters the ELBO

SVI optimizes `E_q[log p(x|z)] − KL(q‖p)`, a generative objective over data; a policy needs `E_π[R(τ)]`. The physics priors shape
the latent space, and the forward-progress signal needs its own route into training. Three mechanisms, in order of how
directly they use the MW machinery:

1. **Reparameterized rollouts through a learned differentiable dynamics model.** Train a CPU-friendly world model from Step 1
   data or cheap random rollouts and back-propagate reward through it; the VAE is the world model and the policy is amortized
   inference over it.
2. **Score-function / variational policy gradient.** Sample from the decoder with a baseline; priors prune the action space so
   fewer samples are needed.
3. **Amortized inference + self-improvement.** Fit `q(z|s)` on seed data, then run Stage 3 with uncertainty-gated queries.
   Likely the first version to ship.

External wording for Step 2: *"GPU optional for a seed corpus; CPU for all training and deployment."*

### C. The first Step 2 experiment

Before adding joints: a **planar hopper or 2-DoF biped, sparse forward reward, CPU-native SVI, no teacher, no reward shaping.**
It is small, fast, and decisive — it shows exactly where to push next. Results will be recorded in
[for_full_replacement/](for_full_replacement/).

### D. Hierarchical (fibre-bundle) latent space

For many skills, decompose the latent as `z_total = [z_balance (3D), z_gait_phase (2D), z_skill_intent (3D)]` using
`ManifoldAlgebra`, rather than one flat 8-D vector. Two points to build in from the start: per-fibre KL weights (so one fibre
does not absorb another's information) and balanced sampling across skills (so rare skills are not averaged away). Prove two
fibres (`z_balance` + `z_gait_phase`) on walk/run before adding `z_skill_intent`.

### E. Open-source plan

1. **License check of ProtoMotions first** (about an hour). Apache 2.0 allows a clean port with attribution; any
   field-of-use restriction changes the plan, and defense buyers will ask.
2. Step 1 working end-to-end on an RTX 4060-class card, with a reproducible script and a demo video.
3. Port `g1_raceengine` as a thin, documented distribution of upstream rather than a hard fork, with a named maintainer.
4. Publish three open artifacts: an MJX-native G1 race engine; a reproducible locomotion baseline (walk / run / push-recover)
   so MW policies can be compared with PPO on the same substrate; and an open **uncertainty / fallback interface**,
   `predict → (action, uncertainty, fallback_flag)`, that any policy can implement.

Kept in-house: the fibre-bundle hierarchical VAE, the physics-prior ELBO, the uncertainty/certification layer and the mocap
retargeting pipeline.

> *Reproduce state-of-the-art humanoid locomotion research and build G1 demos on a consumer laptop — pip-installable,
> MuJoCo-native, no Omniverse, no enterprise GPU — with a physics-constrained policy layer that also deploys to a $500 CPU.*
