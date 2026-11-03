markdown
# Walker2d Demo + CEM/MW Pipeline: Agent Guide

**Purpose.** This document explains the current Walker2d demo and training
files, their role in a CPU-first, gradient-free policy search pipeline,
how they compare to NVIDIA ProtoMotions’ architecture, the concrete gaps,
and a prioritized roadmap for an agent to close those gaps.

---

## 1. System Overview

The current system has two independent training paths and one interactive
demo:
Path A (balance policy) Path B (walk policy)
run_plan_a.sh run_demo_walker.sh
├── cem_policy_search.py ├── forward-progress fitness
├── survival fitness └── cem_walk_theta.npy
└── cem_best_theta.npy
│ │
└──────────────┬───────────────────────┘
▼
run_walker_demo.sh
(Flask server on :5051)
└── walker_app.py
├── loads both .npy files
├── loads walker_mw_detector.ckpt
└── serves real MuJoCo rollouts as JSON frames

text

**Core philosophy.** Replace GPU PPO over large neural networks with CEM
(Cross-Entropy Method) over a small, linear policy, running entirely on
CPU. The MW manifold is an independent validator: it does not generate
the policy, it flags when the policy or the state is physically
implausible.

---

## 2. File-by-File Explanation

### 2.1 `domain/run_plan_a.sh`

**What it does.** Runs CEM over a 96-parameter linear policy on Walker2d,
using **survival steps** as the fitness signal. It saves the best
parameter vector to `cem_best_theta.npy` and a summary to
`cem_summary.json`. It includes held-out evaluation and a random baseline.

**Key design choices.**

- **Parameterisation.** `a = tanh(W @ φ(s) + b)`, where `W` is 6×15 and
  `b` is 6. Total 96 parameters. This is deliberately small; it is the
  only reason CEM is viable at all.
- **Fitness.** `mean survival steps` across seeds. This is the
  reward-hacking vulnerability: standing still maximises survival.
- **Search.** Population 60–80, 15–30 generations, elite fraction 0.2.
- **Output.** `cem_best_theta.npy`, a float32 array of length 96.

**Known limitation.** The 920.9-step result from `--generations 30 --pop
80` is a **stand-still policy**. It does not walk. This is not a bug in
CEM; it is the correct optimum for the fitness function. The policy has
learned that falling is bad and moving is unnecessary.

**Improvement target.** Replace survival fitness with forward-progress
fitness (see `run_demo_walker.sh`, which already does this). Also consider
adding a **gait-phase clock** (sin/cos of a fixed-frequency phase
variable) to the feature vector, as this was the single most effective
fix for forward locomotion in the `race_demo.py` experiments.

---

### 2.2 `domain/run_demo_walker.sh`

**What it does.** Trains a **walking** policy using forward-progress
fitness. Saves `cem_walk_theta.npy`. This is the policy that the “walk”
button in the Flask demo loads.

**Fitness function (the critical line).**

```python
return (x_final - x0) + 0.02 * steps
This is the forward distance + small survival bonus formulation. The
0.02 weight is deliberately small: it makes falling costly (because a
fallen episode stops accumulating steps) without allowing survival alone
to dominate.

Why this matters. In the race_demo.py experiments, pure forward
distance led to a faceplant shortcut (fall forward, gain distance). Pure
survival led to standing still. The 0.02-weighted combination is the
smallest change that avoids both failure modes.

Current status. This script exists but has not yet been run in the
current session. The expected output is cem_walk_theta.npy and
cem_walk_summary.json at the repository root.

2.3 domain/run_walker_demo.sh
What it does. Starts a Flask server on port 5051. Loads
cem_best_theta.npy (balance) and cem_walk_theta.npy (walk), plus the
MW detector from mw_to_protomotions/walker_mw_detector.ckpt. Exposes
/api/walker/run and /api/walker/policies.

Dependency. Requires both .npy files to exist. If
cem_walk_theta.npy is missing, the walk button returns a 404 and the
server still runs.

Known bug (fixed in the current version). The repository-root
detection previously looked for ../cem_policy_search.py instead of
../mw_to_protomotions/cem_policy_search.py. The correct pattern (same
as run_plan_a.sh) is now in use.

2.4 domain/walker_demo/walker_animation.py
What it does. Runs a real MuJoCo rollout driven by a loaded CEM
policy. Returns:

qpos and qvel per step (the raw physics state, not an
interpolation).

mw_distances per step, computed by the MW detector on the
posture features.

Summary fields: forward_distance, fell_at, seed, push.

Critical design decision. Unlike the other six MW domains (beam,
joint, arm, mobile, thermal), which are closed-form physics and
therefore animate by latent-space interpolation, Walker2d has a real
forward simulator. The animation is therefore the actual physics, and
the manifold is a validator, not a generator. This is the honest
scope and should remain so.

2.5 domain/walker_demo/walker_app.py
What it does. The Flask backend. Loads policies and the MW detector
at startup. Defines DOMAIN_EQ (the human-readable physics laws for the
frontend) and two routes.

Schema. The response is:

json
{
  "qpos": [[...], ...],
  "qvel": [[...], ...],
  "mw_distances": [...],
  "steps": 600,
  "forward_distance": 2.31,
  "fell_at": null,
  "policy": "walk",
  "domain": "walker2d",
  "equations": ["joint limits: ...", "reachability: ...", "MW distance: ..."]
}
This mirrors the {frames, domain, equations} shape of the existing
full_flow_animation/app.py, but with raw qpos/qvel instead of
latent-space features.

2.6 domain/walker_demo/static/index.html
What it does. A single-page frontend. Renders a stick figure from
qpos (canvas 2D, no Three.js), plots MW distance over the last 300 steps
on a separate gauge, and shows live stats (step, forward distance, MW
distance, status).

The MW gauge is the key visual. It shows the manifold’s distance
rising before the physics engine terminates. This is the “safety
monitor” story: the manifold sees unphysical states before they become
failures.

2.7 mw_to_protomotions/walker_domain.py (supporting file)
What it does. Defines WalkerPostureVAE, an MW domain over the nine
posture features (height, pitch, pitch_rate, six joint angles). Its
physics() applies:

Exact joint limits from the MuJoCo XML.

A soft pitch band and height band.

This is the manifold that scores the walker’s states. It is not a
policy; it is the validator used by the demo’s MW gauge and by
stage7_robustness.py’s early-warning experiments.

2.8 mw_to_protomotions/cem_policy_search.py (supporting file)
What it does. The original CEM implementation. Defines get_features,
policy_action, rollout, fitness, and cem. The 96-parameter
linear policy and the survival-based fitness live here.

Reuse. Both run_plan_a.sh and run_demo_walker.sh import from this
file. The forward-progress script overrides fitness but reuses
get_features and policy_action unchanged.

3. The End Goal: ProtoMotions Parity
ProtoMotions is a modular framework for humanoid motion tracking. Its
core components are:

Component	ProtoMotions	This pipeline
Policy learning	PPO / AMP over large neural networks	CEM over 96-param linear policy
Simulation	IsaacGym / Newton (GPU-parallel)	MuJoCo CPU (serial or mjbatch)
Motion data	BONES-SEED, ~142K retargeted G1 motions	Walker2d, small held-out set
Retargeting	PyRoki (GPU optimization)	Not yet implemented
Deployment	ONNX Runtime, CPU	Not yet implemented
Policy size	Millions of parameters	96 parameters
The honest position. The current pipeline is a CPU-first
alternative to the learning component, not a full replacement. The
other components (retargeting, large-scale simulation, deployment) are
either not yet present or are handled by simple MuJoCo calls.

What “parity” means in practice. For the purposes of this roadmap,
parity means: a CPU-only pipeline that can retarget a motion, train a
tracking policy with CEM, deploy it via ONNX, and execute it at
control rate, using the same structural decomposition as ProtoMotions
but with CEM instead of PPO.

4. Concrete Gaps and Improvement Roadmap
Gap 1: No motion retargeting
Problem. ProtoMotions retargets human motion data to the target
robot. The current Walker2d pipeline uses the environment’s built-in
dynamics and a hand-tuned reference (or no reference at all, in the
balance case). For any real humanoid, retargeting is mandatory.

Solution. Integrate GMR (General Motion Retargeting), which runs
in real time on CPU and is already used by the TWIST teleoperation
stack. The HUST-3W fork adds automatic IK-config generation
for new humanoid models.

Agent action.

Add GMR as a dependency.

Write domain/retarget.py that takes a human motion clip (e.g.
AMASS) and a robot XML, returns per-timestep joint targets in the
same schema as qpos[3:9].

Replace the hand-authored reference in run_demo_walker.sh with the
GMR output.

Gap 2: CPU simulation is serial, not parallel
Problem. run_plan_a.sh and run_demo_walker.sh evaluate the CEM
population serially. On a 16-core CPU, this wastes ~15× the available
compute. CEM is embarrassingly parallel: every candidate is independent.

Solution. Use mjbatch, a C++ thread-pool library that runs
thousands of MuJoCo simulations in parallel on CPU with the GIL
released. It exposes batched qpos, qvel, and ctrl
views, so a population of 1024 candidates can be stepped simultaneously.

Agent action.

Add mjbatch as a dependency.

Replace the fitness() loop in run_demo_walker.sh with a batched
call: expand the model to batch_size = POP, step all candidates
for MAX_STEPS, read batched qpos to compute forward distance.

Expected wall-clock improvement: 10–15× on a 16-core machine.
A 25-minute training run becomes ~2 minutes.

Gap 3: Policy is a linear map, not a tracking policy
Problem. The 96-parameter policy maps raw state to torques. It has
no notion of a reference trajectory. ProtoMotions’ policy tracks a
target motion; the current policy just survives (or walks forward
blindly).

Solution. Introduce a reference-tracking fitness. The fitness
should be -mean_joint_error(policy_rollout, retargeted_reference),
optionally plus a small forward-progress term.

This is the exact pattern used in ProtoMotions’ tracking setup: the
policy is rewarded for matching joint targets, not for a scalar survival
bonus.

Agent action.

Implement tracking_fitness(theta, reference, seeds) that:

Rolls out the policy.

Computes mean squared joint error against the reference at each
step.

Returns -(mean_error) + 0.01 * forward_distance.

Add a --reference flag to run_demo_walker.sh that accepts a .npz
with shape (T, 6).

If no reference is given, fall back to forward-progress fitness.

Gap 4: No low-rank or waypoint lifting for high-dimensional actions
Problem. The 96-parameter policy is viable on Walker2d (6 actions).
For a 29-DoF humanoid, a full linear policy is 29×F + 29 parameters,
which is well beyond CEM’s dimensional comfort zone.

Solution. Two complementary techniques:

Rank-4 head. a = U @ (V @ φ(s)) + b, with V shape (4, F)
and U shape (29, 4). This reduces the parameter count to
4F + 116 + 29. For F=30, that is 265 parameters, not 900.

Waypoint lifting. Instead of per-step torques, search 8–12
waypoint joint targets over the horizon. A fixed PD controller
tracks the waypoints. The search dimension becomes
12 × 29 + 12 = 360 parameters, which is manageable.

Agent action.

Add domain/low_rank_policy.py with RankKPolicy.

Add domain/waypoint_policy.py with WaypointPolicy (PD tracking
of lifted targets).

Add a --policy-type flag to run_demo_walker.sh: linear (current),
rank4, waypoint.

Benchmark all three on Walker2d, then port the best to a simple
humanoid.

Gap 5: No ONNX deployment path
Problem. The demo runs the policy in Python. ProtoMotions exports to
ONNX and runs on the robot’s CPU at sub-millisecond inference.
Without an ONNX path, the pipeline cannot be deployed.

Solution. Export the trained policy (linear, rank-k, or waypoint
controller) to ONNX. Use ONNX Runtime for inference. A 96-parameter
linear policy exports trivially; a rank-4 head or a waypoint policy is
also ONNX-compatible.

Agent action.

Add domain/export_onnx.py that takes theta and a policy-type
string, builds the corresponding torch.nn.Module, and exports to
.onnx.

Add domain/onnx_runtime.py that loads the .onnx and runs a
control loop.

Validate latency: target < 2 ms per inference step on CPU.

Gap 6: Manifold is a validator, not a reward-shaping term
Problem. The MW distance is computed and displayed, but it does not
influence training. ProtoMotions’ policies are shaped by the
environment’s reward function; the MW manifold is currently
observational only.

Solution. Add MW distance to the fitness function as a penalty:

python
score = forward_distance - λ_mw * mean_mw_distance
This was the intended design in the Plan B spec, but the Plan B
architecture (joint VAE over (s, a, s')) failed its E-gate. The
validator role, however, remains sound. Using the WalkerPostureVAE as
a shaping term is a lower-risk way to incorporate the manifold.

Agent action.

In run_demo_walker.sh, after each rollout, compute
det.mw_distance(posture_df) and subtract λ_mw * mean_distance
from the fitness.

Tune λ_mw from 0.01 to 0.1.

Measure whether MW-shaping improves the final forward distance
relative to the unshaped baseline.

5. Prioritized Roadmap
Priority	Gap	Effort	Expected impact
P0	Gap 2: mjbatch parallel simulation	1–2 days	10–15× training speedup
P0	Gap 3: tracking fitness	2–3 days	Makes policies track references, not just move
P1	Gap 1: GMR retargeting	2–3 days	Required for any humanoid work
P1	Gap 4: rank-4 / waypoint lifting	3–5 days	Enables 29-DoF humanoid search
P2	Gap 5: ONNX export	1–2 days	Deployment path
P2	Gap 6: MW reward shaping	1–2 days	Physically grounded policies
First sprint (2 weeks). P0 items. After this sprint, the pipeline
should train a reference-tracking Walker2d policy in under 3 minutes
on 16 CPU cores, and the demo should show it following a retargeted
gait. This is a concrete, defensible milestone that establishes the
CPU-first learning mechanism before moving to a humanoid.

6. What an Agent Should Not Do
Do not attempt to train a 29-DoF humanoid with the current linear
policy. The dimensionality wall is real. Rank-4 or waypoint lifting
is a prerequisite, not an optional optimisation.

Do not use the Plan B joint VAE over (s, a, s') for policy
learning. It failed its E-gate because the joint form has no
incentive to use the action. If a manifold-based policy is needed
later, use a conditional VAE (p(s' | s, a)).

Do not replace the real MuJoCo rollout in the demo with latent-space
interpolation. The demo’s honesty depends on showing real physics
with the manifold as a validator, not a manifold-generated
trajectory that cannot fail.

7. References and Source Files
domain/run_plan_a.sh — balance policy training (survival fitness).

domain/run_demo_walker.sh — walk policy training (forward-progress
fitness).

domain/run_walker_demo.sh — Flask server.

domain/walker_demo/walker_animation.py — real MuJoCo rollout +
MW distance.

domain/walker_demo/walker_app.py — Flask routes.

domain/walker_demo/static/index.html — frontend.

mw_to_protomotions/cem_policy_search.py — CEM implementation.

mw_to_protomotions/walker_domain.py — WalkerPostureVAE validator.

mw_framework/ — MW core (detector, geometry, VAE training).
