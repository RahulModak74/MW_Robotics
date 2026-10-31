# Path to full robotics — from here to ProtoMotions-scale demos

Where this repo stands today, and the concrete stages between here and demos like
[NVlabs/ProtoMotions](https://nvlabs.github.io/ProtoMotions/tutorials/) — a full humanoid
tracking motion-capture data, retargeted to a real robot, deployed and running. Every stage
below extends a mechanism that's already proven, not a leap into the unknown. Chance of
success is rated High / Med / Med-with-effort for each stage — the floor here is
"Med-with-effort," because every stage builds on something that already works at least once;
none of this starts from zero.

## Where we are now — the proven foundation

- **Six separate physics systems, each validated as an anomaly checker**, trained on exact
  closed-form physics, running on CPU, all using the same unmodified core method. See
  `ROBOTICS.md`.
- **One of them (`legged_contact`) proven as a forecaster against a real physics engine**
  (MuJoCo) — predicts a future state faster than re-simulating, with honest, measured accuracy
  (roughly 7-14% of normal range, depending on jump size). See `FORECASTER.md`.
- **Chaining proven stable**: forecasting several steps ahead, each one building on the last
  guess instead of a real measurement, does not compound error — a real, tested property, not
  an assumption.
- **The core forecasting mechanism — encode a state, encode a target, walk the straight line
  between them, decode — independently proven in production, in a completely different
  domain** (battery degradation), with no RL and no GPU. Two domains, same mechanism, same
  result. That's the strongest single piece of evidence in this whole plan.
- **A real, no-training-needed path to closed-loop control already identified**: every domain
  in this repo has its control input already written into its known equation (torque, speed,
  applied force), so classical control — not RL — already gives real-time "act and react"
  behavior for every system this repo currently models.

## What "ProtoMotions-scale" actually requires

Breaking their tutorial list into the real underlying capabilities:

| Capability | ProtoMotions' version |
|---|---|
| A single, coupled, contact-rich, many-joint body | full humanoid/robot dynamics, not six separate pieces |
| Real-time closed-loop control | a policy that reacts to the actual current state every step |
| Learning from real motion-capture data | training on AMASS and similar human-motion datasets |
| Retargeting to a specific real robot | mapping human motion onto a robot's own joints (G1, H1) |
| Sim2sim and sim2real transfer | the same behavior holding up across simulators, and on real hardware |
| A broad skill library | walking, running, reaching, recovering from a push, and more, in one system |
| Real hardware deployment | running on an actual Unitree G1, safely |

## The stages

### Stage 1 — One coupled body, not six separate ones
Extend the validator mechanism already proven six times to a single body with many joints
moving together (start with a simple biped or quadruped, not a full 20+-joint humanoid —
prove the recipe cheaply first, the same choice already made for the walker experiment).
Physics rules become softer bounds instead of exact equations (joint limits, a balance
margin, non-penetration) because full coupled dynamics don't reduce to three clean laws the
way a single beam or joint does.
**Chance of success: High.** This is the same mechanism that already generalized across a
PDE, two different ODEs, pure geometry, and an impact problem — adding more joints is scaling
a proven pattern, not testing a new one.

### Stage 2 — Real-time closed-loop control, classically
A whole-body balance controller (PD servos plus a center-of-mass/support-polygon balance
check) — mature, well-understood classical robotics, no RL, no training time at all. This is
what actually makes the robot react to a push or a stumble.
**Chance of success: High.** This technique predates and doesn't depend on MW working at all
— it's standard control engineering, and the risk is tuning time, not feasibility.

### Stage 3 — MW's manifold walk as the reference trajectory for that controller
Feed MW's proven geodesic-walk path — the same mechanism already validated for the robot leg
and for battery degradation — into the Stage 2 controller as the path to track, instead of a
naive straight-line joint interpolation. Measure whether the real, controlled motion is
smoother and holds balance better, the same rigorous head-to-head comparison already used
throughout this repo.
**Chance of success: Med.** The mechanism itself is proven twice over; what's new here is
whether it holds its advantage once a real balance constraint is in the loop, which hasn't
been tested yet on a coupled body.

### Stage 4 — Learn from real human motion data, not just physics-generated data
Train MW's manifold on real motion-capture data (AMASS-style) instead of exact closed-form or
simulated trajectories, for the first time. This is the step that would make the skill
library as broad and natural-looking as ProtoMotions' own (walking, running, varied motion),
not just the handful of behaviors a hand-written physics law can generate.
**Chance of success: Med-with-effort.** The underlying method (physics-constrained Bayesian
manifold learning) is proven on clean, physics-generated data in every domain so far; whether
it holds up equally well on much more diverse, noisy real human-motion data is a genuinely
open question worth testing directly, with real tuning likely needed along the way.

### Stage 5 — Retarget to a real robot's own joints
Map the learned motion onto a specific robot's morphology (e.g. Unitree G1). This is mostly
integration work, not new research — off-the-shelf open tools (e.g. PyRoki-style retargeting)
already solve this problem well.
**Chance of success: High.** Solved territory; the work here is wiring, not invention.

### Stage 6 — Sim2sim validation
Test whether a behavior tuned in one simulator holds up when checked in a different one — the
exact discipline this repo already practices (`FORECASTER.md`'s bug-hunting process was built
around exactly this kind of cross-checking). Applying that same discipline to a much
higher-dimensional coupled system is the honest next step, not a new skill to learn.
**Chance of success: Med.** The *method* for doing this rigorously is already proven here;
whether the physics itself transfers cleanly at this scale is the open, testable part.

### Stage 7 — Sim2real and a full skill library on real hardware
Running on an actual robot, safely, with a skill library as broad as ProtoMotions' own. The
biggest stage by far — real physical safety stakes, needing a validated safety/clamping layer
and extensive testing before anything touches real hardware, and depending on Stage 4's open
question already having been answered well.
**Chance of success: Med-with-effort.** The most new engineering and the most caution of any
stage, and the one most worth taking seriously rather than rushing — but every piece it leans
on (Stages 1-6) is itself either already proven or already rated at least Med above. This is
a hard stage, not a closed door.

## Reading this plan

Every stage either reuses a mechanism already proven at least once in this repo or in a
sibling project, or is a well-understood classical technique that doesn't depend on MW working
at all. Nothing here is a bet on an unproven idea from a standing start — it's a sequence of
specific, testable extensions of things that already work, in the order that lets each one be
checked honestly before the next is attempted.
