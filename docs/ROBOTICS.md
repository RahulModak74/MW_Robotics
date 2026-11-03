# Robotics results — status and limits (2026-10-31)

This repo is `MW_Framework` plus six new robot examples, each a genuinely different kind of
physics: a bending beam (a PDE), a motor joint and a wheeled robot (two different ODEs), a
robot arm (pure geometry, no solving needed), a hopping leg's bounce (an impact problem), and
motor heating (a second PDE). Run anything from `examples/00_flexible_robot_validator.py`
through `examples/05_actuator_thermal_validator.py` to try them. [FIXING_THE_GAPS.md](FIXING_THE_GAPS.md)
has concrete next steps for every weak spot below.

MW has now been tried in two different jobs, both against a real physics engine (MuJoCo,
Apache-2.0), on the hopping-leg example: as a **checker** (does this state look physically
real?) and as a **predictor** (can it guess a future state without running a simulator?). See
[FORECASTER.md](FORECASTER.md) for the predictor results in full — three real bugs were found
and fixed to get trustworthy numbers, and the result is a genuine, measured tradeoff between
speed and accuracy, not an unconditional win.

## Two things to know before reading further

**1. The core method works — that's the solid, proven part.** The underlying code (`core.py`)
is unchanged from the original `MW_Framework` repo, where it's already validated on battery
data: used in production, tested against public Sandia/HUST datasets, and checked against an
independent researcher's own data. That same code, with zero changes, now also runs correctly
on six brand-new kinds of robot physics it had never seen before. No GPU cluster needed
anywhere — checking one new reading is actually faster on a CPU than a GPU (see "Speed"
below), and training a full model takes about 3 minutes on a laptop CPU alone.

**2. Each of the six robot examples is a first attempt, not a finished product.** The physics
rules and training data for each one are new code, tested for the first time here. The method
runs correctly in every case; some faults are caught very well, others only partly — which is
a normal, healthy result for brand-new examples, not a failure. They're public so other people
can build on them rather than starting over.

**In short: "MW works on robotics in general" is well supported, by six separate examples.
"These six specific examples are finished and polished" is not true yet — they're version one.**

## Results for all six examples (measured 2026-10-31)

| Example | False alarms on normal data | How well it catches each fault (1.00 = perfect) |
|---|---|---|
| flexible_robot (beam) | 1.6% | no vibration decay **1.00** · a glitch in the bend reading 0.64 · lost stiffness 0.57 |
| joint_actuator | 0.4% | won't settle down **0.97** · torque sensor glitch 0.91 · more friction 0.55 |
| planar_arm | 0.1% | encoder drift 0.89 · velocity glitch 0.89 · bent link 0.67 |
| mobile_robot | 1.7% | overspeed **1.00** · wheel slip 0.89 · gyro glitch 0.73 |
| legged_contact | 0.7% | bad bounce **1.00** · energy from nowhere **1.00** · sinking through the floor 0.99 |
| actuator_thermal | 2.4% | cooling failure **1.00** · flux sensor glitch 0.98 · hot spot / short 0.57 |

**Pattern 1 — a fault that clearly breaks the physics is caught very well, every time.** Lost
damping, a joint that won't settle, overspeed, a bad bounce, energy appearing out of nowhere,
a cooling failure — every one of these scores 0.97 to 1.00, across five unrelated examples.
That's the same kind of thing being caught the same way, over and over — not a lucky result on
one run.

**Pattern 2 — a small, scaled-down fault is harder to catch, every time.** More friction
(0.55), a bent link (0.67), a hot spot in a motor winding (0.57) — all three share the same
shape: a small, proportional change that can look similar to normal variation instead of
clearly breaking a physical rule. This shows up the same way in three unrelated examples, so
it's a real pattern worth fixing, not noise from one run. [FIXING_THE_GAPS.md](FIXING_THE_GAPS.md)
lays out specific steps to try — starting with testing a bigger version of the same fault
first, before changing any code.

**Pattern 3 — in the pure-geometry and bounce examples, made-up (but "normal") states satisfy
their own physics rules more loosely than real data does.** `planar_arm`, `mobile_robot` and
`legged_contact` generate states whose physics-rule error is 10–30x bigger than the real
sensor noise. The beam, joint, and thermal examples stay much closer, about 1x. Fault
detection still works well in two of the three looser ones — this is a separate, specific
finding about what the model generates by default, with concrete next steps in
[FIXING_THE_GAPS.md](FIXING_THE_GAPS.md).

## The beam's one test that came out backwards

One test (borrowed from a similar test in the parent repo) checks what happens when a coarse,
low-detail simulator is used instead of a precise one. For most examples, a coarser simulator
should look "less normal" to MW. For the beam, it was the opposite — the coarser the
simulator, the *more* normal it looked, and the fault-detection rate stayed flat no matter how
coarse it got.

Here's why: a coarse simulator drops a vibration mode, which removes energy from the system.
The beam's current rules only flag a trajectory that has *too much* unexplained energy, not
one that's suspiciously quiet. [FIXING_THE_GAPS.md](FIXING_THE_GAPS.md) proposes a one-line fix
that's already supported by existing code.

## Predicting the future: the first real test

Everything above checks whether a *given* state looks real. One example — `legged_contact` —
has also been tested as a **predictor**: trained directly on real MuJoCo data, guessing a
future state from a current one, with no physics solver running during the guess itself. Full
story, including three real bugs found and fixed, in [FORECASTER.md](FORECASTER.md). Short
version:

- A single jump forward in time (up to about 0.4 real seconds, skipping ~200 small simulation
  steps) can be predicted faster than MuJoCo itself can step through that same interval, once
  the jump is big enough (roughly 50-100 steps). The cost: the guess is off by roughly 7-14%
  of the value's normal range, and that error grows as the jump gets bigger.
- We also tried the same richer-data recipe used for the battery work (many randomly varied
  examples instead of a fixed grid). It fixed the **checker** job dramatically — the false-alarm
  rate on unseen conditions dropped from 15.6% to 0.5%. It did **not** help the **predictor**
  job — accuracy stayed about the same. Full numbers in FORECASTER.md, Part 3.
- This is **not yet** "MW can replace MuJoCo as a simulator." Only one example has been tried,
  on one held-out trajectory, with single jumps between two *already-known* real states — not
  a chain of predictions where each one builds on the last guess instead of a real
  measurement. See FORECASTER.md for that chaining test.
- The other examples that could map onto MuJoCo (`joint_actuator`, `planar_arm`,
  `mobile_robot`) haven't been tried as predictors yet.

## Speed: no GPU needed, measured

How long it takes to check whether one state is physical (median time, after warm-up; this
laptop's CPU — i5-12450H — vs. its RTX 3050 GPU):

| Example | CPU, one check | GPU, one check | CPU, 1000 at once | GPU, 1000 at once |
|---|---|---|---|---|
| flexible_beam | 1.14 ms | 3.43 ms | 107.4 ms | 3.63 ms |
| joint_actuator | 1.30 ms | 2.57 ms | 50.3 ms | 2.43 ms |
| planar_arm | 1.08 ms | 2.07 ms | 87.3 ms | 2.04 ms |
| mobile_robot | 1.08 ms | 3.39 ms | 90.0 ms | 3.47 ms |
| legged_contact | 1.28 ms | 2.08 ms | 70.1 ms | 2.11 ms |
| actuator_thermal | 1.23 ms | 3.56 ms | 58.3 ms | 3.53 ms |

Checking one reading at a time — how you'd actually use this in practice — the CPU is 2-3x
*faster* than the GPU, because sending work to a GPU has a fixed cost that only pays off with
a lot of work at once. The GPU wins only when checking many states in one batch (15-30x faster
at 1000 at a time), which matters for big sweeps, not for reading one sensor value at a time.

Training (`flexible_robot`, 200 epochs on 4000 rows, nothing changed except the device): GPU
105.6 seconds (527.7 ms per epoch) vs. CPU 181.7 seconds (908.4 ms per epoch) — about 1.7x
slower on CPU, but still just 3 minutes on a laptop. You don't need a GPU to train this.

## The bigger idea behind the predictor test

A companion paper, Modak & Walawalkar, "World Models with an Irreversible Time Axis" (Zenodo,
DOI [10.5281/zenodo.22954118](https://doi.org/10.5281/zenodo.22954118), 2026), describes the
piece these six examples are still missing: instead of checking one state, learn a map of all
the states a system could ever reach, with time flowing only one way, and predict the future
by walking across that map from where you are now toward where physics says you're headed.

What's been shown so far, and what's left:
- **Proven in the battery domain already; a smaller, simpler version now tried here.** The
  paper's own evidence comes from a real EV fleet deployment, plus matching the results of a
  published battery-cracking study using the study's own data. The test in
  [FORECASTER.md](FORECASTER.md) is related but much smaller: a straight-line guess between
  two *already known* real states, with none of the paper's actual machinery (no one-way time
  axis, no dedicated clock). It shows the general idea — walk the map instead of solving the
  equations — can work for a robot too, but it isn't a full build of the paper's design.
- **Even the paper itself doesn't ask the map to do everything alone.** It still needs an
  outside physics model to supply the target the map walks toward. Same here — both ends of
  every predicted "hop" in this repo were real, known states, not states supplied by an
  independent prediction. Predicting forward from just one current state, with nothing else
  known, still needs that missing piece.
- **A related idea (the "Lorentzian cone") has been tested on made-up data with injected
  faults**, and it works as designed: normal transitions cluster inside the expected zone,
  injected faults cluster outside it, and the false-alarm rate lands almost exactly where it
  was designed to. Testing it on real recorded faults is the next step.

Building the paper's actual design — one-way time, a dedicated clock, and a target supplied by
the system's own dynamics instead of a second known real state — is the real next step toward
a genuine predictor. `joint_actuator` and `legged_contact` are the easiest places to start,
since both already have an exact formula for how they evolve over time.

## Try it yourself
```bash
cd MW_Robotics && pip install -r requirements.txt
python examples/00_flexible_robot_validator.py          # or 01..05 - uses a GPU automatically if you have one (set MW_DEVICE=cpu to force CPU)
pytest -q
```
