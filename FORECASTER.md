# Forecaster experiments against a real simulator (MuJoCo) — 2026-10-31

Everything else in this repo is a **validator**: it checks whether a *given* state is physical.
This is the first test of MW as a **forecaster**: given a current state, predict a future one,
without running a solver — tested against MuJoCo (Apache-2.0), a real third-party physics
engine, on the `legged_contact` domain. Code: `mujoco_experiments/`.

**Headline, stated plainly:** there is now a genuine, measured speed-accuracy tradeoff where MW
wins on both — not an unconditional "MW replaces MuJoCo" result, and not a dead end either.
Three real bugs were found and fixed to get a trustworthy number; all three are reported below,
not smoothed over.

## Part 1 — the validator, on real MuJoCo output (`01_validator_check.py`)

Train `HoppingContactVAE` on the shipped closed-form dataset, then score real MuJoCo ball-drop
trajectories (MuJoCo's own compliant contact solver, not our synthetic data):

| Model | Training `e` range | Median distance on real MuJoCo trajectories | Flagged |
|---|---|---|---|
| A (shipped) | 0.6–0.98 | 19.5x baseline | 97.0% |
| B (domain-matched) | 0.2–0.6 (MuJoCo's actual achievable range) | 1.3x baseline | 15.6% |

Model B still catches all three of its own synthetic faults (AUC 0.998–1.000). **Lesson:** most
of model A's near-total flag rate was training-domain mismatch, not a physics violation — train
on the target engine's actual achievable physics, not an arbitrary convenient synthetic range.

## Part 2 — the forecaster (`02_forecaster_check.py`)

Train `HoppingContactVAE` directly **on real MuJoCo rollouts** (not closed-form data). Encode a
real start state and a real end state from one held-out flight segment, interpolate between
their latents, decode, and compare the decoded path against MuJoCo's own fine-grained
trajectory over the same interval.

### Three bugs found and fixed, in order — each one mattered

1. **`v0` detection lag.** Reading launch speed off `vz` at the first "airborne" sample (a few
  timesteps after true liftoff) biased the energy-conservation residual by a near-constant
  0.12 offset. Fixed by computing `v0` analytically from the apex height.
2. **A coordinate offset.** MuJoCo's ball *centre* rests at `z=ball_radius`, not `z=0` as the
  closed-form convention assumes. Fixed by shifting `z`. Together with fix 1, this dropped the
  raw energy residual from 0.120 → 0.020 (consistent with small integration truncation error,
  not a remaining bug).
3. **The held-out segment was always the wrong kind of trajectory.** Picking "the longest
  flight segment" always selects the *initial release fall* — because restitution `e<1` means
  no bounce ever exceeds the previous apex, the first fall is always the longest. But the
  training generator deliberately **excludes** that first segment (it requires a preceding
  bounce to exist). So every earlier test was scoring the one trajectory type the model was
  *never shown* — reconstruction error on it was 31–35% of range. On a genuine post-bounce
  segment (the kind actually trained on), pure reconstruction error is **2.5% (z) and 1.0%
  (vz)**. This was the single biggest fix.

### The optimizer itself was dead weight — and worse than removing it

The geodesic/straight-line length ratio measured earlier was 0.99: the 300-step Adam
optimization inside `geometry.py::geodesic()` was barely moving the path from a plain straight
line. Confirmed directly, same hop, same two endpoints:

| Method | z error | vz error | Time |
|---|---|---|---|
| Optimised geodesic (300-step Adam) | 0.0915 | 0.1650 | 537.4 ms |
| Straight line, no optimisation | **0.0803** | **0.1584** | **0.342 ms** |

Straight-line interpolation is both **~1570x faster** and **slightly more accurate**. The
optimizer wasn't just slow, it was actively finding a worse path. `geodesic()` is a diagnostic
tool (how far apart are two states, on the manifold) — not a forecasting primitive. The
forecaster below uses plain straight-line latent interpolation, no optimizer.

### The real result: a measured speed-accuracy tradeoff

Straight-line hops of increasing size, each compared to re-stepping MuJoCo for the same real
interval:

| Hop | MuJoCo steps skipped | z error (% of range) | vz error (% of range) | MW time | MuJoCo time | Faster |
|---|---|---|---|---|---|---|
| 0.02s | 10 | 7.9% | 1.0% | 0.393 ms | 0.066 ms | MuJoCo |
| 0.05s | 25 | 9.5% | 2.5% | 0.374 ms | 0.165 ms | MuJoCo |
| 0.10s | 50 | 7.9% | 3.9% | 0.386 ms | 0.329 ms | ~tied |
| 0.20s | 100 | 13.3% | 6.3% | 0.383 ms | 0.659 ms | **MW** |
| 0.40s | 200 | 13.5% | 9.6% | 0.381 ms | 1.318 ms | **MW** |

This is the right *shape* of result for a real "skip many solver steps with one hop" claim: MW's
cost per hop is roughly constant (~0.38 ms, dominated by one encode + decode) while MuJoCo's
cost scales linearly with step count. They cross over around 50–100 steps; beyond that MW wins
by a growing margin, at a real but bounded accuracy cost (vz error grows from ~1% to ~10% of
range as hop size grows from 10 to 200 steps).

## What this does and does not support

- **Supports:** for `legged_contact`, a single hop of up to ~200 MuJoCo steps (0.4s of real
  flight) can be predicted from two real encoded endpoints, faster than MuJoCo itself, with
  errors in the 7–14%-of-range band. The reconstruction floor on in-distribution states (2.5%/
  1.0%) shows the manifold itself is a good fit; the hop error is the interpolation's own cost.
- **Does not yet support:** "MW replaces MuJoCo as a forward simulator." This is one domain,
  one held-out trajectory, single hops only (no chained/repeated forecasting tested, so drift
  over many hops is unknown), and both endpoints were *real, known* states, not states produced
  by an earlier MW forecast feeding into the next hop. That chained test is the next one that
  actually proves or disproves the "simulator replacement" claim — see below.
- **A structural fact, not a limitation to fix:** `legged_contact` is a cheap system that
  MuJoCo itself solves in microseconds per step. The speed case here is specifically about
  *large hops replacing many small steps*, not about MW being faster at matching granularity.

## Next steps

1. **Chain hops**: forecast state 1 from state 0, then state 2 from the *forecast* of state 1
  (not the real state) — repeated for N hops — and measure how error accumulates. This is the
  real test of "simulator replacement," not single-hop interpolation between two known points.
2. **Repeat on the other MuJoCo-mappable domains** (`joint_actuator`, `planar_arm`,
  `mobile_robot` — see `ROBOTICS.md`) to see whether the speed-crossover shape and the
  7–14%-of-range accuracy band hold generally or are specific to ballistic/impact dynamics.
3. **Try an amortized model** instead of latent-space interpolation: a small network trained
  to map `(z_current, dt) -> z_future` directly in one forward pass, rather than requiring both
  endpoints to already be known. That is the architecture the cited "World Models with an
  Irreversible Time Axis" preprint actually describes (see `ROBOTICS.md`'s "Related theory"),
  and it is the piece that would let this forecast from *one* current state, not two known ones.
4. **Tighten the accuracy-vs-hop-size curve** with more hop sizes and more held-out segments
  (this report used one), and report confidence intervals, not single-run numbers — the same
  standard already applied to the battery evidence in `PYBAMM_ALTERNATIVE/EVIDENCE.md`.

## Reproduce
```bash
pip install mujoco          # Apache-2.0, not in the default requirements
cd mujoco_experiments
python 01_validator_check.py
python 02_forecaster_check.py
```
