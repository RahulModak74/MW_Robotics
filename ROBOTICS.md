# Robotics in MW_Robotics — status and limits (2026-10-31)

This folder is `MW_Framework` plus six new domains, each a genuinely different piece of
robotics physics: a PDE (a flexible-link beam), two different ODEs (a joint actuator, a
mobile robot's nonholonomic kinematics), a pure-kinematics chain (a planar arm), a
hybrid/impact problem (a legged robot's foot contact), and a second PDE (motor thermal
diffusion). Run any of `examples/00_flexible_robot_validator.py` through
`examples/05_actuator_thermal_validator.py`. See [FIXING_THE_GAPS.md](FIXING_THE_GAPS.md)
for concrete, testable next steps on every gap below.

## Status, in two tiers — read this before anything else

**Tier 1 — core MW mechanism: a proven result, and the reason this is worth building on.**
`core.py`'s machinery (priors as likelihood terms via Pyro SVI, a VAE on a physics-constrained
manifold, O(n) inference, MW distance as an anomaly score, calibration) is unchanged from the
parent `MW_Framework` repo, where it is validated on battery data: production deployment,
open Sandia/HUST held-out cells, and an independent researcher's own sweep (see
`PYBAMM_ALTERNATIVE/EVIDENCE.md` in the battery repo). That same mechanism, unedited, now also
runs correctly on six brand-new robotics physics systems it had never seen code for — a PDE,
two different ODEs, pure kinematics, a nonholonomic constraint, and a hybrid/impact problem.
No GPU cluster is needed at any point: inference is faster on CPU than GPU for single
readings (see "Performance" below), and even full training completes in 3 minutes on a
laptop CPU alone. That's the part of this repo backed by real evidence, and it's a genuinely
rare thing to have working at all, let alone six times over.

**Tier 2 — each robotics domain: a first experiment on new ground.** Every domain's three
physics laws and dataset are new, untested-by-anyone-else code. The mechanism runs in every
case; some faults are caught cleanly already, others only partly — exactly the kind of result
a first wave of experiments on brand-new domains should produce: strong enough to show the
approach is real, specific enough to show where to push next. They're public as a starting
point so other researchers can build on them, extend them, or improve the laws, rather than
starting from zero.

**The two claims to keep separate: "the MW mechanism generalizes to robotics" is supported by
six independent runs. "These six specific robotics validators are finished" is not — they're
version one.**

## Results across all six domains (measured 2026-10-31)

| Domain | False alarms (held-out normal) | Fault ROC-AUCs |
|---|---|---|
| flexible_robot (beam) | 1.6% | no_decay **1.00** · moment_glitch 0.64 · stiffness_loss 0.57 |
| joint_actuator | 0.4% | no_settle **0.97** · torque_sensor_glitch 0.91 · friction_increase 0.55 |
| planar_arm | 0.1% | encoder_drift 0.89 · velocity_glitch 0.89 · link_bent 0.67 |
| mobile_robot | 1.7% | overspeed **1.00** · wheel_slip 0.89 · gyro_glitch 0.73 |
| legged_contact | 0.7% | bad_restitution **1.00** · energy_gain **1.00** · ground_penetration 0.99 |
| actuator_thermal | 2.4% | cooling_failure **1.00** · flux_sensor_glitch 0.98 · hotspot_short 0.57 |

**Pattern 1 — faults that are a structural break are caught very well everywhere.** Lost
damping, a stuck/runaway joint, overspeed, bad restitution, an impossible mid-flight energy
gain, a cooling failure: every one of these scores 0.97–1.00, across five unrelated domains.
That's not a coincidence of one lucky run; it's the same mechanism catching the same *kind*
of fault — "this trajectory violates a hard physical identity" — every time it's tried.

**Pattern 2 — faults that are a subtle, scaled change to one channel are consistently
weaker.** `friction_increase` (0.55), `link_bent` (0.67), `hotspot_short` (0.57) all share
the same shape: a proportional perturbation that can overlap the normal range rather than
breaking an identity outright. This is a real, repeated signature across three unrelated
domains, not noise in one run — and [FIXING_THE_GAPS.md](FIXING_THE_GAPS.md) lays out the
specific, testable steps to find out why and fix it (starting with a positive control at a
larger fault magnitude, before changing any code).

**Pattern 3 — generated (prior-sampled) states satisfy their own law far more loosely in the
pure-kinematic and impact domains.** `planar_arm`, `mobile_robot` and `legged_contact`
generate states whose law residual is 10–30x the training sensor noise; the ODE/PDE domains
(`joint_actuator`, `actuator_thermal`, the beam) stay within about 1x. Anomaly detection
still works well in two of the three looser domains — this is a separate, specific finding
about what the unconditional prior produces, with concrete next steps in
[FIXING_THE_GAPS.md](FIXING_THE_GAPS.md).

## The beam's coarse-simulator test: one result that went the wrong way, diagnosed

Example 06 (Navier-Stokes, in the parent repo) showed MW distance rising as a coarse solver's
grid step grew. The equivalent test for the beam went the other way: distance fell as the
approximate simulator's fidelity dropped, and the flag rate stayed flat at the noise floor
across every fidelity level. Diagnosed, not left a mystery: dropping a vibration mode removes
energy, and the beam's current laws don't yet penalise a trajectory that is too smooth/quiet,
only one that is too energetic or inconsistent.
[FIXING_THE_GAPS.md](FIXING_THE_GAPS.md) proposes a concrete one-line fix, already supported
by existing code (`core.py::bound()` already accepts a two-sided band).

## Not yet a forecaster

All six domains are validators: they check whether a *given* state is physical. None of them
yet predicts the *next* state from the current one, the way a robot simulator (or the battery
SoH geodesic walk in the production battery product) does. Turning any of these into a
forward-time model that stands in for a robotics PDE/ODE solver is the natural next step —
see "Related theory" below for the architecture that points the way.

## Performance: no GPU required, measured

Inference (median of repeated `mw_distance` calls after warm-up; this machine's CPU —
i5-12450H — vs. RTX 3050 laptop GPU):

| Domain | CPU, 1 sample | GPU, 1 sample | CPU, batch 1000 | GPU, batch 1000 |
|---|---|---|---|---|
| flexible_beam | 1.14 ms | 3.43 ms | 107.4 ms | 3.63 ms |
| joint_actuator | 1.30 ms | 2.57 ms | 50.3 ms | 2.43 ms |
| planar_arm | 1.08 ms | 2.07 ms | 87.3 ms | 2.04 ms |
| mobile_robot | 1.08 ms | 3.39 ms | 90.0 ms | 3.47 ms |
| legged_contact | 1.28 ms | 2.08 ms | 70.1 ms | 2.11 ms |
| actuator_thermal | 1.23 ms | 3.56 ms | 58.3 ms | 3.53 ms |

At the realistic deployment pattern — one new reading at a time — CPU is 2–3x *faster* than
GPU, because GPU kernel-launch overhead dominates at batch size 1. GPU only wins at batch
scale (15–30x faster at 1000 at once), which matters for sweeps and fleet-wide scoring.

Training (`flexible_robot`, 200 epochs on 4000 rows, identical code, only `MW_DEVICE`
changed): GPU 105.6 s (527.7 ms/epoch) vs. CPU 181.7 s (908.4 ms/epoch) — about 1.7x slower
on CPU, and still 3 minutes end to end on a laptop. Training doesn't need a GPU; it's just
somewhat faster with one.

## Related theory: the architecture for closing the "not yet a forecaster" gap

Modak & Walawalkar, "World Models with an Irreversible Time Axis" (Zenodo, DOI
[10.5281/zenodo.22954118](https://doi.org/10.5281/zenodo.22954118), 2026) proposes exactly the
piece these six domains are missing: instead of a learned transition function, learn a
manifold of admissible states with a forward-only time axis, and *forecast* by walking the
manifold between the current state and a physics-projected target, rather than only checking
a single given state. It frames MW as a fourth family of world models, alongside latent
imagination (Dreamer), joint-embedding prediction (JEPA) and generative video (Genie/Cosmos)
— none of which can express "this state cannot be reached from that one."

What it's shown so far, and what's still ahead:
- **Proven in battery, not yet applied to robotics.** The paper's evidence is an EV fleet
  analytics deployment, plus an external validation against a published SEI-cracking study
  where the result was reproduced on the study author's own data and confirmed by them.
  Building the robotics version is the natural next step, not a result either repo has yet.
- **Even in the paper, the manifold isn't asked to do everything alone.** Its own Limitations
  section notes forecast *endpoints* still come from an external physics model; the manifold
  shapes the path between them. For any of these six domains, that means pairing the manifold
  with something that says where a trajectory ends up — a well-scoped, tractable next piece.
- **The Lorentzian cone (closest to this repo's anomaly work) has been tested on synthetic
  data and injected faults**, with real signal: normal transitions cluster inside the cone,
  injected fast-fade events cluster outside it, and the ~5% false-positive rate matches its
  95th-percentile calibration exactly as designed. Testing on recorded real fast-fade events
  is the next step there too.

Adapting this pattern to any of the six domains here — a forward-only time axis, a dedicated
clock encoder separate from the state encoder, and a target state fed in from the system's
own known dynamics — is the concrete next step toward a real forecaster, not just a
validator. `joint_actuator` and `legged_contact` are the most tractable starting points:
both already have an exact, known time-evolution law to supply the target state from.

## Reproduce
```bash
cd MW_Robotics && pip install -r requirements.txt
python examples/00_flexible_robot_validator.py          # or 01..05 - GPU auto-used if present (MW_DEVICE=cpu to force CPU)
pytest -q
```
