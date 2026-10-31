# MW Robotics — code companion

**Modak-Walawalkar (MW) Framework, trimmed to one domain: robotics.**

The MW Framework learns the physics of a system from data and Bayesian priors (Pyro SVI on a
physics-constrained Bayesian VAE — not plain PyTorch; the priors and the uncertainty they
carry are load-bearing, not a penalty term bolted on afterward), with no GPU cluster and no
PDE solver at inference time. It is validated and in production in one hard domain: battery
fleet analytics, where it holds up against open datasets (Sandia, HUST) and against an
independent researcher's own data. That is the proven core.

This repository takes that same core and points it at a new frontier: **robotics**. It now
spans **six genuinely different robotics physics systems** — a PDE, two ODEs, a pure
kinematic chain, a nonholonomic constraint, and a hybrid/impact dynamics problem — with
**zero changes to the underlying machinery**. Same `core.py`, same Pyro SVI training loop,
same `MWDetector` API, six times over. See [ROBOTICS.md](ROBOTICS.md) for the full result —
strong in parts, with a clear, written-down map of where the next iteration should focus
([FIXING_THE_GAPS.md](FIXING_THE_GAPS.md)), which is exactly what a first wave of experiments
on new ground should produce.

Two Zenodo preprints underlie the approach: the theory above (DOI 10.5281/zenodo.19304813),
and ["World Models with an Irreversible Time Axis"](https://doi.org/10.5281/zenodo.22954118)
(DOI 10.5281/zenodo.22954118), which frames MW as a fourth family of world models alongside
Dreamer, JEPA and generative video — battery evidence so far; see "Read next" below for what
it opens up for robotics.

**Two claims, kept separate so credit lands where it's earned:**
- **The core MW mechanism works, and that is the headline.** It is pushed on the strength of
  real evidence (battery: production use, open-dataset validation, an independent
  researcher's sweep) and it carried over unedited to six brand-new robotics physics systems
  — no small thing for a method with no GPU requirement.
- **Each robotics domain in this repo is a first exploration of new ground**, not a finished
  product: new laws, tested once each, strong on some faults and still being tuned on
  others. They're public exactly so other researchers can build on them faster than starting
  from scratch. See `ROBOTICS.md`'s "Status, in two tiers" and `FIXING_THE_GAPS.md`.

## The idea in four lines

1. Physics confines a real system to a thin surface — a **manifold** — inside its sensor space.
2. A Bayesian VAE **learns** that surface from normal data.
3. Known laws enter as **priors and likelihood terms**, so the learned surface must obey them.
4. Anything new is scored by its **MW distance** from the surface. One forward pass. O(n) in the number of features.

```
classical:  physics -> PDEs -> numerical solve (grid, time steps)     -> answer
MW:         physics -> Bayesian priors -> learned manifold -> one pass -> answer
```

## The six robotics domains in this repo

| # | Example | System | Physics | Laws |
|---|---|---|---|---|
| 00 | `flexible_robot.py` | Flexible-link arm / soft-robot segment | Euler-Bernoulli beam PDE | moment-curvature, energy-decay envelope, strain limit |
| 01 | `joint_actuator.py` | A single PD-controlled motor + link | damped 2nd-order ODE | Newton's-2nd-law torque/dynamics identity, energy envelope, torque limit |
| 02 | `planar_arm.py` | 2-link planar manipulator | pure kinematics (no ODE/PDE) | forward-kinematics identity, Jacobian (velocity) identity, joint limits |
| 03 | `mobile_robot.py` | Differential-drive mobile robot | nonholonomic kinematics ODE | no-sideways-motion constraint, heading-rate identity, speed limit |
| 04 | `legged_contact.py` | Hopping/legged robot foot, flight phase | ballistic motion + restitution (hybrid/impact) | energy conservation in flight, non-penetration, restitution bound |
| 05 | `actuator_thermal.py` | Motor winding / actuator housing | 1D heat-diffusion PDE | Fourier's-law identity, decay envelope, over-temperature limit |

Each one follows the same recipe: an exact closed-form dataset (no numerical solver), three
physics laws written as Bayesian priors, and three realistic injected faults. Run any of
them:

```bash
pip install -r requirements.txt
python examples/00_flexible_robot_validator.py      # or 01..05 - GPU auto-used if present
pytest -q
```

## What the six experiments show

Full tables and discussion in [ROBOTICS.md](ROBOTICS.md); the pattern across all six, stated
plainly:

- **The mechanism transfers every time, with zero framework changes** — a PDE, two
  different ODEs, pure kinematics, a nonholonomic constraint, and a hybrid/impact problem,
  all trained the same way, all scored the same way.
- **"Obviously wrong dynamics" faults are caught very well everywhere**: lost damping,
  cooling failure, overspeed, bad restitution, an impossible energy gain — all score
  ROC-AUC 0.97–1.00.
- **"Subtly scaled" faults are consistently weaker**: a partial friction increase, a bent
  link, a partial winding short — all score ROC-AUC 0.55–0.67. That's a real, repeated
  signature across unrelated domains, not noise, and [FIXING_THE_GAPS.md](FIXING_THE_GAPS.md)
  lays out exactly how to test and fix it.
- **One coarse-simulator test (the beam) came back backwards** — also diagnosed, with a
  concrete one-line fix proposed, in [FIXING_THE_GAPS.md](FIXING_THE_GAPS.md).

A working robotics validator or simulator isn't proven yet. What is proven is that the
mechanism carries over to six different kinds of robotics physics with zero framework
changes, and every experiment's result points at a specific, testable next step rather than
a dead end.

## Performance: no GPU required, measured

Inference (median of repeated `mw_distance` calls, after warm-up, on this machine's CPU —
i5-12450H — and RTX 3050 laptop GPU):

| Domain | CPU, 1 sample | GPU, 1 sample | CPU, batch of 1000 | GPU, batch of 1000 |
|---|---|---|---|---|
| flexible_beam | 1.14 ms | 3.43 ms | 107.4 ms (0.107 ms/sample) | 3.63 ms |
| joint_actuator | 1.30 ms | 2.57 ms | 50.3 ms (0.050 ms/sample) | 2.43 ms |
| planar_arm | 1.08 ms | 2.07 ms | 87.3 ms (0.087 ms/sample) | 2.04 ms |
| mobile_robot | 1.08 ms | 3.39 ms | 90.0 ms (0.090 ms/sample) | 3.47 ms |
| legged_contact | 1.28 ms | 2.08 ms | 70.1 ms (0.070 ms/sample) | 2.11 ms |
| actuator_thermal | 1.23 ms | 3.56 ms | 58.3 ms (0.058 ms/sample) | 3.53 ms |

**At single-sample inference — scoring one new sensor reading as it arrives, the realistic
deployment pattern — CPU is faster than GPU**, by about 2–3x: GPU kernel-launch overhead
dominates at batch size 1 and never gets amortised. GPU only pulls ahead at batch scale
(1000 at once: 15–30x faster), which matters for sweeps and fleet-wide scoring, not for
reading one state at a time.

Training (same model, `flexible_robot`, 200 epochs on 4000 rows, identical code path, only
`MW_DEVICE` changed):

| Device | Time | Per epoch |
|---|---|---|
| GPU (RTX 3050 laptop) | 105.6 s | 527.7 ms |
| CPU (i5-12450H) | 181.7 s | 908.4 ms |

CPU training is about 1.7x slower than GPU here — and still 3 minutes, start to finish, on a
laptop. Training is practical on CPU alone; it doesn't need a GPU, it's just faster with one.

## What is in the box

```
mw_framework/
  core.py          MWNormaliser, MWVAE, constraint(), bound(), MWTrainer, calibrate(), risk_score()
  config.py        JSON domain configs -> ConfigVAE (linear / range / ratio laws) — generic, no config shipped here
  detector.py      MWDetector: scikit-learn API, explain(), save/load
  algebra.py       ManifoldAlgebra: product, fibre, glue, marginal_distance
  geometry.py      pullback metric, geodesics, path_support, LorentzianCone
  domains/
    flexible_robot.py    FlexibleRobotVAE   - Euler-Bernoulli beam PDE
    joint_actuator.py    JointActuatorVAE   - damped 2nd-order ODE
    planar_arm.py        PlanarArmVAE       - forward-kinematics + Jacobian
    mobile_robot.py       MobileRobotVAE     - nonholonomic kinematics
    legged_contact.py    HoppingContactVAE  - ballistic + restitution
    actuator_thermal.py  ActuatorThermalVAE - 1D heat-diffusion PDE
  datasets.py      circle/arc (generic framework sanity-check toys) + one exact generator per domain above
  checkpoint.py    one .pth file = weights + config + normaliser + calibration
  cli.py           train / score / demo-data
examples/
  00_flexible_robot_validator.py .. 05_actuator_thermal_validator.py
docs/
  LIMITATIONS.md   general framework limits — still fully applicable here
tests/             fast pytest suite (one known pre-existing flaky test, see below)
ROBOTICS.md        what is and is not shown, across all six domains
FIXING_THE_GAPS.md  concrete, testable next steps for every gap found so far
```

## Known issue carried over from the parent repo

`tests/test_mw.py::test_sklearn_sign_convention` fails intermittently at this fixed seed —
confirmed to fail identically in the unmodified parent `MW_Framework` repo too, so it predates
this trim and is not a robotics-specific regression. Left as-is rather than quietly tuned to pass.

## Read next

- **[ROBOTICS.md](ROBOTICS.md)** — the full result across all six domains and what it would take to push further.
- **[FIXING_THE_GAPS.md](FIXING_THE_GAPS.md)** — specific, testable fixes for every gap found, in the order to try them.
- **[docs/LIMITATIONS.md](docs/LIMITATIONS.md)** — what the MW approach is not, and where it fails in general.
- Modak & Walawalkar, ["World Models with an Irreversible Time Axis"](https://doi.org/10.5281/zenodo.22954118)
  (Zenodo, 2026) — the forward-only-time / causal-cone architecture that points toward closing
  this repo's validator-vs-forecaster gap (battery evidence so far; see ROBOTICS.md's "Related
  theory" section for what it opens up for a robotics version).

## Scope

Research code and six worked examples, not a production robotics simulator. Not a safety
system. See [ROBOTICS.md](ROBOTICS.md) and [docs/LIMITATIONS.md](docs/LIMITATIONS.md).

## License

Apache-2.0 (see `LICENSE`). If you use this, please cite `CITATION.cff` and the MW theory document.
