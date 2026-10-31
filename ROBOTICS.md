# Robotics in MW_Robotics — status and limits (2026-10-31)

This folder is `MW_Framework` plus one new domain: `mw_framework/domains/flexible_robot.py`,
the Euler-Bernoulli beam PDE that governs a flexible-link robot arm or a soft-robot segment
(a rigid robot's joints are ODEs; its compliant links and soft actuators are this PDE).
Run `python examples/00_flexible_robot_validator.py`.

## Status, in two tiers — read this before anything else

**Tier 1 — core MW mechanism: a proven result, and the reason this is worth building on.**
`core.py`'s machinery (priors as likelihood terms, a VAE on a physics-constrained manifold,
O(n) inference, MW distance as an anomaly score, calibration) is unchanged from the parent
`MW_Framework` repo, where it is validated on battery data: production deployment, open
Sandia/HUST held-out cells, and an independent researcher's own sweep (see
`PYBAMM_ALTERNATIVE/EVIDENCE.md` in the battery repo). That mechanism also ran correctly here,
unedited, on a brand-new robotics PDE it had never seen code for, on the first attempt — no
GPU cluster, no PDE solver at inference time. That is the part of this repo backed by real
evidence, and it's a genuinely rare thing to have working at all.

**Tier 2 — the robotics domain itself: the first experiment on new ground.** `flexible_robot.py`'s
three physics laws and the beam dataset are new, untested-by-anyone-else code. The mechanism
runs; one fault is caught cleanly already, and two are only partly separated so far (numbers
below) — exactly the kind of result a first experiment on a brand-new domain should produce:
strong enough to show the approach is real, specific enough to show where to push next. It's
public as a starting point so other researchers can build on it, extend it, or improve the
laws, rather than starting from zero.

**The two claims to keep separate: "the MW mechanism generalizes to robotics" is supported by
this run. "This specific robotics validator is finished" is not — it's version one.**

## What this shows
1. **The MW mechanism transfers unchanged to a new, genuine robotics PDE.** Zero edits to
   `core.py`, `geometry.py`, `algebra.py`, `detector.py`. Only a new domain file with this
   PDE's own laws was needed — exactly the pattern the parent repo's other six domains
   already demonstrate (battery, RF, Navier-Stokes, Ohm's law...), now extended one more step.
2. **One fault type is caught cleanly.** "no_decay" (a stuck damper / loose joint — the
   robot keeps vibrating long after it should have settled) scores ROC-AUC 1.00.
3. **The learned manifold itself obeys the laws.** States generated from pure noise through
   the trained decoder satisfy the moment-curvature constitutive law and the strain limit to
   within the training sensor noise.

## Where the next iteration should focus (measured 2026-10-31)
1. **Two of three labelled faults are only partly separated so far.** `stiffness_loss` (a
   crack: local loss of stiffness) and `moment_glitch` (a moment-sensor fault) score ROC-AUC
   ~0.57 and ~0.64 — real signal, not yet a clean separation. A good target for the next
   version of the laws.
2. **The coarse-simulator validator test came back the wrong way round.** Example 06
   (Navier-Stokes, in the parent repo) showed MW distance rising as a coarse solver's grid
   step grew. The same test here went the other way: distance fell as the approximate beam
   simulator's fidelity dropped, and the flag rate stayed flat at the ~1% noise floor across
   every fidelity level. Likely cause, diagnosed rather than left a mystery: dropping a
   vibration mode removes energy, and the three laws as written don't yet penalise a
   trajectory that is too smooth/quiet, only one that is too energetic or inconsistent. A
   specific, fixable gap — reproduced on rerun, so it's a real finding, not noise.
3. **This is a validator, not yet a forecaster.** It checks whether a given state is
   physical; it doesn't yet predict the next state from the current one, the way a robot
   simulator (or the battery SoH geodesic walk in the production battery product) does.
   Turning this into a forward-time model that stands in for a robotics PDE solver is the
   natural next step — see "Related theory" below for the architecture that points the way.

## What this is evidence of
The mechanism is portable to a genuine, novel robotics PDE with no changes to the core
framework code — proven on this run. Whether a strong robotics validator or forecaster can be
built on top of it is the open, exciting part: this first attempt is already strong on one
fault and has a clear, diagnosed path forward on the other two. Next step: sharpen the laws
(or the feature set) so the fidelity-sweep test moves in the right direction, re-test, and
build from there — the same standard already applied to the battery evidence in
`PYBAMM_ALTERNATIVE/EVIDENCE.md`.

## Related theory: the architecture for closing gap 3 above

Modak & Walawalkar, "World Models with an Irreversible Time Axis" (Zenodo, DOI
[10.5281/zenodo.22954118](https://doi.org/10.5281/zenodo.22954118), 2026) proposes exactly the
piece this repo is missing: instead of a learned transition function, learn a manifold of
admissible states with a forward-only time axis, and *forecast* by walking the manifold
between the current state and a physics-projected target, rather than only checking a single
given state. It frames MW as a fourth family of world models, alongside latent imagination
(Dreamer), joint-embedding prediction (JEPA) and generative video (Genie/Cosmos) — none of
which can express "this state cannot be reached from that one."

What it's shown so far, and what's still ahead:
- **Proven in battery, not yet applied to robotics.** The paper's evidence is an EV fleet
  analytics deployment, plus an external validation against a published SEI-cracking study
  where the result was reproduced on the study author's own data and confirmed by them.
  Building the robotics version is the natural next step, not a result either repo has yet.
- **Even in the paper, the manifold isn't asked to do everything alone.** Its own Limitations
  section notes forecast *endpoints* still come from an external physics model; the manifold
  shapes the path between them. For the beam PDE, that means pairing the manifold with
  something that says where a trajectory ends up — a well-scoped, tractable next piece.
- **The Lorentzian cone (closest to this repo's anomaly work) has been tested on synthetic
  data and injected faults**, with real signal: normal transitions cluster inside the cone,
  injected fast-fade events cluster outside it, and the ~5% false-positive rate matches its
  95th-percentile calibration exactly as designed. Testing on recorded real fast-fade events
  is the next step there too.

Adapting this pattern to `FlexibleRobotVAE` — a forward-only time axis, a dedicated clock
encoder separate from the state encoder, and a target state fed in from the beam's known
linear decay — is the concrete next step toward a real forecaster, not just a validator.

## Reproduce
```bash
cd MW_Robotics && pip install -r requirements.txt
python examples/00_flexible_robot_validator.py          # GPU auto-used if present (MW_DEVICE=cpu to force CPU)
```
