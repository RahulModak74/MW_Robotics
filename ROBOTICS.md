# Robotics in MW_Robotics — what this is, honestly (2026-10-31)

This folder is `MW_Framework` plus one new domain: `mw_framework/domains/flexible_robot.py`,
the Euler-Bernoulli beam PDE that governs a flexible-link robot arm or a soft-robot segment
(a rigid robot's joints are ODEs; its compliant links and soft actuators are this PDE).
Run `python examples/00_flexible_robot_validator.py`.

## Status, in two tiers — read this before anything else

**Tier 1 — core MW mechanism: works, pushed on that basis.** `core.py`'s machinery (priors
as likelihood terms, a VAE on a physics-constrained manifold, O(n) inference, MW distance as
an anomaly score, calibration) is unchanged from the parent `MW_Framework` repo, where it is
validated on battery data: production deployment, open Sandia/HUST held-out cells, and an
independent researcher's sweep (see `PYBAMM_ALTERNATIVE/EVIDENCE.md` in the battery repo).
That mechanism also ran correctly here, unedited, on a brand-new robotics PDE it had never
seen code for. That transfer is a real, demonstrated result — not an aspiration.

**Tier 2 — the robotics domain itself: a prototype, not a result.** `flexible_robot.py`'s
three physics laws and the beam dataset are new, untested-by-anyone-else code. The honest
status is "mechanism confirmed to run; whether these specific laws make a USEFUL robotics
validator is open and partly negative" (one fault caught, two weak, one headline test
backwards — see below). This is pushed as a prototype and an invitation: the repo, the bug
found while building it, and the failure are all public so a researcher can pick it up,
disagree with the laws, or fix them, rather than taking "it works" on faith.

Keep these two claims separate when talking about this repo: **"the MW mechanism generalizes"
is supported; "this robotics validator works" is not yet.**

## What this shows
1. **The MW mechanism (priors as likelihood terms, VAE on the constrained manifold, O(n)
   inference, MW distance as an anomaly score) transfers unchanged to a new, genuine
   robotics PDE.** Zero edits to `core.py`, `geometry.py`, `algebra.py`, `detector.py`. Only
   a new domain file with this PDE's own laws was needed — exactly the pattern the repo's
   other six examples already demonstrate (battery, RF, Navier-Stokes, Ohm's law...).
2. **One fault type is caught cleanly.** "no_decay" (a stuck damper / loose joint — the
   robot keeps vibrating long after it should have settled) scores ROC-AUC 1.00.
3. **The learned manifold itself obeys the laws.** States generated from pure noise through
   the trained decoder satisfy the moment-curvature constitutive law and the strain limit to
   within the training sensor noise.

## What this does NOT yet show (measured 2026-10-31)
1. **Two of three labelled faults are only weakly caught.** `stiffness_loss` (a crack: local
   loss of stiffness) and `moment_glitch` (a moment-sensor fault) score ROC-AUC ~0.57 and
   ~0.64 — barely above chance.
2. **The coarse-simulator validator test came back backwards.** Example 06 (Navier-Stokes)
   showed MW distance rising as a coarse solver's grid step grew. The same test here showed
   the OPPOSITE: distance fell as the approximate beam simulator's fidelity dropped, and the
   flag rate stayed flat at the ~1% noise floor across every fidelity level. The likely cause:
   dropping a vibration mode removes energy, and the laws as written do not penalise a
   trajectory that is too smooth/quiet, only one that is too energetic or inconsistent. That
   is a real gap in these three laws, not sensor noise or a training-run fluke (reproduced on
   rerun).
3. **This is a validator, not a forecaster.** It checks whether a GIVEN state is physical. It
   does not predict the next state from the current one, the way a robot simulator (or the
   battery SoH geodesic walk in the battery product) does. Replacing a forward-time PDE
   solver — the actual "bypass Isaac Sim for physics" claim — needs that second kind of
   model, trained the way the battery forecaster was (on trajectories, scored against
   held-out ground truth over time), and it does not exist here yet.

## What this is evidence of, stated carefully
The MECHANISM is portable to a genuine, novel robotics PDE with no changes to the core
framework code — that much is now demonstrated, not just argued. Whether a USEFUL robotics
validator or forecaster can be built this way is still open: this first attempt caught one
fault well and two poorly, and failed its own version of example 06's headline test. The
honest next step is to fix the laws (or the feature set) so the fidelity-sweep test passes in
the right direction, re-test, and only then write up a result — the same standard already
applied to the battery evidence in PYBAMM_ALTERNATIVE/EVIDENCE.md.

## Related theory: what would close gap 3 above

Modak & Walawalkar, "World Models with an Irreversible Time Axis" (Zenodo, DOI
[10.5281/zenodo.22954118](https://doi.org/10.5281/zenodo.22954118), 2026) proposes the
architecture this repo's gap 3 is missing: instead of a learned transition function, learn a
manifold of admissible states with a forward-only time axis, and *forecast* by walking the
manifold between the current state and a physics-projected target, rather than checking a
single given state.

Read plainly, not oversold:
- **This has not been applied to robotics.** The paper's only evidence is battery (an EV
  fleet analytics deployment, plus an external validation against a published SEI-cracking
  study where the result was reproduced on the study author's own data and confirmed by
  them). Building the robotics version is future work, not a result either repo has.
- **Even in the paper, it is not a from-scratch simulator.** Its own Limitations section
  says forecast *endpoints* still come from an external physics model; the manifold only
  shapes the path between them. That matters here: closing gap 3 for the beam PDE would
  still need something outside the manifold to say where a trajectory ends up.
- **The Lorentzian cone (the part closest to this repo's anomaly work) is tested on
  synthetic data and injected faults only**, not recorded real fast-fade events, and about
  5% of normal transitions fall outside the cone by construction (95th-percentile
  calibration) — the same kind of honest ceiling this repo's own fault-detection AUCs have.

Adapting this pattern to `FlexibleRobotVAE` — a forward-only time axis, a dedicated clock
encoder separate from the state encoder, and a target state fed in from the beam's known
linear decay — is the concrete next step toward a real forecaster, not just a validator.

## Reproduce
```bash
cd MW_Robotics && pip install -r requirements.txt
python examples/00_flexible_robot_validator.py          # GPU auto-used if present (MW_DEVICE=cpu to force CPU)
```
