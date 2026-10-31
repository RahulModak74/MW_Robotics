# Robotics in MW_Robotics — what this is, honestly (2026-10-31)

This folder is `MW_Framework` plus one new domain: `mw_framework/domains/flexible_robot.py`,
the Euler-Bernoulli beam PDE that governs a flexible-link robot arm or a soft-robot segment
(a rigid robot's joints are ODEs; its compliant links and soft actuators are this PDE).
Run `python examples/00_flexible_robot_validator.py`.

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

## What this does NOT yet show (measured 2026-10-31, not glossed over)
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

## Reproduce
```bash
cd MW_Robotics && pip install -r requirements.txt
python examples/00_flexible_robot_validator.py          # GPU auto-used if present (MW_DEVICE=cpu to force CPU)
```
