# Fixing the gaps — concrete next steps, not just a list of problems

This is the companion to `ROBOTICS.md`. That file reports what six robotics domains show
today; this one proposes specific, testable fixes for the three patterns that showed up
across them, in the order a researcher should probably try them. Every suggestion below
references code that already exists in this repo — none of it needs new infrastructure.

## Pattern 1: faults that are a *scaled* change to one channel score weakly

Across every domain, the fault that is a proportional multiplier on one reading —
`friction_increase` (joint, AUC 0.55), `link_bent` (arm, 0.67), `hotspot_short` (thermal,
0.57) — scores far below the faults that are a structural break (no_settle, overspeed,
bad_restitution: all 0.97–1.00). That is a consistent signature, not three unrelated weak
spots, and it points at one mechanism.

**Likely cause.** The anomaly's injected scale factor is sampled from a range (e.g.
`friction_increase` cuts `omega` by 30–55%) that can land inside the natural spread of
normal data, because normal data already varies a lot across `theta_ref`, `t`, position
along the beam, etc. A fault that is "30% off" is only detectable if 30% off is actually
rare in the training distribution at that particular combination of the other features.

**What to try, in order:**
1. **Positive control first, same principle used for the Rutooj sweep in `PYBAMM_ALTERNATIVE`.**
   Before changing anything, inject the *same* fault at 2–3x the magnitude (e.g. cut
   `omega` by 80% instead of 30–55%) and confirm the AUC rises toward 1.0. If it does, the
   current weak AUC is a real sensitivity limit at that fault size, not a bug — useful to
   know and report as such. If it does NOT rise, something else is wrong (see 3 below).
2. **Check `feature_attribution()` for these fault rows specifically**
   (`det.feature_attribution(df)` in `detector.py`, already built). If the top driver it
   names is NOT the channel the fault actually touched, the manifold is explaining the
   anomaly away through a different coordinate instead of flagging the right one — the
   same failure mode the Lorentzian tutorial's "split encoder" lesson warns about
   (`BATTERY_RIEMANNIAN_LORENTZIAN.py`'s header: a single encoder lets an anomaly signal
   hide in the wrong latent coordinate).
3. **Compare against a naive single-feature threshold** (e.g. z-score of `omega` alone, or
   of the single law residual most relevant to the fault). If the naive threshold beats the
   full MW distance, the extra flexibility of the manifold is absorbing exactly the signal
   that should trigger it — the fix is a tighter `law_tolerance` on that specific
   constraint, not a bigger model.
4. **Only after 1–3**, consider shrinking `latent_dim` for the affected domain (less room
   for the decoder to "explain away" a subtle anomaly) or tightening `obs_sigma` on the
   one most diagnostic feature.

## Pattern 2: the prior-sampled (generated) states satisfy their own law 10–30x looser than the training noise — in the pure-kinematic and impact domains only

`flexible_robot`, `joint_actuator` and `actuator_thermal` (all ODE/PDE domains, laws tied
to a differential equation) generate states whose law residual is within about 1x the
training sensor noise. `planar_arm`, `mobile_robot` and `legged_contact` (pure algebraic/
kinematic identities, or a ballistic+impact law) generate states 10–30x looser. Anomaly
detection still works well in two of those three (`legged_contact` 0.99+, `mobile_robot`
0.73–1.00); it is a separate finding from Pattern 1, about what the unconditional prior
produces, not about fault sensitivity.

**Likely cause, two candidates to test, not yet distinguished:**
- **Scale mismatch.** `planar_arm`'s `x,y` range over roughly ±1.8 and `legged_contact`'s
  `z,vz` range over a few units; a `law_tolerance` of 0.03 is a much smaller fraction of
  that range than it is of the beam's or joint's more tightly-scaled features. Check
  `obs_sigma` and the constraint tolerance against the actual per-feature standard
  deviation of the training data (`MWNormaliser` already computes this) before assuming the
  architecture is at fault.
- **Latent capacity in low-data regions.** `docs/LIMITATIONS.md` point 4 already documents
  this exact risk: "too many [latent dims], and the decoder invents states in regions with
  no data; geodesics will cut through them." `geometry.py::path_support()` exists
  specifically to check this — run it on these three domains before assuming it's a
  tolerance problem rather than a dimensionality one.

**What to try, in order:**
1. Run `path_support()` (already in `geometry.py`) on `PlanarArmVAE`, `MobileRobotVAE`,
   `HoppingContactVAE` to see whether generated points actually fall in sparsely-trained
   latent regions.
2. If so, reduce `latent_dim` for these three (currently 5, 5, 4) toward the system's true
   degrees of freedom (`planar_arm` has 4 independent inputs — `theta1, theta2, theta1dot,
   theta2dot` — everything else is derived; `legged_contact` has 2 — `t, v0` — plus `e`).
3. If not, recheck `obs_sigma` scaling per feature against each domain's actual data range,
   the same check `docs/LIMITATIONS.md` point 2 recommends ("start loose, tighten while
   watching held-out AUC").

## Pattern 3: the beam's coarse-simulator test moved the wrong way (already in ROBOTICS.md)

Documented there as: dropping a vibration mode removes energy, and the beam's current laws
don't penalise a trajectory that is too *smooth*, only one that is too energetic or
inconsistent. This one has a concrete, small fix already available in the codebase:

**`core.py::bound()` already accepts both `low` and `high`.** The beam's current C2 is
one-sided: `bound("c2_energy_envelope", excess - ceiling, high=0.0, ...)` only penalises
*exceeding* the decay envelope. Replacing it with a narrow two-sided band around the
expected decay curve —
`bound("c2_energy_envelope", excess, low=0.9*ceiling, high=1.1*ceiling, ...)` (tune the
0.9/1.1 width against held-out normal data first) — would also penalise a trajectory that
decays unnaturally fast, which is exactly what the coarse simulator with mode 2 dropped
does. This is a one-line change to test, not new infrastructure. Re-run
`examples/00_flexible_robot_validator.py`'s coarse-fidelity sweep after making it, and
check the direction flips the right way before declaring it fixed.

## How to tell a fix actually worked

Use the same standard already applied to the battery evidence in
`PYBAMM_ALTERNATIVE/EVIDENCE.md`, not a single improved number:
1. **Re-run with a different seed.** A single-seed AUC jump from 0.57 to 0.85 could be
   retraining noise; check it holds across at least 3 seeds before writing it up.
2. **Re-run the positive control alongside the real fix**, so a change that only works
   because the fault got easier (bigger injected magnitude) is distinguishable from a change
   that works because the model actually improved at the original magnitude.
3. **Report the faults that stay weak, not just the ones that improve.** The honest
   write-up style this repo already uses (see `ROBOTICS.md`) is the standard to hold a fix
   to as well — a partial improvement reported plainly is more useful to the next person
   than a single good number with the rest quietly dropped.
