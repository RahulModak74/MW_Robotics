# Fixing the gaps — concrete next steps

This is the companion to `ROBOTICS.md`. That file reports what the six robot examples show
today. This one is about what to actually try next, for the three weak spots that showed up.
Every suggestion below uses code that's already in this repo — nothing new needs to be built.

## Gap 1: small, scaled-down faults are hard to catch

Across every example, a fault that's just "a bit more/less of something" — more friction in
the joint (score 0.55), a bent arm link (0.67), a hot spot in a motor winding (0.57) — scores
much worse than a fault that clearly breaks the physics (0.97 to 1.00). This same pattern shows
up in three unrelated examples, so it's worth chasing down.

**Likely reason.** The fault is sized as a percentage change (e.g. the friction fault cuts
speed by 30-55%), and normal data already varies quite a bit depending on other conditions. A
"30% off" reading only stands out if 30% off is actually unusual for that particular
combination of conditions — if normal readings already swing that much, the fault blends in.

**What to try, in this order:**
1. **First, test with a bigger version of the same fault** (e.g. cut speed by 80% instead of
   30-55%) before touching any code. If the score jumps toward 1.0, the current weak score is
   just a real limit on how small a fault can be and still get caught — worth knowing and
   saying so, not a bug to fix. If it does *not* improve, something else is wrong — move to
   step 3.
2. **Check which feature the model blames for these fault cases** (`det.feature_attribution(df)`,
   already built into `detector.py`). If it's pointing at the wrong feature — not the one the
   fault actually touched — the model is explaining the anomaly away through some other
   number instead of flagging the right one.
3. **Compare against a simple baseline**: just check if that one feature (or the one most
   relevant physics-rule error) looks unusual on its own, no model needed. If that simple
   check actually beats the full model, the model has too much flexibility and is absorbing
   the exact signal that should trip the alarm — the fix is a tighter rule, not a bigger model.
4. **Only after trying 1-3**, consider giving the model less room to work with (a smaller
   latent space) or tightening how much sensor noise it's allowed to assume on the one most
   telling feature.

## Gap 2: made-up "normal" states loosely satisfy their own physics rules

`flexible_robot`, `joint_actuator`, and `actuator_thermal` (the ones governed by a differential
equation) generate made-up states that stick close to the physics, about as close as real
sensor noise. `planar_arm`, `mobile_robot`, and `legged_contact` (pure geometry, or a
bounce/impact rule) generate states that are 10-30x looser. Fault detection still works well
on two of those three, so this is a separate issue from Gap 1 — it's about what the model
generates by default, not about missing faults.

**Two possible reasons, not yet tested against each other:**
- **A scale mismatch.** `planar_arm`'s numbers range over about ±1.8, `legged_contact`'s over
  a few units — a fixed tolerance can be a much bigger or smaller fraction of the real range
  depending on the example. Worth checking the actual spread of each feature before assuming
  the model itself is at fault.
- **Too much room in the model for these three.** `docs/LIMITATIONS.md` already warns about
  this exact risk: give the model more capacity than the system actually needs, and it starts
  inventing states in places with no real data to anchor it. There's already a function,
  `path_support()`, built to check for exactly this.

**What to try, in order:**
1. Run `path_support()` (already in `geometry.py`) on the arm, wheeled-robot, and hopping-leg
   examples to see if the made-up points are landing in parts of the model with little real
   training data.
2. If so, give the model less room to work with for these three (fewer latent dimensions,
   closer to how many numbers the system actually needs to describe its state — the arm needs
   4, the hopping leg needs 2).
3. If not, re-check the noise tolerance for each feature against its real-world spread, same
   advice as `docs/LIMITATIONS.md` point 2: start loose, then tighten while watching accuracy.

## Gap 3: the beam's coarse-simulator test came out backwards (also in ROBOTICS.md)

A coarse, low-detail simulator drops a vibration mode, which removes energy from the system.
The beam's current rules only flag *too much* unexplained energy, not a trajectory that's
suspiciously quiet. There's already a small, ready-to-try fix:

The code that checks this (`core.py::bound()`) already supports a two-sided check — low and
high. Right now the beam only uses the high side: it flags energy *above* the expected decay
curve, but not below it. Switching to a narrow band on both sides of the expected curve would
also catch a trajectory that decays unnaturally fast — exactly what the coarse simulator does.
This is a one-line change to try, not new code. After making it, re-run
`examples/00_flexible_robot_validator.py`'s coarse-simulator test and check that the result
now points the right direction before calling it fixed.

## How to tell if a fix actually worked

Don't trust a single improved number — the same standard already used for the battery work:
1. **Re-run with a different random seed.** A jump from 0.57 to 0.85 on one seed could just be
   noise from retraining. Check it holds across at least 3 seeds before writing it up.
2. **Re-run the "bigger fault" test alongside the real fix**, so you can tell whether an
   improvement is because the model actually got better, or just because the test got easier.
3. **Report what's still weak, not just what improved.** A partial improvement, reported
   honestly, is more useful to the next person than one good number with the rest left out.
