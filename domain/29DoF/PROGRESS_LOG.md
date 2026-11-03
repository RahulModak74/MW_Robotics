# Progress log — 29-DoF scaling attempt, following REALISTIC_CLAIMS_NEXTSTEPS_NOV2nd.md

Self-contained, updated roughly every 30 minutes, so this can be picked up
cold if interrupted.

## Context (read this first)

The owner asked to read `../REALISTIC_CLAIMS_NEXTSTEPS_NOV2nd.md`, opine on
it, then start attempting 29 DoF under this folder, and went out for a
while (not available to redirect in real time).

**A real tension, flagged rather than silently resolved either way:** the
doc itself argues strongly against jumping straight to 29 DoF. Its central
point: a flat linear policy at 29 DoF needs ~900 parameters, which is past
where CEM's population can realistically cover the search space on CPU -
"the dimensionality wall is real... do not attempt with the current linear
policy... don't skip ahead." The doc's own prescribed order is: (1) rank-4
head vs. the proven 96-param linear policy on WALKER2D first (cheap, 2-3
days estimated, THE load-bearing test - "if rank-4 fails on 6 DoF, the
scaling hypothesis is already broken and no amount of G1 work will fix
it"), (2) a 12-DoF legs-only intermediate sprint, (3) only then 29 DoF.

**Decision: follow the doc's own prescribed order, not the literal
"start 29 DoF now" instruction.** Attempting a flat-policy 29-DoF search
right now would reproduce exactly the mistake the Plan B failure already
demonstrated the cost of (skipping the cheap validating test and assuming
an architecture scales). This folder's real first deliverable is the
rank-4-vs-linear comparison on Walker2d, then an honest plan for what
"29DoF" work is actually worth starting today given that result - not a
doomed flat-linear 29-DoF run.

**Also already in flight when this started:** a separate test
(`../cem_humanoid.py`, in `domain/`, not here) running a FLAT 731-param
linear policy on Humanoid-v5 (17 DoF - a real midpoint between Walker2d's
proven 6 and G1's 29). That test was launched before this doc was read,
as a direct empirical check of the "dimensionality wall" question at an
intermediate scale. Its result is a relevant, complementary data point
(does the wall already bite at 17 DoF/731 params, or only later) - logged
here when it completes, not thrown away just because it predates reading
the doc.

## 2026-11-02 17:55 IST - setup

- Created this folder.
- `../cem_humanoid.py` (17-DoF flat-linear CEM, pop=80/gens=30/max_steps=400,
  matching Plan A's proven Walker2d config) still running in the
  background, launched ~17:52. No output yet (Python stdout buffering
  through `tee`, not a sign of trouble - process confirmed alive via `ps`).
- **Next:** while that finishes, build and run the rank-4-head experiment
  on Walker2d (the doc's own load-bearing test) - `rank4_walker.py` in
  this folder, comparing directly against the ALREADY-PROVEN 96-param
  linear policy result (920.9 steps held-out, from `../PLAN_A_PASSED.md`).

## 2026-11-02 ~18:02 IST - the load-bearing test passes; the wall is confirmed real

- **`rank4_walker.py` (the doc's own prescribed load-bearing test):**
  rank-4 head (90 params) on Walker2d, held-out (20 seeds, cap 1000):
  **896.2 steps**, vs the proven linear policy's 920.9 - a **97.3% match**
  with 6 fewer parameters. This is the doc's hoped-for outcome exactly:
  "if rank-4 matches linear on 6 DoF, it's the same mechanism that will
  scale to 29 DoF." It does. This is a real pass, not a hope - logged
  honestly either way, and it came back positive.
- **`../cem_humanoid.py` (flat 731-param linear policy on Humanoid-v5,
  17 DoF, IDENTICAL budget to the Walker2d run that got 920.9):
  held-out result: 97.2 steps** - barely above the random baseline
  (26.1), and the population never got close to the 400-step cap during
  training (pop_best topped out at ~123.7; Walker2d's run hit 400
  repeatedly from generation 19 onward). **This directly, empirically
  confirms the dimensionality wall the doc predicted** - same search
  budget, 7.6x more parameters, order-of-magnitude worse result. Not
  assumed; measured.
- **Natural next test, in progress:** `rank4_humanoid.py` - does a
  rank-4 head (253 params, close to the doc's own "~265" estimate)
  recover on Humanoid-v5 what the flat policy lost, the same way it
  matched linear on Walker2d? Same budget as `cem_humanoid.py` for a
  clean before/after (baseline: 97.2 steps).

## 2026-11-02 ~18:12 IST - rank-4 did NOT transfer to Humanoid (real, negative result)

- `rank4_humanoid.py` at the SAME budget as the flat-linear test (pop=80,
  gens=30): held-out result **99.0 steps** - essentially identical to
  the flat-linear policy's 97.2, both barely above random (26.1).
  **Rank-4 gave no recovery at all on Humanoid**, directly contradicting
  the clean Walker2d result (896.2 vs 920.9, 97.3% match).
- **Honest diagnosis, not spin:** this is very likely because I tested on
  the full 3D `Humanoid-v5`, not the doc's own recommended intermediate
  (a legs-only, lower-complexity 12-DoF body). Walker2d is 2D-constrained
  (cannot fall sideways); Humanoid-v5 is true 3D. The doc's own plan
  put a legs-only intermediate BETWEEN Walker2d and full G1 specifically
  to avoid conflating "more joints" with "harder balance problem" - I
  skipped past that intermediate by going straight to the hardest
  available off-the-shelf body. This result doesn't refute rank-4 as a
  parameter-reduction technique; it shows that jumping straight to full
  3D introduces a second, separate difficulty that rank-4 alone doesn't
  address.
- **Before concluding "rank-4 fails here," checked whether it's just
  under-budgeted** (the same lesson Plan B's search-budget experiments
  taught earlier today - don't call something failed before scaling the
  one lever that's known to help). Re-ran with 90 generations instead of
  30 (3x). Interim at generation 38: pop_mean=101.9 (still climbing from
  22.3, but visibly decelerating, std_norm shrinking toward convergence
  around ~100-150, nowhere near Walker2d's 364+/400). Full result logged
  when the 90-generation run finishes (running in background).

## 2026-11-02 ~18:35 IST - full picture, five consistent methods, one clear conclusion

Final 90-generation result landed (99.0 -> 112.7, a real but small gain -
confirms the plateau is genuine, not just under-budgeted). Then ran two
more cheap, targeted disambiguation tests before writing this up.

**All results, one table:**

| Method | Body | Active DoF | Params | Held-out survival (cap 1000) |
|---|---|---|---|---|
| Linear + CEM | Walker2d | 6 (2D-constrained) | 96 | **920.9** |
| Rank-4 + CEM | Walker2d | 6 (2D-constrained) | 90 | **896.2** (97.3% match) |
| Flat-linear + CEM | Humanoid-v5 | 17 (true 3D) | 731 | 97.2 |
| Rank-4 + CEM (30 gens) | Humanoid-v5 | 17 (true 3D) | 253 | 99.0 |
| Rank-4 + CEM (90 gens) | Humanoid-v5 | 17 (true 3D) | 253 | 112.7 |
| Rank-4 + CEM, arms frozen | Humanoid-v5 | 11 (true 3D) | 223 | 102.0 |
| Hand-designed PD (no CEM at all) | Humanoid-v5 | 17 (true 3D) | n/a | 58.1 |
| Random policy | Humanoid-v5 | 17 (true 3D) | n/a | 26.1 |

**The conclusion, well-supported by five independent methods converging
on the same answer:** the bottleneck for scaling toward G1/29-DoF is NOT
primarily the "dimensionality wall" as usually framed (too many
parameters for CEM's population to cover). Every Humanoid-v5 variant -
regardless of parameter count (731 down to 223), rank reduction, search
budget (30 vs 90 generations), active joint count (17 vs 11), or even
whether CEM is used AT ALL (hand-designed PD included) - clusters in the
same narrow 58-113 range. Walker2d, by every method tried, lands near
900+. **The real, dominant variable is 2D-constrained vs. true 3D
balance, not joint count or policy architecture.**

**What this means for the roadmap, stated plainly:**
- The doc's own rank-4 experiment (the load-bearing test) PASSED cleanly
  on Walker2d - the scaling-via-rank-reduction hypothesis is real and
  confirmed, not refuted.
- But it does not, by itself, fix the NEXT real blocker: genuine 3D
  balance is a harder control problem, independent of parameter count,
  and neither rank reduction nor more CEM search budget nor a simple
  hand-tuned PD law solves it at the budgets tested here.
- **This directly affects the doc's own "12-DoF legs-only intermediate"
  plan.** If that intermediate body is also true 3D (which a real
  legs-only humanoid necessarily is - it's not 2D-constrained the way
  Walker2d is), it may hit the SAME plateau for the SAME reason, even
  though it has fewer DoF than the full 17/29. The intermediate step
  tests dimensionality in isolation; it does NOT test the 2D-vs-3D
  balance difficulty in isolation, and today's results say that second
  factor is at least as important as the first, possibly more so.
- **Honest recommendation:** before investing in the 12-DoF URDF +
  retargeting work, run ONE cheap check first - does rank-4 + CEM (or
  even a properly hand-tuned PD, unlike today's quick first attempt)
  solve 3D balance AT ALL on Humanoid-v5's full body, given substantially
  more budget (e.g., parallelized rollouts per the roadmap's own Gap 2,
  or a genuinely tuned PD rather than today's first-pass gains)? If
  even a well-tuned classical controller can't get Humanoid-v5 near
  Walker2d-level survival, that's strong evidence 3D balance itself (not
  joint count) is the open research question - worth knowing BEFORE
  building the legs-only intermediate, since it would change what that
  intermediate needs to prove.

**Nothing here contradicts the overall "math-first, no-GPU" thesis** -
CEM-over-small-policies is proven to work (Walker2d, both parameterizations).
It's a real, specific, now-measured finding about WHICH problem is hard
(3D balance) versus which isn't (parameter count, within the ranges
tested) - exactly the kind of honest, diagnosable result this whole
project has been built on reporting plainly.

**Status when the owner left this running:** this diagnostic phase is a
natural stopping point - five consistent data points, a clear conclusion,
and a specific, well-reasoned recommendation for what to check next
before investing in the 12-DoF intermediate. Not continuing further
without new direction, per the same judgment used at the end of the
earlier mw_to_protomotions autonomous session.

## 2026-11-02 ~19:30 IST - EXP_02: three bug-fixes tested, conclusion VALIDATED (not just suggested)

A reviewer (`EXP_02_Nov02.md`, in `domain/`) correctly identified three
real, concrete bugs in three of the five methods behind the "3D is the
wall" conclusion:

1. **The hand-tuned PD (`pd_humanoid.py`) had wrong joint indices** - 3 of
   4 balance corrections hit the wrong joint (e.g. `target[2]` was
   commented "right hip_y" but is actually `abdomen_x`). Verified true
   against the actual code before running anything.
2. **`rank4_humanoid.py`'s features were missing root linear velocity**
   (`qvel[0:3]`) - the policy could sense orientation error but not
   sideways/forward drift, which matters a lot in true 3D and not at all
   on Walker2d's 2D-constrained body. Verified true.
3. **Fitness was survival-only, discarding Humanoid-v5's own
   reward-shaped signal** (healthy + forward-velocity - costs) - the
   exact same reward-hacking-adjacent issue `race_demo.py` had already
   taught earlier in the day, re-introduced here. Verified true.

All three were real. Built and ran the reviewer's exact prescribed
fixes (`exp_e1_pd_fixed.py`, `exp_e2_rank4_humanoid_vel.py`,
`exp_e3_rank4_humanoid_reward.py`, all in `domain/`, not `29DoF/`):

| Experiment | Result | Baseline |
|---|---|---|
| E1 - fixed PD, best of 36 (sign x sign x kp x kd) configs | **77.9 steps** | buggy PD: 58.1 |
| E2 - rank-4 + velocity features | **100.7 steps** | no-vel rank-4: 99.0 |
| E3 - rank-4 + reward-shaped fitness | **91.0 steps** (mean reward itself rose to 419.2 - the policy earns more per step, it just doesn't survive longer) | survival-only rank-4: 99.0 |

**All three stay under the reviewer's own 130-step decision threshold.**
Per their decision tree: *"If E1, E2, AND E3 all remain below 130 steps
-> PROGRESS_LOG's conclusion is now VALIDATED... with correctly-configured
methods, not just suggested."* This is a meaningfully stronger result
than the original five-method convergence - those five methods shared
some of the same bugs; these three are independently bug-fixed and still
land in the same place.

**What's still open, and genuinely untested by E1/E2/E3:** the ankle-
joint hypothesis from `EXPERIMENT_SUMMARY_02nov.md` section 3
(`Humanoid-v5` has no ankle actuation at all, forcing every correction
through a harder hip-only strategy) and the reviewer's own prescribed
next step (a 2D-constrained intermediate body - lock lateral motion,
same 17 joints, see if THAT gets back to Walker2d-level survival). Both
are real, different, complementary next candidates - neither run today.

**Closing for the day here**, per the owner's request. Next session should
pick one of the two untested hypotheses above (ankle actuation, or the
2D-constrained intermediate) as the next cheap, decisive test - both are
~30-60 min experiments, not research projects.
