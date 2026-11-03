# mw_to_protomotions

The first real, tested pass through the 7-stage plan (now archived at
`../../old_content/specs_and_plans/path_to_full_Robotics.md`; the current plan is `../../ROADMAP.md`) — from this
repo's earlier state towards ProtoMotions-scale robotics. Everything here uses MuJoCo's Walker2d (6 joints, 2 legs) as the coupled
body, plus one stretch test on the full Humanoid-v5 (17 joints, true 3D).

**Start here: [STATUS.md](STATUS.md)** — a "read this first" five-minute summary at the top,
then full numbers and honest diagnosis for every stage below it.

**If you want the hour-by-hour story, or need to pick this up after an interruption:**
[PROGRESS_LOG.md](PROGRESS_LOG.md) — timestamped notes written roughly every 30 minutes
while this ran, self-contained enough for a fresh agent to resume cold.

## The headline, in one paragraph

All 7 planned stages got a real, honestly-measured attempt — not every one succeeded, and
the write-up says so plainly where it didn't. Stage 1 (the validator mechanism on a coupled
body) and Stage 4 (richer training data helps generalization) both worked cleanly on the
first attempt. Stage 3 (does MW's path beat a naive one?) failed its first test, was
diagnosed, retested properly, and then showed a small real advantage. Stage 6 (sim2sim)
found a real weak spot and then mostly fixed it with the same richer-data principle that
worked for Stage 4. Stage 7 produced the strongest result of the whole session: MW's
validator gave a reliable early warning before every single simulated fall, across every
disturbance tested. An unplanned stretch test (the full 17-joint Humanoid-v5, much closer to
the real ProtoMotions target than Walker2d) confirmed the core mechanism scales, while
finding a new, specific weak point (false-alarm calibration) - which more training data then
cleanly fixed, confirmed in the same session, not left open.

## Files

```
STATUS.md                   read this first - results and honest diagnosis per stage
PROGRESS_LOG.md               timestamped working notes, for resuming cold

classical_controller.py       Stage 2, attempt 1: hand-authored gait + PD (fell fast)
tune_balance.py                Stage 2: PD gain search for standing balance (best: 2.1s)
lqr_balance.py                 Stage 2, attempt 2: LQR (real numerical snag, documented)
test_controller.py             quick manual test harness for the above

datagen.py                     Stage 1 data: real trajectories from the tuned PD controller
walker_domain.py                Stage 1: the WalkerPostureVAE MW domain (soft priors)
stage1_validator_test.py        Stage 1 test: validator on a coupled body

stage3_reference_test.py        Stage 3, closed-loop test (no advantage shown)
stage3_openloop_retest.py       Stage 3, open-loop retest (small advantage shown)

stage4_richer_datagen.py        Stage 4: richer/varied-physics data generator
stage4_compare.py               Stage 4 test: single-config vs richer training data
stage4_dose_response.py         Stage 4 follow-up: a 5-point diversity dose-response curve

stage6_sim2sim.py                Stage 6 test: generalisation across solver settings
stage6_fix_attempt.py            Stage 6 follow-up: train across multiple settings instead

stage7_robustness.py            Stage 7: robustness curve + MW-as-safety-monitor test
stage7_lead_time_sweep.py        Stage 7 follow-up: does the early warning hold up broadly?

stretch_humanoid.py              unplanned stretch test: does Stage 1 scale to 17 joints, 3D?
stretch_humanoid_fix.py          follow-up: more training data fixes the false-alarm gap

*_full.log, *_full.log          saved stdout from each full (200-epoch) run, for the record
```

## What this does NOT claim

- **No real hardware was used or will be** - there is none in this environment, and real-
  hardware actions wouldn't be attempted autonomously here regardless (see STATUS.md).
- **Stage 5 (retarget to a real robot) was not attempted** - no target robot was specified
  for this task. Deferred, not faked.
- **The humanoid stretch test omits orientation and velocity features** - height + joint
  angles only, a stated simplification, not a claim of a complete 3D posture check.
- **Nothing here is a finished controller or a safety system.** Same scope note as the rest
  of this repo.
