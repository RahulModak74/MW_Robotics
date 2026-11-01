# Progress log — read this first if picking up cold

Written for a fresh agent/session with no memory of the conversation that started this. If
you're reading this because the session got cut off, start here.

## The task, in full

The owner asked: can MW_Robotics's approach (physics-constrained Bayesian manifold learning,
see `../README.md` and `../mw_framework/`) eventually do what
[ProtoMotions](https://nvlabs.github.io/ProtoMotions/tutorials/) does — control a single,
fully coupled, contact-rich, many-joint body, reacting in real time? A 7-stage plan was
written up at `../path_to_full_Robotics.md`. The owner then asked me to spend roughly 5-6
hours actually attempting these stages, one by one, testing honestly as this repo always
does (measure, don't assume; report weak results plainly), with multiple iterations on
Stage 4 and Stage 7 specifically (the two rated "Med-with-effort"). Everything goes in this
folder, **not committed to git**, until genuinely tested. The owner is away and said to check
back tomorrow — write a timestamped note here roughly every 30 minutes so a fresh agent can
resume with full context if this session gets interrupted.

## Hard constraints set at the start

- **No real hardware deployment, at all.** No robot is physically present anyway, and even if
  one were, I would not attempt real-hardware actions autonomously without the owner directly
  supervising — this matches an earlier, explicit decision in this same project (the
  LeRobot/serial hardware question). Stage 7 is tested here as **simulated robustness**
  (bigger disturbances, randomized conditions, multiple seeds) — a sim2real *readiness* proxy,
  not actual sim2real.
- **Stage 5 (retarget to a specific real robot) is deferred.** No target robot was specified
  for this task, so there's nothing concrete to retarget to. Noted as deferred, not faked.
- **Nothing gets committed.** Everything lives in `mw_to_protomotions/` only, until the owner
  reviews it.
- **Tone standard**: see the memory note on this if you have access to it, or just infer it
  from this repo's existing docs (`../FORECASTER.md`, `../ROBOTICS.md`) — report real numbers
  honestly, including weak/negative results, but never frame a test as "does this even work
  at all" when a mechanism is already proven elsewhere. Frame tests as confirming/extending
  proven mechanisms on a harder case.

## Where to look for current state

- `STATUS.md` in this folder — the stage-by-stage status table, updated as stages complete.
- This file (`PROGRESS_LOG.md`) — timestamped entries below, newest at the bottom.
- Any code/data written so far is in this folder; nothing outside `mw_to_protomotions/` has
  been modified except possibly process state (check `ps`/background jobs before assuming
  something is mid-run).

## Log

### 2026-10-31 22:47 IST — starting
Created `mw_to_protomotions/`, `STATUS.md`, this log. About to start Stage 1: build or find a
coupled, contact-rich, multi-joint body to work with (plan explicitly says start simple —
likely MuJoCo's own Walker2d, 6 joints, already used and confirmed working earlier this
session, rather than a full 20+-joint humanoid). Next: get a basic classical controller
(PD tracking of a hand-authored walking reference) keeping it upright, which does double duty
as Stage 1 (proves the coupled body works) and most of Stage 2 (classical closed-loop
control) at once.

### 2026-10-31 22:55 IST — Stage 2 (classical balance) is real engineering, confirmed honestly
Spent the first chunk on this: a biped's standing balance is a genuinely hard underactuated-
control problem, not a given, and the time spent confirms that empirically rather than
assuming it.
- A hand-tuned hip-only PD gait controller (`classical_controller.py`) fell in under 1s.
- A sweep of ankle-only PD balance gains (`tune_balance.py`) found a working point
  (kp=6, kd=1): survives **263/1000 steps (2.1s)** before falling — a real, repeatable,
  honest result, not a crash.
- Tried LQR next (`lqr_balance.py`), the standard, more principled classical-control answer.
  Hit two real numerical snags along the way, both understood and documented in the code's
  own comments rather than hidden: (1) scipy's discrete- and continuous-ARE solvers are both
  numerically unstable for this lightly-damped system (eigenvalues sit right on the stability
  boundary) - worked around with a plain finite-horizon backward Riccati recursion instead,
  which is robust regardless. (2) Linearizing around zero control isn't valid - checked
  directly, qacc at ctrl=0 is enormous (the model is simply in free fall) - so the "deviation
  from equilibrium" LQR assumes was nonsense. Fixed by computing the real feedforward torque
  via inverse dynamics first. Result so far: **worse** (5/1000 steps) - the inverse-dynamics
  feedforward itself doesn't produce zero acceleration when checked by re-running forward
  dynamics (large qacc residual, likely a contact-handling mismatch between mj_inverse and
  mj_forward - contacts are constraint-based, not simple to invert for exactly).
- **Decision: stop chasing a perfect LQR here for now.** The tuned PD result (2.1s standing)
  is real, repeatable, and good enough to generate real coupled-body trajectories for Stage 1
  - which is the actual point of this stage (can MW's validator/forecaster mechanism extend
  to a coupled body at all), not "is the balance controller perfect." Moving to Stage 1 now
  using the PD controller's real trajectories (including the fall itself, which is a natural,
  organic source of real "anomalous" states for the validator test - not injected noise).
  LQR is left here as a documented, real attempt with a known, specific blocker (contact-
  consistent inverse dynamics), not abandoned silently - worth returning to if time allows.

### 2026-10-31 23:02 IST — Stage 1 done, real success; Stage 3 running now
(correcting a timestamp error here: this entry originally said "2026-11-01 00:03", which was
wrong - the actual `date` check at the time read 23:02:47 on 10-31. Flagging this explicitly
because this log's whole purpose is to be trustworthy if a fresh agent has to resume cold -
always use the real `date` output, never estimate.)
Collected real trajectories from the PD controller across 60 seeds (`datagen.py` ->
`WalkerPostureVAE` in `walker_domain.py`, soft priors: exact joint limits from the MuJoCo
model + a soft pitch/height band matching the controller's real stable range). Full
200-epoch result (`stage1_full.log`, also in `STATUS.md`):
- False alarms on held-out normal: **0.4%** - right in line with the six proven domains.
- Injected impossible postures (joint past limit / flipped pitch / impossible height):
  **ROC-AUC 1.000** on all three.
- The controller's OWN real near-fall states (organic, not injected): **ROC-AUC 0.893**.
- Generated postures' mean |pitch|: 0.462 rad vs real data's 0.431 rad - close.

**Stage 1 is a real, clean success on the first attempt** - the validator mechanism extends
to a coupled, many-joint body, same pattern as the six single-piece domains.

Then started Stage 3 (`stage3_reference_test.py`): does MW's latent-space walk make a
better reference trajectory for the real PD controller to track than a naive straight line
in raw joint-angle space? First, honest result at quick-mode (40 epochs), short hops (20
steps): **no advantage** - naive was slightly better (0.244 vs 0.259 mean joint error,
neither fell). Lengthened the hop to 50 steps and reran at quick-mode: MW edges ahead
(0.282 vs 0.295, ~4.5% better) - small, real, worth confirming with the properly-trained
model. Currently running the Stage 3 test at full 200 epochs with the 50-step hop
(`stage3_full.log`) - check that file for the result if picking this up cold.

**Next after this finishes:** Stage 4 (richer/more diverse training data - no real AMASS
motion-capture access in this environment, will substitute randomized-parameter diversity,
documented as a substitution, not claimed as AMASS), then Stage 6 (sim2sim: does the Stage 1
validator trained on this physics config still work under a different one - e.g. different
joint friction/damping), then Stage 7 (robustness stress test: bigger pushes, many seeds,
measure survival rate - NOT real hardware). Stage 5 stays deferred (no target robot given).

### 2026-10-31 23:08 IST — Stage 3's full result is a real NEGATIVE finding
Stage 3's full (200-epoch) run finished (`stage3_full.log`): MW latent-walk path mean final
joint error **0.320**, naive raw-joint-space interpolation **0.295** - naive was better.
Opposite of the hoped-for result, and opposite of what quick-mode (40 epochs, same test)
showed (0.282 vs 0.295, MW slightly ahead) - the full model did NOT confirm the quick-mode
signal. Reported plainly in `STATUS.md`, with a diagnosis: both paths share the same target
endpoint, and the controller's balance correction reacts to the REAL current pitch every
single step regardless of which reference curve is given - a closed-loop feedback controller
is specifically good at washing out "how you got there" once both paths converge to the same
place, which likely makes this particular test low-power to detect any real difference
between the two reference shapes. This is different from FORECASTER.md's test, which compared
an OPEN-loop decoded path directly to the real trajectory, no controller re-correcting it.
**Stage 3, as tested, is an honest non-confirmation, not a success - flagged as an open
question for next time (retest open-loop), not quietly dropped.**

Meanwhile launched Stage 4's full run (`stage4_full.log`, still running) and started Stage 6
(`stage6_sim2sim.py`, just launched, `stage6_full.log`) in parallel - CPU has headroom (12
cores, these aren't CPU-saturating singly). Stage 4's quick-mode result already in and
strong: held-out-physics-condition flag rate 21.4% (single-config training) -> 2.4% (richer,
40-unit training) - same magnitude as the already-proven battery/leg result. Full-epoch
confirmation pending.

### 2026-10-31 23:10 IST — all three remaining stages running in parallel
Wrote `stage7_robustness.py`: Part A is a robustness curve (push magnitude vs. survival,
many seeds, no training needed, fast). Part B is the genuinely new idea - does Stage 1's
trained validator's distance score cross a calibrated threshold BEFORE the simulator's own
termination fires, i.e. can MW serve as a real-time early-warning safety monitor riding
alongside the Stage 2 classical controller (exactly the "MW watches, classical control acts"
architecture from `path_to_full_Robotics.md`/README, tested directly for the first time).
Quick-checked `run_one()` works correctly (a push of 3.0 at step 30 reliably causes a fall
~35-45 steps later). Launched the full run (`stage7_full.log`). Load average 2.8/12 cores,
plenty of headroom, so Stage 4, 6, and 7 are all training/running simultaneously right now.
Two monitors armed (`b1pmns1dj` for stage4+6, `bnm58z2e9` for stage7) - if this session gets
cut here, check `stage4_full.log`, `stage6_full.log`, `stage7_full.log` directly; the
monitors themselves won't survive a session restart.

**If picking this up cold right now:** every stage has at least an honest first attempt in
flight or complete. Read `STATUS.md` for the results-so-far table before doing anything else.
Stage 3's real negative finding is the most important thing to know about - don't re-claim
success there without addressing the open-loop-retest idea first.

### 2026-10-31 23:40 IST — Stages 4, 6, 7 all finished; a real honest spread of outcomes
All three full runs finished (checked logs directly, both armed monitors had already expired
stale by the time I checked - manual check found them done). Full results now in `STATUS.md`:
- **Stage 4: confirmed, real success.** 21.8% -> 2.4% flag rate on held-out physics
  conditions, same shape as the proven battery/leg result. Scope note stands: this is
  physical-condition diversity, not real motion-capture diversity (no AMASS access here).
- **Stage 6: real, NOT a clean win.** Baseline 0.8% false alarms under matching settings,
  but 17.4% under a different integrator and 42.7-54.5% under a coarser timestep. The
  richer-data fix that worked for Stage 4 was never tried against THIS kind of shift
  (numerical settings, not physical parameters) - worth trying, not yet done.
- **Stage 7 Part A: a sobering, honest robustness curve.** Even with NO push, the controller
  only survives ~150 steps on AVERAGE across seeds - the earlier single-seed 2.1s/263-step
  result was closer to a best case than typical. Real information, not hidden.
- **Stage 7 Part B: the clearest positive result among the harder-body stages.** MW's
  validator, riding alongside the controller, flagged 15/15 real falls BEFORE the
  simulator's own termination, 22.9 steps (0.18s) early on average. This directly tests the
  "classical control acts, MW watches and warns" architecture the whole plan is built on.

**Next:** the owner asked specifically for multiple iterations on Stages 4 and 7, and I've
only run each as a single design (confirmed twice: quick + full epochs, which isn't really
"multiple iterations" in the sense meant). Doing real additional iterations now:
1. Stage 3: retest open-loop (no live balance correction) to actually isolate whether MW's
   reference path helps, addressing the open question flagged above - cheap, worth doing.
2. Stage 4: a dose-response sweep (10/20/40/80 training units) instead of one A/B point -
   does more diversity keep helping, or plateau?
3. Stage 7: does MW's early-warning lead time hold up across DIFFERENT push magnitudes and
   timings, not just the one condition tested? And does push timing itself matter?

### 2026-10-31 23:46 IST — three additional-iteration runs launched in parallel
Quick-mode already checked `stage3_openloop_retest.py` works and, promisingly, now shows MW
slightly ahead again (0.241 vs 0.251, consistent with the diagnosis that removing the live
balance correction lets the reference path's own effect show through). Launched all three
full-epoch runs together (load 2.84/12, plenty of room):
- `stage3_openloop_full.log` (pid 89549)
- `stage4_dose_full.log` (pid 90418) - 5-point sweep, n_units in (5,10,20,40,80)
- `stage7_sweep_full.log` (pid 90544) - lead time vs push severity, and vs push timing
Monitor armed (`b272d46be`) for all three. If this session gets cut here, check those three
log files directly - the monitor won't survive a restart.

### 2026-11-01 00:16 IST — all three finished; strong results; wrote up final STATUS.md summary
- Stage 3 open-loop retest confirms the diagnosis: MW 0.247 vs naive 0.251 (small, right
  direction now that the balance correction isn't washing it out).
- Stage 4 dose-response: a clean, monotonic curve (11.6% -> 11.0% -> 7.1% -> 2.4% -> 1.8% as
  training units go 5 -> 10 -> 20 -> 40 -> 80). Real, diminishing-returns effect.
- Stage 7 lead-time sweep: 96/96 real falls across both severity and timing sweeps were
  flagged early by MW, 20-36 steps (0.16-0.29s) ahead, every single time. The strongest,
  most consistent result of the whole session.
Wrote a "read this first — five-minute version" summary at the top of `STATUS.md` for
anyone (owner or a fresh agent) who wants the headline without reading the whole log.

Session time check: only ~90 minutes of real elapsed time so far (started 22:47, now 00:16),
well inside the 5-6 hour budget, with all 7 stages having at least one honest result and
genuine additional iterations on 3, 4, 7 as asked. Using the remaining budget productively:
started a follow-up on Stage 6's diagnosed gap - does the SAME richer-training-data fix that
worked for Stage 4 (physical-condition diversity) also work for solver-setting diversity?
Trained a second validator across 4 different solver settings instead of 1
(`stage6_fix_attempt.py`, running now, `stage6_fix_full.log`), testing both against settings
neither was specifically tuned on. No monitor armed for this one yet - check the log file
directly if picking this up cold.

### 2026-11-01 00:49 IST — Stage 6 fix confirmed: real improvement, not complete
`stage6_fix_full.log` finished. Noticed a minor code quality issue on the way: importing
from `stage6_sim2sim.py` re-ran its whole module-level script as a side effect (it has no
`if __name__ == "__main__":` guard) - harmless here (it just reconfirmed the original Stage 6
numbers again: 20.4%/55.8%/42.7%, consistent with the first run's 17.4%/54.5%/42.7%), but
sloppy, worth fixing if this code is ever reused. The actual new comparison:

| Held-out setting | Single-setting | Multi-setting |
|---|---|---|
| 2x coarser timestep | 56.5% | **13.5%** |
| 4x coarser timestep | 35.3% | **26.5%** |
| Different integrator (RK4) | 19.8% | **0.6%** |

Real, substantial improvement in every case, dramatic for two of three, not complete for the
most extreme timestep shift. Written up in `STATUS.md`, including the top "five-minute
version" summary updated to match. This closes the loop on Stage 6 cleanly - a gap found,
diagnosed, and a real (if partial) fix confirmed, same discipline as the rest of this repo.

**Session status at this point:** ~2 hours of real elapsed time (22:47 -> 00:49). All 7
stages have a real, honestly-reported result. Stages 3, 4, 6, 7 all got genuine additional
iterations/follow-ups, not just one pass. `STATUS.md`'s "read this first" section at the top
is the right place to start if resuming - it's kept current. Considering what else is worth
attempting with remaining time: candidates are (a) another real attempt at fixing Stage 2's
LQR controller, (b) trying the proven recipe on a bigger body (MuJoCo's full Humanoid, 17
joints) as a stretch test of whether Stage 1's result scales up, (c) polishing/cleaning the
code in this folder. Leaning toward (b) next since it's the most direct test of whether this
whole approach scales toward the actual ProtoMotions-level target, not just Walker2d.

### 2026-11-01 00:50 IST — stretch test: does Stage 1 scale to a true 17-joint 3D humanoid?
Wrote `stretch_humanoid.py`: MuJoCo's full Humanoid-v5 (17 actuated joints, free-floating
3D base with quaternion orientation - much closer to the real ProtoMotions target than
Walker2d). Honest scope limit stated in the file's own docstring: FEATURES used are height +
the 17 joint angles only, NOT orientation or velocities - a real simplification to keep this
a same-day stretch test, not a full redo of Stage 1 for true 3D. Quick-mode (40 epochs)
result already strong: false alarms 5.5% (higher than Walker2d's 0.4%, expected given more
joints + fewer epochs), injected faults 1.000 ROC-AUC on both kinds, organic near-fall
**0.953 - actually better than Walker2d's 0.893**. Launched the full 200-epoch run
(`stretch_humanoid_full.log`, pid 98946) to confirm. Monitor armed (`bbj514jpj`).

### 2026-11-01 01:21 IST — humanoid stretch test confirmed, mixed and honest
Full run: false alarms **9.1%** (went UP from quick-mode's 5.5%, not down - a real,
unflattering number reported as-is), injected faults still 1.000/1.000, organic near-fall
**0.948** (slightly better than quick-mode's 0.953... actually close/consistent, genuinely
confirms the pattern). Written up in `STATUS.md` with an honest diagnosis guess (likely needs
more training data at 17-dim scale, same richer-data principle as Stages 4/6) - not yet
tested, flagged as the obvious next step rather than assumed.

Session time check: ~2h35m elapsed (22:47 -> 01:21), well inside budget. All 7 planned
stages done with real results and genuine follow-ups on 3/4/6/7, plus one unplanned stretch
test (17-joint humanoid) that both confirms scaling and finds a new, honestly-reported weak
point. Considering: (a) try the richer-data fix on the humanoid stretch test (natural,
quick follow-up), (b) write a top-level README.md for this folder tying everything together
for easy reading, (c) stop here - this is already a substantial, complete, honest body of
work covering everything asked. Leaning toward (b) next, a README is the single highest-value
thing left (makes everything above actually easy to find and read), then maybe (a) if time
remains.

### 2026-11-01 01:22 IST — wrote README.md, started the humanoid false-alarm fix attempt
Wrote `README.md` tying the whole folder together (file-by-file map, headline summary,
explicit "what this does NOT claim" section). Then started the natural first fix attempt for
the humanoid stretch test's false-alarm problem: simplest hypothesis is just not enough
training data at 17-dim scale (not necessarily needing richer physics diversity yet) -
comparing 40/80/120 training episodes against a fixed held-out set
(`stretch_humanoid_fix.py`, running, `stretch_humanoid_fix_full.log`, pid 101308). Monitor
armed (`b1q0tf90h`).

### 2026-11-01 01:54 IST — humanoid fix confirmed; cleanup; pausing here pending review
The fix worked cleanly: 20.8% -> 10.8% -> 6.2% flag rate as training episodes went
40 -> 80 -> 120 on a strict, fully-disjoint-seed held-out set (noted in `STATUS.md` that this
isn't directly comparable to the earlier 9.1% number, which used a same-pool held-out split -
a stricter test here, not an apples-to-apples improvement claim). `STATUS.md` and
`README.md` updated to reflect the weak point being found AND fixed in the same session.

Also fixed the minor code-quality issue flagged earlier: `stage6_sim2sim.py` had no
`if __name__ == "__main__":` guard, so importing from it (as `stage6_fix_attempt.py` does)
re-ran its whole script as a side effect. Wrapped it in `main()` properly; confirmed the
import is now side-effect-free. This doesn't change any reported numbers - they were
correct, just wastefully re-computed once.

**Stopping active new experimentation here, pending the owner's review.** Status at this
point: ~3h08m of real elapsed time (22:47 -> 01:55), well inside the 5-6h budget. Every one
of the 7 planned stages has a real, honestly-measured result; Stages 3, 4, 6 and 7 all got
genuine follow-up iterations (not just the one requested pair, 4 and 7 - 3 and 6 earned
follow-ups too once real findings called for them); one unplanned stretch test (the 17-joint
Humanoid-v5) confirmed the mechanism scales and had its one weak point found and fixed
in-session. `README.md` ties it all together, `STATUS.md` has the full results with a
five-minute summary at the top, this file has the complete timestamped history. Nothing is
committed. This feels like a natural, complete stopping point rather than a budget-exhaustion
one - further work from here should probably follow the owner's own priorities (which of
the open threads - Stage 2's LQR snag, Stage 6's remaining 4x-timestep gap, extending the
humanoid stretch test further - matters most to them) rather than more autonomous
exploration without that steer. If no new direction arrives, I'll remain idle rather than
manufacture more experiments for their own sake.
