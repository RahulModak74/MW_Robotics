# Experiment Summary — Nov 2, 2026 — where this is stuck, for the next agent

Written so an agent picking this up cold (ideally on a stronger model) can
get oriented in minutes, not hours. Read this before re-deriving anything
below — it's all been tested, not guessed.

## 1. What's solidly proven (don't re-verify, build on it)

- **CEM over a small linear policy, on CPU, no GPU, no backprop, works
  on Walker2d (6 joints, 2D-constrained — it physically cannot fall
  sideways).** `../mw_to_protomotions/cem_policy_search.py`, fitness =
  survival steps. Proven result: **920.9 steps held-out** (20 seeds, cap
  1000), vs. random's ~15-35 steps. See `PLAN_A_PASSED.md`.
- **Forward-progress fitness (not just survival) produces a real walking
  gait on the same body.** `run_demo_walker.sh` / `cem_walk_theta.npy`.
  Verified through the actual Flask backend (`walker_demo/`), properly
  multi-seeded (5 seeds, not one lucky trial): **1.34–2.60 m forward,
  600/600 steps survived, every seed.** The MW validator gauge is wired
  up and working end-to-end (real bug was a missing `sys.path` entry for
  `mw_to_protomotions/`, not a schema mismatch — fixed in
  `walker_demo/walker_app.py` and `walker_animation.py`).
  - **Known gap, not yet fixed:** the MW detector was trained only on
    the *balance* controller's data. The walk policy's real postures
    score mean MW-distance 85–239 against a calibrated threshold of
    67.77 — i.e. the gauge reads "THREAT" almost the whole time during a
    perfectly successful walk. Plumbing is fixed; calibration for the
    walk gait specifically is not.
- **Rank-4 head matches the full linear policy on Walker2d — the
  scaling-via-rank-reduction hypothesis is real, not just a hope.**
  `29DoF/rank4_walker.py`: a = tanh(U@(V@φ(s))+b), V∈R^{4×15},
  U∈R^{6×4}, 90 params (vs linear's 96). Result: **896.2 steps held-out
  — 97.3% of the linear policy's 920.9**, same budget (pop=80, gens=30,
  max_steps=400). This was `REALISTIC_CLAIMS_NEXTSTEPS_NOV2nd.md`'s own
  prescribed load-bearing test, and it passed cleanly.
- **Plan B (manifold-as-policy, SPEC_with_MW_3.md) is NOT the way
  forward — don't revisit it without a specific new idea.** The joint
  VAE over (s,a,s') has a real, measured confound: because all training
  data came from one fixed PD controller, `a` is close to a deterministic
  function of `s` alone, so apparent "inverse dynamics" success was
  partly (not entirely) memorization, not genuine causal use of the
  transition. See `../mw_g1_racer/confound_check.py` and its result. Not
  fully dead (there IS a real, smaller signal — correlation drops from
  0.94 to 0.58 when `s'` is shuffled, which it wouldn't if the model used
  zero information from `s'`), but not a productive direction right now.

## 2. Where this is stuck: 3D balance, not dimensionality

The working hypothesis entering today's session (from
`REALISTIC_CLAIMS_NEXTSTEPS_NOV2nd.md`) was: **the blocker to scaling
toward G1/29-DoF is the "dimensionality wall"** — too many parameters
for CEM's population to cover (96 → ~900 for a flat policy at 29 DoF),
fixed by a rank-4 head or waypoint lifting.

**That hypothesis is only half right.** Rank-4 passed on Walker2d. But
testing the next step up — `Humanoid-v5` (17 actuated joints, **true 3D**,
not 2D-constrained like Walker2d) — produced a result that doesn't fit
the dimensionality-wall story at all:

| Method | Body | Active DoF | Params | Held-out survival (cap 1000, 20 seeds) |
|---|---|---|---|---|
| Linear + CEM | Walker2d | 6 (2D) | 96 | **920.9** |
| Rank-4 + CEM | Walker2d | 6 (2D) | 90 | **896.2** |
| Flat-linear + CEM | Humanoid-v5 | 17 (3D) | 731 | 97.2 |
| Rank-4 + CEM, 30 gens | Humanoid-v5 | 17 (3D) | 253 | 99.0 |
| Rank-4 + CEM, 90 gens (3x budget) | Humanoid-v5 | 17 (3D) | 253 | 112.7 |
| Rank-4 + CEM, arms frozen | Humanoid-v5 | 11 (3D) | 223 | 102.0 |
| Hand-designed PD, no CEM at all | Humanoid-v5 | 17 (3D) | n/a | 58.1 |
| Random policy | Humanoid-v5 | 17 (3D) | n/a | 26.1 |

Every single Humanoid-v5 variant — regardless of parameter count (731
down to 223), rank reduction, 3x search budget, fewer active joints
(arms frozen), or even abandoning CEM entirely for a hand-tuned PD law —
clusters in the same narrow **58–113 step band**. Walker2d, by every
method tried, lands near **900+**. Tripling the CEM budget (30→90
generations) moved the needle by only 14% (99.0→112.7) and the training
curve visibly flattened (`std_norm` shrank to ~0.37, `pop_mean`
oscillating with no further real climb from generation ~77 onward) — a
genuine plateau, not an under-budgeted search.

**Conclusion so far:** the dominant variable is **2D-constrained vs.
true 3D balance**, not joint count, not policy architecture, not search
budget — at least not within anything tested today. This is a different,
more fundamental blocker than the one the roadmap docs were written
around, and it directly undermines the planned "12-DoF legs-only
intermediate" strategy: a real legs-only humanoid is still true 3D, so
it may hit the exact same plateau for the exact same reason, even with
far fewer DoF than 17 or 29.

## 3. A specific, likely-important clue not yet investigated

**`Humanoid-v5`'s joint list has NO ankle joints at all:**
`['abdomen_z','abdomen_y','abdomen_x', 'right_hip_x','right_hip_z','right_hip_y','right_knee', 'left_hip_x','left_hip_z','left_hip_y','left_knee', 'right_shoulder1','right_shoulder2','right_elbow', 'left_shoulder1','left_shoulder2','left_elbow']`
— hips, knees, abdomen, shoulders, elbows. **No ankle actuation.**

In real human/robot bipedal standing, the finest, cheapest, most
important correction for small perturbations is the **ankle strategy**
(small ankle torques, used constantly, below the threshold where hip
motion is even needed). Without any ankle joint, this MuJoCo body can
only ever use the **hip strategy** — larger, more disruptive torso-angle
corrections — for every disturbance, however small. This is a plausible,
mechanistic, MORE SPECIFIC explanation than "it's 3D" for why balance is
so much harder here, and it has **not been tested or ruled out**. It
would also explain why even a reasonable hand-tuned PD law (58.1) did
only modestly better than random (26.1) — there may be no good reactive
hip-only solution at this gain structure, independent of how it's found.

**This is the single highest-value next experiment:** find or build a
MuJoCo humanoid body that DOES have ankle actuation (several standard
Gymnasium/MuJoCo humanoid variants and most real humanoid robot XMLs,
including anything resembling the actual G1, have ankle pitch/roll
joints) and re-run the identical rank-4 + CEM recipe on it. If ankle
actuation closes most of the Walker2d-vs-Humanoid gap, the real fix for
29-DoF scaling is "make sure the body has ankles and use them," not
"figure out how to do hip-only 3D balance" — a completely different,
more tractable problem.

## 4. Other things that have NOT been tried, in rough priority order

1. **Confirm or rule out the ankle hypothesis (section 3).** Highest
   priority — cheap to test, potentially explains everything.
2. **Properly tune the hand-designed PD controller.** `29DoF/pd_humanoid.py`'s
   gains (`BAL_KP=8.0, BAL_KD=1.5`, per-joint `KP`/`KD` arrays) were a
   single, quick, UNTUNED first guess — not a gain search like
   `mw_to_protomotions/tune_balance.py` did properly for Walker2d. A real
   gain sweep might do much better than 58.1 and would be a fairer
   "is 3D balance solvable by classical control at all" baseline before
   concluding anything about CEM's limits specifically.
3. **Parallelize the CEM rollouts** (roadmap Gap 2 — `mjbatch`, or even
   just Python `multiprocessing` across cores). Today's budgets (pop=80,
   30-90 generations) took 345-1848 seconds serially on one core. A
   10-15x wall-clock speedup would make MUCH larger population/generation
   sweeps affordable, which hasn't been tried (only 30 vs 90 generations
   at the same pop=80 — population SIZE itself was never scaled up for
   Humanoid, unlike the search-budget sweep done for Plan B's gating
   experiment, which found large, real gains from scaling population
   specifically, not just generations).
4. **Check `Humanoid-v5`'s own termination/healthy-range conditions.**
   Not inspected today — if the termination z-range is unusually strict
   relative to what a stable-but-imperfect hip-only balance naturally
   looks like, that could be inflating the apparent difficulty
   independent of the control problem itself. Quick to check via
   `env.unwrapped.model` / the Gymnasium Humanoid-v5 source.
5. **Try a feature/architecture change beyond rank-4**: a small recurrent
   policy, or explicit integral/momentum terms (a reactive linear policy
   has no memory — real balance controllers often use a state observer
   or at least a filtered estimate of center-of-mass velocity, which a
   single-step feature vector doesn't give it).
6. **Revisit the Plan B manifold idea, but fix the actual confound**
   found (section 1's note): generate training transitions from VARIED
   actions (exploration noise added to a baseline controller), not one
   fixed PD law, so `a` stops being a near-deterministic function of `s`.
   Lower priority than items 1-3, but flagged so it isn't silently
   re-attempted with the same known flaw.

## 5. File map

- `29DoF/rank4_walker.py` — the passing load-bearing test (Walker2d, rank-4)
- `29DoF/cem_humanoid.py` (lives in `domain/`, not `29DoF/`) — flat-linear 17-DoF baseline
- `29DoF/rank4_humanoid.py` — rank-4 on Humanoid-v5, `--gens` arg, both 30 and 90-gen runs logged
- `29DoF/rank4_humanoid_legsonly.py` — arms-frozen (11 active DoF) variant
- `29DoF/pd_humanoid.py` — hand-designed PD baseline, untuned, first pass only
- `29DoF/PROGRESS_LOG.md` — full timestamped narrative of today's session, including the
  reasoning tension with `REALISTIC_CLAIMS_NEXTSTEPS_NOV2nd.md`'s literal "attempt 29 DoF now"
  instruction (resolved by following the doc's own prescribed order instead)
- `../mw_g1_racer/confound_check.py` and `PROGRESS_LOG.md` — the Plan B confound finding
- `walker_demo/` — the working Flask demo (balance + walk policies, MW gauge fixed)
- `CHANGES_NEEDED_AFTER_FIRST_RUN.md`, `EXPLANATION_OF_WALKER_FOR_AGENT.md`,
  `REALISTIC_CLAIMS_NEXTSTEPS_NOV2nd.md` — the three planning docs this session worked from

## 6. The honest one-paragraph status

CEM-over-small-policies, CPU-only, is proven on Walker2d (6-DoF, 2D) for
both balance and forward walking, and the rank-4 parameter-reduction
technique is proven to preserve that performance. The attempt to extend
this to Humanoid-v5 (17-DoF, 3D) — as a direct empirical stand-in for
scaling toward G1/29-DoF — plateaus hard, around 6-12% of Walker2d's
survival performance, regardless of parameter count, search budget, or
even whether CEM is used at all. The leading, untested hypothesis is that
this specific MuJoCo body has no ankle joints, forcing all balance
correction through a much harder hip-only strategy — not that 3D
balance is inherently unsolvable by this method. That's the next thing
to check, and it's cheap.
