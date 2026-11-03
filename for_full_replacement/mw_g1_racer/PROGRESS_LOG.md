# Progress log — Plan B (SPEC_with_MW_3.md), live session

Self-contained notes, updated roughly every 30 minutes, so a fresh agent (or
the owner) can pick this up cold if the session is interrupted.

## Context (read this first)

The owner asked: read `../SPEC_with_MW_3.md` (Plan B - "manifold as policy,
no learned controller"), then run a full training attempt, reporting live
in chat as it goes (owner is actively watching, not away), saving progress
notes here every ~30 minutes. The stated goal: imitate the SPIRIT of
NVIDIA's g1-racer demo (`~/ProtoMotions/demos/g1-racer` - 6 full G1 humanoids
racing, driven by one 403MB GPU-trained policy) using Plan B's architecture
instead - same "disruption" the owner already applied to MW batteries
(manifold replaces the learned model, not just augments it).

**Honest scope decisions, stated up front, not hidden:**
- Using Walker2d (6 joints), not the real G1 (~30 joints) - no G1 MuJoCo
  model is unpacked locally (the demo's `game.zip`/meshes require a large
  Hugging Face download this session won't trigger unprompted). "Imitate
  g1-racer" here means the ARCHITECTURE (manifold stands in for the GPU-
  trained policy), not the literal robot body.
- Spec section 8's gating experiment calls for "1M Walker2d transitions,
  200 epochs". Measured cost on this machine: ~2s/epoch per ~9k rows (from
  earlier Stage-1 fits), so 1M rows x 200 epochs extrapolates to roughly
  13 hours on one CPU core - not done today. Scaled down to a dataset and
  epoch count that fits one real sitting; exact numbers logged below as
  they're decided, not promised in advance.
- Plan B's own spec (section 8) says: run the cheap gating experiment
  FIRST, before writing the runtime (B1/B2). That's the priority order
  here too - get a real viable/not-viable number before building more.
- `mw_framework/domains/` (the shared library) is left untouched -
  `TransitionVAE` lives in this experiment folder
  (`transition_domain.py`), same precedent as `walker_domain.py` in
  `mw_to_protomotions/` - unvalidated domains stay with the experiment,
  not the shared library, until proven.
- Not committed to git unless/until asked, same practice as the rest of
  this repo's experiment folders.

## 2026-11-02 02:05 IST - setup

- Created this folder, read the full spec.
- Built `transition_domain.py`: `TransitionVAE(MWVAE)` over 24 features
  (9 state + 6 action + 9 next-state). One deliberate simplification vs
  spec section 2.3: a flat joint VAE, not the full fibre-bundle
  (`ManifoldAlgebra`) construction - the section 8 gating experiment only
  needs `mw_distance()` on the concatenated row, which a flat VAE gives
  directly. Fibre-bundle refinement deferred unless gating says this
  direction is worth it.
- Built `transition_datagen.py`: same richer-datagen recipe as
  `mw_to_protomotions/stage4_richer_datagen.py` (randomized friction/
  damping per "unit"), logging every `(s, a, s')` step, with prior
  filters from spec section 3.1 (joint limits, ground penetration,
  one-step reachability).
- **Real bug found and fixed in the filter, within the first smoke test**:
  the reachability filter was applied to ALL state features including
  `pitch_rate` - but `pitch_rate` is a velocity, not a position, and
  velocities legitimately jump fast during ground contact. Applying a
  "can't teleport" filter meant for positions to a velocity rejected 95%
  of normal rows (4.6% kept). Fixed by excluding `pitch_rate` from that
  specific filter (positions only) - 93.4% kept after the fix. The
  domain's own `physics()` C4 constraint already excluded it correctly;
  only the datagen-side filter had the bug.
- Smoke test: 500 rows in 0.6s with a persistent env (model params mutated
  per "unit" instead of recreating `gym.make()` each time - the efficiency
  lesson learned in `mw_to_protomotions`'s earlier CEM work).

## 2026-11-02 02:20 IST - dataset generated, training launched

- Generated the real training set: 2000 randomized-physics units x up to
  400 steps each, filtered -> **300,356 transition rows**
  (`transitions_train.parquet`), in 115s.
- Generated a disjoint held-out set (seeds 50000+, never touched by
  training): **11,837 rows** (`transitions_heldout.parquet`), for the
  section 8 gating experiment later.
- Timed a small subsample (20k rows, 3 epochs, batch=512): 6.02s/epoch.
  Linear extrapolation to the full 300k-row set suggested ~90s/epoch,
  meaning the spec's literal "200 epochs" would be ~5 hours - more than
  one sitting. Decided to train in checkpointed CHUNKS instead of
  committing to a total epoch count up front: each chunk calls
  `MWDetector.fit()` again, passing in the PREVIOUS chunk's already-
  trained VAE as the starting point (confirmed in `detector.py` - `fit()`
  deep-copies whatever `vae=` it's given, so this is real incremental
  training, continuing from learned weights, not a restart each time).
  This also naturally gives checkpointed saves close to the requested
  30-minute cadence, instead of an all-or-nothing multi-hour run.
- Launched chunk 1 (15 epochs) at 02:20 IST. Running in the background;
  will report real measured time (not the extrapolated guess) and decide
  the remaining chunk count/size from that actual number.

## 2026-11-02 02:50 IST - chunk 1 done, early validation, real gating result

- Chunk 1 (15 epochs, 300k rows, batch=512) finished: **69.73s/epoch real
  measured** (close to the 90s/epoch extrapolate). Checkpoint saved to
  `transition_mw_walker.ckpt`.
- Ran spec section 4.3's validation metrics on this 15-epoch checkpoint
  before committing to more training time:
  - V1 reconstruction floor: held-out/train ratio **0.98x** (pass, < 1.5x)
  - V2 injected faults: joint_limit, teleport, action_saturation all
    **AUC = 1.000** (pass, > 0.90) - consistent with every other domain
    in this repo; injected/obvious faults are reliably easy to catch.
  - These pass cleanly even at 15 epochs, so they weren't the deciding
    question - moved straight to the real decisive test rather than
    spending another hour of training first on a question already
    answered.
- **Ran the section 8 gating experiment - the actual decisive test.**
  Implemented `inverse_query` as a 1-step CEM over actions only (pop=32,
  3 generations), vectorized across both the 5000 held-out rows AND the
  population in one batched `mw_distance()` call per generation (ran in
  60s for the full 5000 rows). For each held-out `(s, s')` pair, searched
  for `a* = argmin_a d_M(s, a, s')` and compared to the real `a`:
  - mean `|a*-a|` = **0.626** (spec threshold for "viable": < 0.20)
  - std `|a*-a|` = **0.320** (spec threshold: < 0.40 - PASSES)
  - **Spec's own decision rule: "SYSTEMATICALLY BIASED but recoverable -
    proceed cautiously."** Not a clean pass (mean error is ~3x over the
    viable threshold), not an ill-posed failure either (std is within
    bounds - the search isn't unstable/random, it's consistently off in
    some direction, which per the spec is the recoverable case, not the
    "Plan B won't work" case).
- **Next, now in progress:** this result is from only 15 epochs. Before
  concluding anything about Plan B's viability, testing the most likely
  explanation first - under-trained manifold, not a fundamental
  ill-posedness - by continuing training (chunk 2, ~25 more epochs) and
  re-running the IDENTICAL gating experiment, to see whether the bias
  shrinks with more training. This is a real before/after test, not a
  guess.
- Chunk 2 (25 epochs) launched in the background.

## 2026-11-02 ~02:58 IST - owner flagged a new Plan A result, raises the V10 bar

- Owner shared `~/MW_Robotics/domain/PLAN_A_PASSED.md` - a separate,
  larger-budget run of the Plan A balance CEM (pop=80, 30 generations,
  max_steps=400, vs this session's earlier pop=60/gens=15-18 runs):
  **920.9 steps held-out mean** (20 seeds), vastly beating both the PD
  baseline (263 steps) and this folder's earlier Plan A reference
  (208.3 steps). Saved theta: `~/MW_Robotics/cem_best_theta.npy`.
- This directly matters for Plan B's own spec (section 7.1, test V10:
  "B1 vs Plan A... B1 within 20% of the CEM baseline"). The bar Plan B's
  eventual B1 runtime has to clear is now **920.9 steps**, not 208.3 -
  a much harder target. Noted for when B1 is actually built and tested;
  not re-run here, just recorded as the number to compare against later.

## 2026-11-02 ~03:05 IST - chunk 2 done (40 epochs total), bias diagnosed, two fixes found

- Chunk 2 (25 epochs) finished: **61.11s/epoch** (slightly faster than
  chunk 1, likely less CPU contention near the end). Total epochs: 40.
  Loss/sample dropped from 249.8 (epoch 1 overall) to 47.2 (epoch 16
  overall) - real, substantial learning, not a plateau.
- **While chunk 2 trained, ran two fast diagnostics on the 15-epoch
  checkpoint** (owner's suggestion: the inverse-query SEARCH itself is
  cheap - seconds, not minutes - so it's worth experimenting there before
  assuming more training is the only lever):
  - **H1 (search budget):** mean action-recovery error shrank from 0.629
    (pop=32, 3 gens - the original gating run) to 0.484 (pop=64, 6 gens)
    to **0.392** (pop=128, 10 gens), still falling, not plateaued. A real
    chunk of the earlier "biased" result was simply an under-optimized
    1-step CEM, not a manifold limitation - free to fix, no retraining.
  - **H2 (per-dimension structure):** the blended error is NOT uniform
    across the 6 action dims. Hip/knee torques (thigh, leg, both legs)
    recover with correlation **0.96-0.99** to the true action - the
    manifold clearly knows these well, with a mild VAE regression-to-
    mean shrinkage (recovered std ~0.75 vs true ~0.97). Ankle/foot
    torques recover much worse (correlation ~0.59, recovered variance
    less than half true variance) - a real, specific, physically
    plausible weak spot (foot torque has a smaller joint range and
    weaker leverage on the resulting transition, genuinely harder to
    infer backward from (s,s') alone), not a uniform failure.
## 2026-11-02 ~03:15 IST - combined result: real progress, one hypothesis corrected

- Ran the full 5000-row gating experiment on the 40-epoch checkpoint with
  pop=128/gens=10 search (combining both fixes): **mean |a*-a| = 0.299**
  (down from 0.626), std = 0.256 (still comfortably under the 0.40
  threshold). Still above the 0.20 "viable" line (1.5x over, vs 3.1x
  before), but closing fast and monotonically with both training and
  search budget - took 12 minutes (pop=128/gens=10, N=5000, uncontended).
- **Per-dimension breakdown corrects the earlier diagnosis.** At 15
  epochs, foot/ankle torques looked like a genuine, physically-plausible
  structural weak spot (correlation ~0.59 vs hip/knee's ~0.96-0.99). At
  40 epochs, foot/ankle correlation is now **0.886-0.892** - nearly
  matching hip/knee (0.886-0.990 across ALL six dimensions, much more
  uniform). That was under-training, not a fundamental limit - hip/knee
  just happened to converge faster. Worth recording as a corrected
  hypothesis, not quietly dropped: the honest conclusion changed with
  more evidence, and that's the right response to it.
- All six action dimensions are now in a tight, uniform mean-error band
  (0.21-0.35), rather than two bad dimensions dragging a mixed average.
  That's consistent with "more of the same (training + search)" being
  the right next lever, not a different architecture change.
- **Next:** launch chunk 3 (another ~25 epochs) to see if the clean
  trend (0.626 -> 0.299, correlation 0.59 -> 0.89 on the weakest dim)
  continues toward the 0.20 threshold, or starts to plateau.

## 2026-11-02 ~03:43 IST - chunk 3 done (65 epochs total)

- Chunk 3 (25 epochs) finished: 60.50s/epoch, consistent with chunks 1-2.
  Total epochs: 65. Loss/sample: 35.3 (epoch 41 overall), down from 47.2
  (epoch 16) and 249.8 (epoch 1) - still falling, but the relative drop
  is smaller each chunk (249.8->47.2 vs 47.2->35.3), the expected shape
  of SVI convergence slowing down, not a red flag by itself.
- Re-running the gating experiment (pop=128, gens=10, N=5000) on this
  65-epoch checkpoint now, to see whether the 0.626 -> 0.299 trend
  continues toward 0.20 or has started to plateau.

## 2026-11-02 ~03:57 IST - training has plateaued; search budget is the live lever

- Gating result at 65 epochs (same pop=128/gens=10 search as the 40-epoch
  run): mean |a*-a| = **0.314** - essentially flat vs 0.299 at 40 epochs,
  even slightly worse (within run-to-run noise). 25 more epochs of
  training bought nothing measurable.
- **This cleanly separates the two levers that were combined in the
  0.626 -> 0.299 jump.** Search budget went 32x3 -> 128x10 at the same
  time training went 15 -> 40 epochs, so that result alone couldn't say
  which lever did the work. Now, with search budget HELD FIXED (128x10)
  across the 40-epoch and 65-epoch checkpoints, training alone produced
  no further gain - so the earlier jump was mostly the search-budget
  increase, not the extra training. The standalone search-budget
  diagnostic (H1, logged above) also directly supports this: at N=500
  it was still falling (not plateaued) at pop=128 (0.629 -> 0.484 ->
  0.392), unlike training, which has now visibly flattened.
- **Decision: stop the training-chunk loop here (65 epochs).** Running
  a chunk 4 would very likely burn another ~25-30 minutes for no real
  gain, based on direct evidence, not a guess. The next experiment, if
  continuing, should push the search budget further (e.g. pop=256,
  gens=15+) on the fixed 65-epoch checkpoint, since that's the lever
  that was still producing returns.
- **Current honest status against the spec's own section 8 decision
  rule:** mean 0.31 / std 0.26 - still "SYSTEMATICALLY BIASED but
  recoverable - proceed cautiously," same category as the very first
  (15-epoch, under-searched) result, but the absolute gap to the 0.20
  viable line has roughly halved (3.1x over -> ~1.5x over).

## 2026-11-02 ~04:15 IST - owner directed: option 2 then option 3

Owner chose, from a menu of next directions: (2) check whether a slower
real control rate gives B1 enough time budget for the accuracy that
needs it, then (3) test B2 (MPC) instead of patching B1.

### Option 2: real-time control-rate budget for B1

- Measured VAE throughput from the search-budget sweep: ~8500 evals/sec.
  Cross-referencing with the accuracy-vs-budget curve:
  - 50Hz (20ms/step, the spec's own target): ~170 evals affordable ->
    near the pop=32x3 tier -> mean error ~0.58-0.63. NOT viable.
  - 20Hz (50ms): ~425 evals -> near pop=64x6 tier -> ~0.41-0.48. NOT viable.
  - 2Hz (500ms): ~4250 evals -> near pop=256x16 -> ~0.24. Close.
  - 1Hz (1000ms): ~8500 evals -> near pop=384x20 -> ~0.21. Near-viable.
- **Conclusion: B1-as-specified (raw torque output, every control tick)
  needs roughly a 1-2Hz decision rate to reach near-viable accuracy -
  30-100x slower than the spec's own 50Hz target, and slower than the
  balance task's own failure timescale** (the best classical PD
  controller falls in ~2.1s - a 0.5-1s decision latency is comparable to
  or longer than the entire window in which a fall happens, so a B1
  controller at this budget could not react in time).
- **This doesn't kill B1, it relocates it.** The spec's own section 5.3
  already describes the right fix: B1 as a SLOW (~1-2Hz) layer that sets
  a TARGET/GOAL, with a FAST existing classical controller
  (`classical_controller.py`'s PD, running at full rate) handling
  moment-to-moment stabilization underneath - i.e. Stage 3's "MW path as
  reference, PD tracks it" pattern, generalized. Not tested further
  today (time), but this is now a concretely scoped, evidence-based next
  experiment rather than a vague idea.

### Option 3: B2 (MPC) - a real, humbling lesson in single-seed noise

- Built `forward_query` (predicts s' from (s,a) via iterative imputation
  on the flat joint VAE - a scope simplification, noted in `b2_mpc.py`'s
  docstring: a proper fibre-bundle conditional model would do this more
  directly) and `b2_plan` (random-shooting MPC: sample N action
  sequences of length H, roll out on the manifold, rank by a dense
  balance reward, execute the first action of the best sequence).
- **Found a real architecture gap first:** the manifold's 9 state
  features exclude x-position/velocity (same deliberate choice as
  `walker_domain.py` - x is "unconstrained"). That means this manifold
  cannot score forward-locomotion progress at all. Tested B2 on the
  BALANCE goal instead (which the manifold does represent) - an honest
  substitution, explicitly not the original locomotion target.
- **First pass (single seed each) looked exciting and was wrong to
  trust:** N=32,H=4 survived 144/200 steps on seed 0 - a strong-looking
  number. N=96,H=8 (the spec's own suggested default) survived only
  41/200 - worse despite more compute. Isolating N vs H (still single-
  seed): H=8 configs were consistently bad (41-52/200) regardless of N;
  H=4 configs were consistently good (131-144/200) regardless of N -
  looked like a clean, confirmed "horizon-length causes model-error
  compounding" story.
- **Caught before over-claiming it: single-seed trials are exactly the
  noise trap this whole project has repeatedly flagged.** Properly
  averaging N=32,H=4 over 5 seeds: **mean 53.6/200 (std 19.9)**, individual
  runs [81,67,58,30,32] - nothing like the "144" single lucky trial.
  Random-action baseline on the identical 5 seeds: mean 25.4/200. So the
  real, trustworthy result is: **B2 beats random by about 2x, nowhere
  near the classical PD baseline (~1000-cap survival far exceeding this
  200-step cap) or Plan A's big CEM result (920.9/1000).** A genuine,
  modest, honest result - not the exciting one the first single trial
  suggested.
- **Open, not yet re-verified:** the H=4-beats-H=8 "horizon compounding"
  story was built on the same kind of single-seed trials before the
  correction above - plausible (it matches known MPC theory about model-
  error compounding) and worth believing provisionally, but has NOT yet
  been re-confirmed with the same multi-seed rigor. Flagged explicitly
  rather than quietly treated as settled.
