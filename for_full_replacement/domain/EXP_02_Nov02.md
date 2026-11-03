a thorough, actionable document.

One more thing — the title should probably be something like "EXP_02_Nov2.md — Three bug-fixes to test the '3D is the wall' conclusion." Or more simply "Experiment 02 — November 2nd". Let me go with a descriptive title.

Let me also be careful about E1's fix. The original PD:

python
def controller(data):
    qpos_j = data.qpos[7:24]
    qvel_j = data.qvel[6:23]
    roll_proxy = data.qpos[4]      # quat_x
    pitch_proxy = data.qpos[5]     # quat_y
    roll_rate = data.qvel[4]       # root_wx
    pitch_rate = data.qvel[5]      # root_wy

    target = TARGET.copy()
    pitch_corr = -BAL_KP * pitch_proxy - BAL_KD * pitch_rate
    roll_corr = -BAL_KP * roll_proxy - BAL_KD * roll_rate
    target[2] += pitch_corr   # WRONG: is abdomen_x, not right_hip_y
    target[8] += pitch_corr   # WRONG: is left_hip_z, not left_hip_y
    target[0] += roll_corr    # WRONG: is abdomen_z, not right_hip_x
    target[7] += roll_corr    # OK: is left_hip_x, but only one side
Fixed:

python
def controller(data):
    qpos_j = data.qpos[7:24]
    qvel_j = data.qvel[6:23]
    
    roll_proxy = data.qpos[4]        # quat_x ≈ half roll
    pitch_proxy = data.qpos[5]       # quat_y ≈ half pitch
    roll_rate = data.qvel[4]         # root_wx
    pitch_rate = data.qvel[5]        # root_wy
    
    target = TARGET.copy()
    pitch_corr = -BAL_KP * pitch_proxy - BAL_KD * pitch_rate
    roll_corr = -BAL_KP * roll_proxy - BAL_KD * roll_rate
    
    # Pitch: abdomen_y(1), right_hip_y(5), left_hip_y(9)
    target[1] += pitch_corr
    target[5] += pitch_corr
    target[9] += pitch_corr
    
    # Roll: abdomen_x(2), right_hip_x(3), left_hip_x(7)
    target[2] += roll_corr
    target[3] += roll_corr
    target[7] += roll_corr
    
    torque = KP * (target - qpos_j) - KD * qvel_j
    return np.clip(torque, -0.4, 0.4)
Wait — but the sign of the correction matters. If pitch is positive (leaning forward or backward?) and we want to counter it, we need to lean the other way. The sign of pitch_corr depends on how the joint axis maps to the world. This is the hard part.

For a typical humanoid:

Standing upright at quat = [1,0,0,0] = no rotation

Small pitch tilt forward (say, +5 degrees around Y axis in the body frame) → quat_y becomes positive (approximately +sin(θ/2) ≈ +0.044)

To counter this and straighten up, you push the hip_y joints backward (negative direction if + pushes forward)

So pitch_corr = -BAL_KP * pitch_proxy should give a negative target, which (if +hip_y rotates leg forward relative to body) pushes the leg backward relative to body — which lifts the body forward. Correct sign.

But this is empirical. The safest approach: try both signs and let the data decide. That's what tune_balance.py did for Walker2d (swept both signs and both gains). The spec should include a sign sweep.

OK, I think the cleanest E1 is:

Fix indices (per above)

Sweep BAL_KP, BAL_KD in a small grid

Sweep sign (+1, -1)

Report best

That's maybe 20 PD runs at ~10s each = 200s = 3 min. Fine.

Actually wait, looking at the original tune_balance.py from earlier — it did a grid sweep of (sign, kp_b, kd_b). Let me do the same for Humanoid.

Actually, the sign of the correction also depends on the sign of the joint axis which is fixed by the URDF. So a "sign sweep" is really just a "which way does +y rotate the leg" question. It should be consistent across all three pitch joints and all three roll joints. So it's 2 pitch signs × 2 roll signs = 4 combinations. Plus 3×3 grid of (kp, kd) = 9 combos. Total 36 runs at ~10s each = 6 min. Acceptable.

Or, to be more efficient: sweep pitch sign × roll sign × a few key gain combos = ~16 runs. Good enough for the spec.

Hmm, actually for E1 to be cleanly comparable to the "hand PD got 58.1" reference, I should report the best result and note that it's a "best of N grid" number — that's honest. Or report the grid distribution.

Let me be pragmatic: run a small grid, report the best and the distribution. If best > 300, that's a big deal.

OK let me write the spec now. Structure:

text
# EXP_02_Nov2.md

## TL;DR
Three specific bug-fixes, each addressing a specific reason the current PROGRESS_LOG conclusion is premature.
Two hours total. E1 alone resolves 70% of the ambiguity.

## Why this exists
- PROGRESS_LOG concluded "3D is the wall"
- But three of the five data points in that log have known issues
- Fix the issues, the conclusion may reverse

## The three bugs
1. PD joint indices are wrong (3/4 corrections on wrong joints)
2. Features missing root linear velocities (blind to forward/sideways drift)
3. Fitness is survival-only (Humanoid reward signal discarded)

## E1 — PD joint index fix
### What changes
### File: exp_e1_pd_fixed.py
### Command
### Expected
### Interpretation

## E2 — velocity features
### ...

## E3 — reward fitness
### ...

## E2b — combined (conditional, run only if E2 or E3 shows partial gain)
### ...

## Run all three
### File: run_exp_02.sh

## Results table template
## Decision tree
## Time budget
## What NOT to do
Let me write it.

Actually, one more consideration: should E1 be completely separate from E2/E3 (which use CEM)? Yes — E1 is "does classical control work?" while E2/E3 are "does CEM+rank-4 work with better inputs?" They're independent questions and both inform the wall hypothesis. Keep them separate.

Also, I realized I should be explicit about one thing: the current PROGRESS_LOG's "3D is the wall" claim is based on convergence of five methods to the 58-113 range. But:

The buggy PD's 58.1 tells us nothing about "PD can't do better than CEM" — it tells us the PD was broken

The flat-linear 97.2 and rank-4 99-113 all used the same blind features and same survival-only fitness, so they're all testing the same (possibly flawed) setup

The "3D is the wall" conclusion might be better stated as "the current pipeline's feature/fitness choices work at 6 DoF but not at 17 DoF" — which is a very different claim from "3D balance is intrinsically harder than 2D"

If E1+E2+E3 all pass, the wall hypothesis is refuted — the problem was the pipeline details, not the physics. If they all fail, the wall is confirmed but for a different reason than stated.

Either way, the current conclusion doesn't survive. That's what the spec should establish.

OK, writing now.

I'll aim for a dense but scannable markdown. Not too long — the user already has a lot of context. Maybe 400-500 lines? Let me aim for something comprehensive but not bloated.

markdown
# EXP_02_Nov2.md — Three Bug-Fixes Before Accepting "3D Is the Wall"

**One-sentence claim this file tests:** The `PROGRESS_LOG.md` conclusion that
*"true 3D balance is the dominant wall, not dimensionality"* is **premature**,
because three of the five methods in its convergence table have concrete,
identifiable bugs. Fix the bugs. If the results don't move, the conclusion
stands (and is now actually supported). If they do move, the conclusion is
refuted and the roadmap changes.

**Total time:** ~3 hours wall-clock on a laptop CPU.

---

## 0. Context

`PROGRESS_LOG.md` (Nov 2, 18:35 IST) reports five independent methods all
landing in the range **58–113 steps** on Humanoid-v5, versus Walker2d's
**896–921**, and concludes that 3D balance itself is the wall. Three of
those five methods have:

| Method | Buggy element |
|---|---|
| Hand-designed PD (58.1 steps) | 3 of 4 balance corrections applied to wrong joints |
| Flat-linear CEM (97.2 steps) | Features missing root linear velocities |
| Rank-4 CEM (99.0 / 112.7 steps) | Same missing velocity features, same survival-only fitness |

Fix the bugs, re-run. The comparison becomes meaningful.

---

## 1. The Three Bugs, Precisely

### Bug 1 — PD joint indices are wrong

`pd_humanoid.py` (reproduced in `code_snapshot_29dofpy.txt`) claims to
apply `pitch_corr` to `right_hip_y` and `left_hip_y`, and `roll_corr` to
`right_hip_x` and `left_hip_x`. The actual joints those indices touch:

```python
# Current (buggy):
target[2] += pitch_corr   # comment says "right hip_y"
                          # actual joint: abdomen_x  (a ROLL joint)
target[8] += pitch_corr   # comment says "left hip_y"
                          # actual joint: left_hip_z (a YAW joint)
target[0] += roll_corr    # comment says "right hip_x"
                          # actual joint: abdomen_z  (a YAW joint)
target[7] += roll_corr    # comment says "left hip_x"
                          # actual joint: left_hip_x ✓ (only correct one)
Humanoid-v5 joint order (verified from the env's XML, 0-indexed):

idx	joint	rotates	for
0	abdomen_z	yaw	torso
1	abdomen_y	pitch	torso
2	abdomen_x	roll	torso
3	right_hip_x	roll	right leg
4	right_hip_z	yaw	right leg
5	right_hip_y	pitch	right leg
6	right_knee	knee	right leg
7	left_hip_x	roll	left leg
8	left_hip_z	yaw	left leg
9	left_hip_y	pitch	left leg
10	left_knee	knee	left leg
11–16	shoulders, elbows	arm	both arms
Correct PD corrections:

Pitch (forward/back balance): target[1], target[5], target[9]

Roll (side balance): target[2], target[3], target[7]

Yaw (facing) — often not needed for balance; leave at 0.

Note on sign: whether +pitch_corr or −pitch_corr pushes the body
upright depends on the joint axis direction, which is fixed by the URDF.
The sign must be swept, not assumed. The spec does this.

Bug 2 — features missing root linear velocities
rank4_humanoid.py::get_features returns 42 dims. Missing:
data.qvel[0:3] — the world-frame root linear velocities (vx, vy, vz).

Present (42 dims)	Missing (3 dims)
height (1)	vx (forward/lateral drift)
quaternion (4)	vy (lateral drift)
17 joint angles	vz (vertical bob)
3 root angular velocities	
17 joint velocities	
Without vx/vy, the policy is blind to sideways drift — it can
perceive orientation error but not translational error. On Walker2d,
lateral drift is physically impossible (2D-constrained), so this doesn't
matter. On Humanoid-v5, lateral drift is one of the two main ways to
fall. The policy literally cannot learn to correct drift it can't see.

Bug 3 — fitness is survival-only
rank4_humanoid.py::fitness returns mean(steps_survived). Humanoid-v5's
environmental reward is 5 × healthy_reward + 1.25 × forward_velocity − control_cost — it's already shaped for walking. Using survival-only as
fitness throws that signal away and rewards standing still (max survival
for a non-walking body is: stand indefinitely).

run_demo_walker.sh (Walker2d) already showed that replacing
survival-only with forward_distance + small_survival produces an actual
gait (1.33 m forward). The same fix must be applied to Humanoid.

2. Experiment E1 — Fixed PD Controller
Tests: Does a correctly-wired classical controller get Humanoid-v5
above the current 58.1 baseline, and if so, by how much?

Independent of E2/E3: E1 is a classical control test, not a CEM test.
Whether it passes or fails has no direct bearing on CEM's scalability —
but it does test whether "3D balance is intrinsically hard" or "the
specific controller was broken".

File: domain/exp_e1_pd_fixed.py
python
#!/usr/bin/env python3
"""
E1 — Fixed joint indices for the hand-designed PD balance controller.
Sweeps pitch-sign and roll-sign, and a small (kp, kd) grid, because the
joint-axis sign cannot be assumed from the URDF alone.
"""
import sys
import numpy as np
import gymnasium as gym

# Humanoid-v5 joint limits read from the env; the exact values are
# not important for correctness, but keeping them inside the actuator
# range is (torque is clipped to ±0.4 anyway).
KP_STATIC = np.array([2,2,2,  3,2,3, 3,  3,2,3, 3,  1,1,1, 1,1,1], dtype=float)
KD = np.array([0.3] * 17)
TARGET = np.zeros(17)
TARGET[6] = -0.3
TARGET[10] = -0.3

# Correct joint indices (see header comment)
PITCH_IDX = [1, 5, 9]    # abdomen_y, right_hip_y, left_hip_y
ROLL_IDX  = [2, 3, 7]    # abdomen_x, right_hip_x, left_hip_x


def controller(data, pitch_sign, roll_sign, bal_kp, bal_kd):
    qpos_j = data.qpos[7:24]
    qvel_j = data.qvel[6:23]

    # quat_x ≈ half-roll, quat_y ≈ half-pitch for small tilts from identity
    roll_proxy  = data.qpos[4]
    pitch_proxy = data.qpos[5]
    roll_rate   = data.qvel[4]
    pitch_rate  = data.qvel[5]

    target = TARGET.copy()
    pitch_corr = pitch_sign * (-bal_kp * pitch_proxy - bal_kd * pitch_rate)
    roll_corr  = roll_sign  * (-bal_kp * roll_proxy  - bal_kd * roll_rate)

    for i in PITCH_IDX:
        target[i] += pitch_corr
    for i in ROLL_IDX:
        target[i] += roll_corr

    torque = KP_STATIC * (target - qpos_j) - KD * qvel_j
    return np.clip(torque, -0.4, 0.4)


def rollout(pitch_sign, roll_sign, bal_kp, bal_kd, seed, max_steps=1000):
    env = gym.make("Humanoid-v5")
    obs, info = env.reset(seed=seed)
    data = env.unwrapped.data
    steps = 0
    for _ in range(max_steps):
        act = controller(data, pitch_sign, roll_sign, bal_kp, bal_kd)
        obs, r, term, trunc, info = env.step(act)
        steps += 1
        if term:
            break
    env.close()
    return steps


def sweep(pitch_signs, roll_signs, kp_vals, kd_vals, n_held_out=10):
    """Returns list of (pitch_sign, roll_sign, kp, kd, mean_steps, individual)."""
    results = []
    held_out_seeds = list(range(100, 100 + n_held_out))
    for ps in pitch_signs:
        for rs in roll_signs:
            for kp in kp_vals:
                for kd in kd_vals:
                    perfs = [rollout(ps, rs, kp, kd, s) for s in held_out_seeds]
                    mean = float(np.mean(perfs))
                    results.append((ps, rs, kp, kd, mean, perfs))
                    print(f"  pitch_sign={ps:+d}  roll_sign={rs:+d}  "
                          f"kp={kp:>4.1f}  kd={kd:>4.1f}  "
                          f"mean={mean:6.1f}  "
                          f"best={max(perfs):>4d}  worst={min(perfs):>3d}",
                          flush=True)
    return results


if __name__ == "__main__":
    print("E1 — Fixed PD controller, Humanoid-v5")
    print("Reference (buggy PD from PROGRESS_LOG.md): 58.1 steps")
    print("Reference (rank-4 CEM, best so far):      112.7 steps")
    print("Reference (Walker2d linear, for scale):   920.9 steps")
    print()

    # Sweep both signs and a small gain grid. 2 * 2 * 3 * 3 = 36 runs.
    results = sweep(
        pitch_signs=[+1, -1],
        roll_signs=[+1, -1],
        kp_vals=[4.0, 8.0, 12.0],
        kd_vals=[0.8, 1.5, 2.5],
        n_held_out=10,
    )
    results.sort(key=lambda r: -r[4])
    print(f"\nTop 5 configurations:")
    for ps, rs, kp, kd, mean, perfs in results[:5]:
        print(f"  pitch_sign={ps:+d}  roll_sign={rs:+d}  "
              f"kp={kp:>4.1f}  kd={kd:>4.1f}  mean={mean:6.1f}")

    best = results[0]
    print(f"\nBest mean: {best[4]:.1f} steps")
    print(f"  vs buggy PD:              58.1  ({best[4] / 58.1:>5.2f}x)")
    print(f"  vs rank-4 CEM (112.7):    {best[4] / 112.7:>5.2f}x")
    print(f"  vs Walker2d (920.9):      {best[4] / 920.9:>5.2f}x")

    np.save("exp_e1_pd_best_config.npy",
            np.array([best[0], best[1], best[2], best[3]], dtype=float))
    print(f"\nSaved best config to exp_e1_pd_best_config.npy")
Command
bash
cd ~/MW_Robotics/domain
python3 exp_e1_pd_fixed.py 2>&1 | tee logs/exp_e1_$(date -u +%Y%m%d_%H%M%S).log
Expected output
Two branches:

best mean ≥ 250 → PD is not broken; the CEM methods are genuinely behind classical control. Conclusion flips: the wall is not 3D physics, it's CEM's search efficiency. That changes the whole roadmap.

best mean ∈ [80, 200] → PD improves with fixed wiring but doesn't dominate CEM. Weakly confirms 3D is hard, but the 58.1 baseline was just buggy. Data point updates, conclusion survives but weaker.

best mean < 100 → Genuinely confirms the PROGRESS_LOG's conclusion. Correctly-wired PD also plateaus. The 3D hypothesis is now backed by a valid method.

Interpretation notes
If the two signs (pitch_sign, roll_sign) matter a lot — e.g., +1,+1 gives 200 but −1,−1 gives 50 — that means the joint axes are fixed and the correct sign is knowable. Note which sign wins and hard-code it in future PDs.

If all sign combinations give similar means, the controller is probably unstable regardless of sign — that's a different diagnosis (gains wrong, or model has ground-truth offset).

3. Experiment E2 — Add Velocity Features
Tests: Does adding data.qvel[0:3] (root linear velocities) to the
feature vector move the rank-4 CEM result above the current 99–113 band?

Isolation: Only the feature vector changes. Fitness stays
survival-only. Same rank-4 head, same hyperparameters, same budget as
rank4_humanoid.py. This isolates the feature effect, so if E2 succeeds
and E3 (fitness) doesn't, we know which one mattered.

File: domain/exp_e2_rank4_humanoid_vel.py
python
#!/usr/bin/env python3
"""
E2 — Rank-4 humanoid with root linear velocities added to features.
Everything else identical to rank4_humanoid.py (same DIM philosophy,
same fitness, same pop/gens/max_steps).
"""
import sys, os, time
import numpy as np
import gymnasium as gym

ACT_DIM = 17
# OLD FEAT_DIM = 42. NEW adds 3 velocities (qvel[0:3]) → 45.
FEAT_DIM = 45
RANK = 4
DIM = RANK * FEAT_DIM + ACT_DIM * RANK + ACT_DIM   # 4*45 + 17*4 + 17 = 265


def get_features(data):
    return np.concatenate([
        [data.qpos[2]],         # height
        data.qpos[3:7],         # quaternion
        data.qvel[0:3],         # ← NEW: linear velocities (vx, vy, vz)
        data.qpos[7:24],        # 17 joint angles
        data.qvel[3:6],         # 3 root angular velocities
        data.qvel[6:23],        # 17 joint velocities
    ])


def unpack(theta):
    i = RANK * FEAT_DIM
    j = i + ACT_DIM * RANK
    V = theta[:i].reshape(RANK, FEAT_DIM)
    U = theta[i:j].reshape(ACT_DIM, RANK)
    b = theta[j:]
    return V, U, b


def policy_action(theta, feat):
    V, U, b = unpack(theta)
    z = V @ feat
    return 0.4 * np.tanh(U @ z + b)


def rollout(theta, env, seed, max_steps):
    obs, info = env.reset(seed=seed)
    data = env.unwrapped.data
    steps = 0
    for _ in range(max_steps):
        feat = get_features(data)
        act = policy_action(theta, feat)
        obs, r, term, trunc, info = env.step(act)
        steps += 1
        if term:
            break
    return steps


def fitness(theta, env, seeds, max_steps):
    return np.mean([rollout(theta, env, s, max_steps) for s in seeds])


def cem(pop_size=80, generations=30, max_steps=400, n_eval_seeds=3, seed=0):
    rng = np.random.default_rng(seed)
    env = gym.make("Humanoid-v5")
    mean = np.zeros(DIM); std = np.ones(DIM) * 0.3
    n_elite = max(1, int(pop_size * 0.2))
    for gen in range(generations):
        pop = mean + std * rng.standard_normal((pop_size, DIM))
        gen_seeds = rng.integers(0, 100_000, size=n_eval_seeds).tolist()
        fits = np.array([fitness(t, env, gen_seeds, max_steps) for t in pop])
        elite = pop[np.argsort(fits)[-n_elite:]]
        mean, std = elite.mean(0), elite.std(0) + 1e-3
        print(f"  gen {gen:2d}  pop_mean={fits.mean():7.1f}  "
              f"pop_best={fits.max():7.1f}  std_norm={np.linalg.norm(std):.3f}",
              flush=True)
    env.close()
    return mean


if __name__ == "__main__":
    gens = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    print(f"E2 — rank-4 + velocity features on Humanoid-v5")
    print(f"  DIM={DIM} (was 253 without velocities)")
    print(f"  budget: pop=80, gens={gens}, max_steps=400")
    print(f"  baseline: rank4_humanoid.py @ 30 gen = 99.0, @ 90 gen = 112.7")
    print()

    t0 = time.time()
    best_theta = cem(generations=gens)
    print(f"\nCEM finished in {time.time()-t0:.1f}s")

    env = gym.make("Humanoid-v5")
    held_out = fitness(best_theta, env, list(range(100, 120)), max_steps=1000)
    env.close()
    print(f"\nHeld-out mean (20 seeds, cap 1000): {held_out:.1f} steps")
    print(f"  rank4 baseline (30 gens, no vel): 99.0")
    print(f"  rank4 baseline (90 gens, no vel): 112.7")
    print(f"  random baseline: 26.1")

    np.save("exp_e2_rank4_humanoid_vel_theta.npy", best_theta)
Command
bash
python3 exp_e2_rank4_humanoid_vel.py 2>&1 | tee logs/exp_e2_$(date -u +%Y%m%d_%H%M%S).log
Expected output
held_out ≥ 250 → velocity features were the missing signal. CEM was blind to drift. Conclusion flips.

held_out ∈ [130, 250] → velocity features help meaningfully but 3D is still harder. Partial credit; combine with E3.

held_out ≈ 99–120 → velocity features don't matter at this budget. Confirms the wall is not (primarily) about feature engineering.

Cost
~20–25 min CPU (same budget as the original rank-4 run).

4. Experiment E3 — Reward-Based Fitness
Tests: Does using the environment's own reward signal (which is already
shaped for walking) instead of step count move the rank-4 result above
the 99–113 band?

Isolation: Only the fitness changes. Features stay at the original 42
dims (no velocities). This is the cleanest single-variable test of the
"survival-only fitness was the problem" hypothesis.

File: domain/exp_e3_rank4_humanoid_reward.py
python
#!/usr/bin/env python3
"""
E3 — Rank-4 humanoid with env-reward-sum fitness instead of survival steps.
Everything else identical to rank4_humanoid.py.
"""
import sys, os, time
import numpy as np
import gymnasium as gym

ACT_DIM = 17
FEAT_DIM = 42   # original features, NO velocities — E2 tested that separately
RANK = 4
DIM = RANK * FEAT_DIM + ACT_DIM * RANK + ACT_DIM   # 253


def get_features(data):
    return np.concatenate([
        [data.qpos[2]],
        data.qpos[3:7],
        data.qpos[7:24],
        data.qvel[3:6],
        data.qvel[6:23],
    ])


def unpack(theta):
    i = RANK * FEAT_DIM
    j = i + ACT_DIM * RANK
    V = theta[:i].reshape(RANK, FEAT_DIM)
    U = theta[i:j].reshape(ACT_DIM, RANK)
    b = theta[j:]
    return V, U, b


def policy_action(theta, feat):
    V, U, b = unpack(theta)
    z = V @ feat
    return 0.4 * np.tanh(U @ z + b)


def rollout_reward(theta, env, seed, max_steps):
    """Return (sum_reward, steps_survived) — needed for both fitness and
    diagnostics."""
    obs, info = env.reset(seed=seed)
    data = env.unwrapped.data
    total_r = 0.0
    steps = 0
    for _ in range(max_steps):
        feat = get_features(data)
        act = policy_action(theta, feat)
        obs, r, term, trunc, info = env.step(act)
        total_r += float(r)
        steps += 1
        if term:
            break
    return total_r, steps


def fitness_reward(theta, env, seeds, max_steps):
    return float(np.mean([rollout_reward(theta, env, s, max_steps)[0] for s in seeds]))


def cem(pop_size=80, generations=30, max_steps=400, n_eval_seeds=3, seed=0):
    rng = np.random.default_rng(seed)
    env = gym.make("Humanoid-v5")
    mean = np.zeros(DIM); std = np.ones(DIM) * 0.3
    n_elite = max(1, int(pop_size * 0.2))
    for gen in range(generations):
        pop = mean + std * rng.standard_normal((pop_size, DIM))
        gen_seeds = rng.integers(0, 100_000, size=n_eval_seeds).tolist()
        fits = np.array([fitness_reward(t, env, gen_seeds, max_steps) for t in pop])
        elite = pop[np.argsort(fits)[-n_elite:]]
        mean, std = elite.mean(0), elite.std(0) + 1e-3
        print(f"  gen {gen:2d}  mean_reward={fits.mean():9.1f}  "
              f"best_reward={fits.max():9.1f}  std_norm={np.linalg.norm(std):.3f}",
              flush=True)
    env.close()
    return mean


if __name__ == "__main__":
    gens = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    print(f"E3 — rank-4 + env-reward-sum fitness on Humanoid-v5")
    print(f"  DIM={DIM}, budget: pop=80, gens={gens}, max_steps=400")
    print(f"  baseline: rank4_humanoid.py @ 30 gen = 99.0")
    print()

    t0 = time.time()
    best_theta = cem(generations=gens)
    print(f"\nCEM finished in {time.time()-t0:.1f}s")

    env = gym.make("Humanoid-v5")
    held_out_seeds = list(range(100, 120))
    rewards, steps = [], []
    for s in held_out_seeds:
        r, st = rollout_reward(best_theta, env, s, max_steps=1000)
        rewards.append(r); steps.append(st)
    env.close()
    print(f"\nHeld-out (20 seeds, cap 1000):")
    print(f"  mean sum_reward = {np.mean(rewards):9.1f}")
    print(f"  mean steps      = {np.mean(steps):6.1f}   "
          f"(baseline 99.0)")
    print(f"  random baseline steps = 26.1")

    np.save("exp_e3_rank4_humanoid_reward_theta.npy", best_theta)
Command
bash
python3 exp_e3_rank4_humanoid_reward.py 2>&1 | tee logs/exp_e3_$(date -u +%Y%m%d_%H%M%S).log
Expected output
held_out_steps ≥ 250 → reward-shaped fitness was the fix. The policy learns to walk and stays upright because walking pays better than standing still. Conclusion flips.

held_out_steps ∈ [150, 250] → meaningful improvement but not dominance.

held_out_steps ≈ 99–120 → reward shaping doesn't help at this budget. Confirms 3D complexity.

Cost
~20–25 min CPU.

5. Optional E2b — Combined Fix
Only run this if E2 OR E3 alone shows partial gain (steps in 130–250).
If both E2 and E3 individually give < 130, the combination is unlikely to
cross the threshold and the wall is confirmed. If both individually give
≥ 250, the wall is already refuted and E2b is a waste.

E2b is literally the two fixes stacked:

python
# In exp_e2b file:
FEAT_DIM = 45                                   # E2's feature fix
def fitness(theta, env, seeds, max_steps):
    return fitness_reward(theta, env, seeds, max_steps)   # E3's reward fix
DIM becomes 4*45 + 17*4 + 17 = 265. Same budget. Run it. Same
interpretation criteria as E2/E3.

Skip E2b if you're time-constrained. It's a combination follow-up, not
a primary diagnosis.

6. Master runner — domain/run_exp_02.sh
bash
#!/usr/bin/env bash
# run_exp_02.sh — Run E1, E2, E3 in sequence, collate results.
# EXP_02_Nov2.md companion script.

set -euo pipefail
if [[ -t 1 ]]; then GRN=$'\033[32m'; RED=$'\033[31m'; YEL=$'\033[33m'; BLU=$'\033[34m'; RST=$'\033[0m'
else GRN=""; RED=""; YEL=""; BLU=""; RST=""; fi
say(){ printf "%s [%s] %s\n" "$(date -u +%H:%M:%SZ)" "${BLU}exp02${RST}" "$*"; }
ok(){  printf "%s [%s] %s\n" "$(date -u +%H:%M:%SZ)" "${GRN}exp02${RST}" "$*"; }
die(){ printf "%s [%s] %s\n" "$(date -u +%H:%M:%SZ)" "${RED}exp02${RST}" "$*" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p logs

PY="${PYTHON:-python3}"
"$PY" -c "import gymnasium, numpy, torch" || die "missing deps"

# Skip flags
SKIP_E1=0; SKIP_E2=0; SKIP_E3=0
[[ "${1:-}" == "--skip-e1" ]] && SKIP_E1=1
[[ "${1:-}" == "--skip-e2" ]] && SKIP_E2=1
[[ "${1:-}" == "--skip-e3" ]] && SKIP_E3=1

STAMP="$(date -u +%Y%m%d_%H%M%S)"
LOG_E1="logs/exp_e1_${STAMP}.log"
LOG_E2="logs/exp_e2_${STAMP}.log"
LOG_E3="logs/exp_e3_${STAMP}.log"

# ── E1
if [[ $SKIP_E1 -eq 0 && -f exp_e1_pd_fixed.py ]]; then
    say "E1 — fixed PD controller (~5–10 min, 36 configs × 10 seeds)"
    set +e
    "$PY" exp_e1_pd_fixed.py 2>&1 | tee "$LOG_E1"
    E1_RC=${PIPESTATUS[0]}
    set -e
    if [[ $E1_RC -ne 0 ]]; then
        say "E1 exited non-zero ($E1_RC) — see $LOG_E1"
    fi
else
    say "E1 — skipped"
fi

# ── E2
if [[ $SKIP_E2 -eq 0 && -f exp_e2_rank4_humanoid_vel.py ]]; then
    say "E2 — rank-4 + velocity features (~25 min)"
    set +e
    "$PY" exp_e2_rank4_humanoid_vel.py 2>&1 | tee "$LOG_E2"
    E2_RC=${PIPESTATUS[0]}
    set -e
    if [[ $E2_RC -ne 0 ]]; then
        say "E2 exited non-zero ($E2_RC) — see $LOG_E2"
    fi
else
    say "E2 — skipped"
fi

# ── E3
if [[ $SKIP_E3 -eq 0 && -f exp_e3_rank4_humanoid_reward.py ]]; then
    say "E3 — rank-4 + reward fitness (~25 min)"
    set +e
    "$PY" exp_e3_rank4_humanoid_reward.py 2>&1 | tee "$LOG_E3"
    E3_RC=${PIPESTATUS[0]}
    set -e
    if [[ $E3_RC -ne 0 ]]; then
        say "E3 exited non-zero ($E3_RC) — see $LOG_E3"
    fi
else
    say "E3 — skipped"
fi

# ── collate
say "Collating results"

E1_BEST="n/a"
E2_STEPS="n/a"
E3_STEPS="n/a"

if [[ -f "$LOG_E1" ]]; then
    E1_BEST=$(grep "Best mean:" "$LOG_E1" | tail -1 | awk '{print $3}' || echo "n/a")
fi
if [[ -f "$LOG_E2" ]]; then
    E2_STEPS=$(grep "Held-out mean" "$LOG_E2" | tail -1 | awk '{print $NF}' || echo "n/a")
fi
if [[ -f "$LOG_E3" ]]; then
    E3_STEPS=$(grep "mean steps      =" "$LOG_E3" | tail -1 | awk '{print $NF}' || echo "n/a")
fi

{
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  EXP_02 RESULTS — Three Bug-Fixes vs '3D Is the Wall'"
    echo "════════════════════════════════════════════════════════════════"
    echo ""
    echo "  E1 — fixed PD              : $E1_BEST steps    (buggy baseline: 58.1)"
    echo "  E2 — rank-4 + vel features : $E2_STEPS steps    (baseline 99.0–112.7)"
    echo "  E3 — rank-4 + reward fit   : $E3_STEPS steps    (baseline 99.0)"
    echo ""
    echo "  Reference: Walker2d linear = 920.9, Walker2d rank-4 = 896.2"
    echo "  Reference: Humanoid-v5 random baseline = 26.1"
    echo ""
    echo "  Decision matrix (any one cell green is enough to act):"
    echo "    E1 ≥ 250 → classical control works on 3D. Roadmap pivots to"
    echo "               CEM-vs-classical, not CEM-vs-3D."
    echo "    E2 ≥ 250 → velocity features were the gap. Add to all future runs."
    echo "    E3 ≥ 250 → reward shaping was the gap. Add to all future runs."
    echo "    All < 130 → PROGRESS_LOG's conclusion is now VALIDATED (with"
    echo "               correctly-configured methods), not just suggested."
    echo ""
    echo "  Logs:"
    echo "    E1: $LOG_E1"
    echo "    E2: $LOG_E2"
    echo "    E3: $LOG_E3"
    echo "════════════════════════════════════════════════════════════════"
} | tee -a logs/exp_02_summary_${STAMP}.md

exit 0
Command
bash
cd ~/MW_Robotics/domain
chmod +x run_exp_02.sh
./run_exp_02.sh                                  # full, ~1 hr
./run_exp_02.sh --skip-e2 --skip-e3              # E1 only, ~10 min
7. Interpreting the results together
Fill in this table after run_exp_02.sh finishes:

Experiment	Result	vs baseline	Verdict
E1 fixed PD	___ steps	vs 58.1	
E2 vel features	___ steps	vs 99.0	
E3 reward fit	___ steps	vs 99.0	
Decision tree (read top to bottom, stop at the first match):

If E1, E2, AND E3 all remain below 130 steps
→ PROGRESS_LOG's conclusion is now VALIDATED. Correctly-wired PD,
features that can perceive drift, and reward-shaped fitness all still
plateau. The wall is genuinely 3D balance (or something else structural
that none of these fixes touch).

Next action: build the 2D-constrained intermediate (a custom URDF with
sagittal-plane lock on the hip_y joints, same 17 joints, no lateral
motion). Run the same rank-4 pipeline on it. If it gets 800+, the 3D-vs-2D
dichotomy is confirmed. If it also plateaus, the wall is dof count, not
3D.

If exactly ONE of E1/E2/E3 crosses 250
→ The bug that got fixed was the bottleneck. Apply that fix to
rank4_humanoid.py and re-run at 90 generations. Combine with the other
unfixed bugs one at a time (ablation) to confirm.

Next action: the fixed pipeline becomes the new baseline. Re-run the
Walker2d equivalents to make sure the fix doesn't hurt 2D. If E1 (PD)
crossed, pivot to a hybrid approach (CEM initializes, PD stabilizes).

If TWO or THREE of E1/E2/E3 cross 250
→ The "3D is the wall" hypothesis is REFUTED. The pipeline was buggy.
Fix everything, re-run at moderate budget, and the humanoid should be
approaching Walker2d performance.

Next action: re-evaluate the entire 29-DoF roadmap against the new
baseline. The legs-only intermediate, waypoint lifting, and 12-DoF sprint
may all need re-ordering. Publish the negative-then-positive result as a
paper: "Empirical pitfalls in scaling gradient-free RL to 3D humanoids."

If E1 crosses 250 but E2/E3 don't
→ CEM is behind classical control on 3D humanoid, independent of
feature/fitness choices. This is the most interesting outcome: it says
the wall isn't physics, it's the algorithm.

Next action: pivot to a hybrid — use PD/LQR to stabilize, CEM to
optimize gait shape on top. Or compare against CMA-ES and PPO at matched
budget (E7 from the earlier roadmap).

If E2 or E3 crosses 250 but E1 doesn't
→ Feature/fitness engineering beats classical control here. Add the
winning fix to rank4_humanoid.py permanently and move on to the
intermediate-body roadmap.

8. Time budget and failure modes
Step	Wall-clock	Failure mode
E1	5–10 min	Gym version mismatch (indices wrong) — verify env.action_space.shape == (17,) first
E2	20–25 min	Feature dim mismatch — the code uses FEAT_DIM = 45, DIM = 265. If DIM in the log doesn't show 265, the feature count is wrong.
E3	20–25 min	Reward is float vs np.float64 casting — the code wraps in float(r) for safety
E2b (optional)	20–25 min	Only run if E2 or E3 partially succeeded
Total: ~1 hr if all three run. E1 alone (the cheapest decider) is ~10 min.

9. What NOT to do
Do not skip E1. It's the cheapest experiment (10 min) and its
result alone resolves ~70% of the ambiguity in the "3D is the wall"
claim. If E1 shows PD at 400, none of E2/E3 matter — the
architecture comparison was against a broken baseline.

Do not change the fitness in E2, or the features in E3. These
are single-variable tests. Combined changes (E2b) come only after
the individual tests tell you which direction to combine.

Do not run E2b first. It's a follow-up, not a lead. Running it
first would tell you "combined fix works" without telling you "which
single fix mattered," which is what you actually need for the roadmap.

Do not call E2/E3's result "confirming the wall" if the metric is
still survival steps. If E3 (reward fitness) succeeds at 400 steps
survival, the reward fix was the win. If it succeeds at 400 mean
reward but still dies at step 100, the reward fix wasn't the win.
Read both numbers.

Do not spend more than 1 hour on E1. If the 36-config sweep hasn't
found a config > 200 by then, stop and note the ceiling. The point
was to establish whether any classical controller works, not to
find the perfect one.

10. What this experiment is — and isn't
This experiment IS:

A minimum-effort test of whether the "3D is the wall" conclusion
survives three obvious, specific bug-fixes.

A single-variable ablation: E2 tests feature engineering in isolation,
E3 tests fitness engineering in isolation. Neither can be explained
away by the other.

Cheap. ~1 hr. Result available same day.

This experiment IS NOT:

A final architecture comparison. It does not try to beat PPO or match
ProtoMotions. It only tries to establish whether the current CEM
pipeline's humanoid numbers are genuinely representative of the
method's capability, or of a broken configuration.

A substitute for the 12-DoF intermediate sprint (E8). That's next, and
is worth ~2 weeks. This is a 1-hour sanity check that decides whether
E8 is worth starting.

11. Files this spec produces
After running everything:

text
domain/
├── exp_e1_pd_fixed.py
├── exp_e1_pd_best_config.npy                (if E1 succeeds)
├── exp_e2_rank4_humanoid_vel.py
├── exp_e2_rank4_humanoid_vel_theta.npy      (if E2 runs)
├── exp_e3_rank4_humanoid_reward.py
├── exp_e3_rank4_humanoid_reward_theta.npy   (if E3 runs)
├── run_exp_02.sh
└── logs/
    ├── exp_e1_<stamp>.log
    ├── exp_e2_<stamp>.log
    ├── exp_e3_<stamp>.log
    └── exp_02_summary_<stamp>.md
Nothing else in the repo is touched. PROGRESS_LOG.md should get a
follow-up entry (dated Nov 2, after EXP_02 results) that either confirms
or refutes its own earlier conclusion, using the new data.
