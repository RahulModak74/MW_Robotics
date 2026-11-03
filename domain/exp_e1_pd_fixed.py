#!/usr/bin/env python3
"""
E1 - Fixed joint indices for the hand-designed PD balance controller.
Sweeps pitch-sign and roll-sign, and a small (kp, kd) grid, because the
joint-axis sign cannot be assumed from the URDF alone.
"""
import sys
import numpy as np
import gymnasium as gym

KP_STATIC = np.array([2,2,2,  3,2,3, 3,  3,2,3, 3,  1,1,1, 1,1,1], dtype=float)
KD = np.array([0.3] * 17)
TARGET = np.zeros(17)
TARGET[6] = -0.3
TARGET[10] = -0.3

# Correct joint indices (abdomen_z,y,x | r_hip_x,z,y | r_knee | l_hip_x,z,y | l_knee | arms)
PITCH_IDX = [1, 5, 9]    # abdomen_y, right_hip_y, left_hip_y
ROLL_IDX  = [2, 3, 7]    # abdomen_x, right_hip_x, left_hip_x


def controller(data, pitch_sign, roll_sign, bal_kp, bal_kd):
    qpos_j = data.qpos[7:24]
    qvel_j = data.qvel[6:23]

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
    print("E1 - Fixed PD controller, Humanoid-v5")
    print("Reference (buggy PD from PROGRESS_LOG.md): 58.1 steps")
    print("Reference (rank-4 CEM, best so far):      112.7 steps")
    print("Reference (Walker2d linear, for scale):   920.9 steps")
    print()

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
