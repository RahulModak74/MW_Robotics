#!/usr/bin/env python3
"""Quick test: does the classical gait controller keep Walker2d upright,
and for how long / how far? No training, just run it and measure."""
import sys
import numpy as np
import gymnasium as gym
from classical_controller import WalkGaitController

n_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
push_at = int(sys.argv[2]) if len(sys.argv) > 2 else -1
push_force = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0

env = gym.make("Walker2d-v5", exclude_current_positions_from_observation=False)
obs, info = env.reset(seed=0)
ctrl = WalkGaitController(dt=env.unwrapped.dt)
data = env.unwrapped.data

xs, pitches = [], []
for step in range(n_steps):
    qpos, qvel = data.qpos.copy(), data.qvel.copy()
    action = ctrl.act(obs, qpos, qvel)
    if step == push_at:
        data.qvel[0] += push_force   # instantaneous horizontal velocity kick
    obs, r, term, trunc, info = env.step(action)
    xs.append(data.qpos[0])
    pitches.append(data.qpos[2])
    if term or trunc:
        print(f"ENDED at step {step} (term={term} trunc={trunc}): x={xs[-1]:.3f} pitch={pitches[-1]:.3f}")
        break
else:
    print(f"SURVIVED all {n_steps} steps: final x={xs[-1]:.3f}, pitch range [{min(pitches):.3f},{max(pitches):.3f}]")

print(f"x traveled: {xs[-1] - xs[0]:.3f} m over {len(xs)} steps ({len(xs)*env.unwrapped.dt:.2f}s)")
env.close()
