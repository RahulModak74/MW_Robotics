#!/usr/bin/env python3
"""
Stage 2, take 2: hand-tuned PD balance (classical_controller.py /
tune_balance.py) couldn't hold standing balance reliably - capped at
~2s even after a real search. That's a real, useful negative result
about PD alone on an underactuated biped, not a dead end: switching to
LQR (Linear-Quadratic Regulator), the standard classical-control answer
to exactly this problem (balancing an underactuated system near an
equilibrium). Still zero learning, zero RL - this is linear algebra
applied to MuJoCo's own real dynamics, linearized via finite differences
around the standing pose.
"""
import numpy as np
import mujoco
import scipy.linalg


def build_lqr(env):
    m, d = env.unwrapped.model, env.unwrapped.data
    nx = 2 * m.nv   # state = [qpos_deviation(nv), qvel(nv)] in MuJoCo's own tangent space
    nu = m.nu

    m.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)
    qpos0 = d.qpos.copy()

    # ctrl=0 is NOT an equilibrium: at zero control qacc is huge (free
    # fall + joints flopping, checked directly: qacc[1] (height) is
    # -9.8 m/s^2, same as gravity alone - the model is simply falling).
    # Linearizing "around" a point that's already accelerating hard makes
    # the whole LQR regulation problem meaningless. Fix: solve for the
    # feedforward torque that actually holds qacc=0 here (inverse
    # dynamics), use THAT as the nominal control, and linearize there.
    d.qacc[:] = 0.0
    mujoco.mj_inverse(m, d)
    gear = m.actuator_gear[:, 0]
    ctrl0 = np.clip(d.qfrc_inverse[3:9] / gear, -1.0, 1.0)
    d.ctrl[:] = ctrl0

    A = np.zeros((nx, nx))
    B = np.zeros((nx, nu))
    eps = 1e-6
    mujoco.mjd_transitionFD(m, d, eps, True, A, B, None, None)

    # cost: penalise pitch & its rate heavily, barely touch unactuated
    # absolute position (rootx/rootz aren't directly driven by any
    # actuator - fighting their drift made the DARE solve infeasible).
    # qpos/tangent order: 0 rootx, 1 rootz, 2 rooty, 3..8 the six joints.
    Q = np.eye(nx) * 0.05
    Q[0, 0] = 0.0                     # rootx: don't care about absolute drift
    Q[1, 1] = 2.0                     # rootz (height): mild
    Q[2, 2] = 400.0                   # rooty (pitch): the thing that must not fall over
    Q[m.nv + 0, m.nv + 0] = 0.1        # x velocity: mild
    Q[m.nv + 1, m.nv + 1] = 1.0        # z velocity: mild
    Q[m.nv + 2, m.nv + 2] = 40.0       # pitch rate: important
    R = np.eye(nu) * 0.5

    # Both the discrete and continuous ARE solvers are numerically
    # unstable here (the Hamiltonian/symplectic pencil sits right on the
    # stability boundary - expected for a lightly-damped mechanical
    # system). Side-step that with a finite-horizon backward Riccati
    # recursion instead - plain iteration, no eigenvalue decomposition,
    # robust regardless of marginal stability, and converges to the same
    # steady-state gain for a stabilizable system as the horizon grows.
    P = Q.copy()
    for _ in range(2000):
        K_i = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
        P_next = Q + A.T @ P @ (A - B @ K_i)
        if np.max(np.abs(P_next - P)) < 1e-10:
            P = P_next
            break
        P = P_next
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K, qpos0, ctrl0, A, B


def run(K, qpos0, ctrl0, n_steps=1000, push_at=-1, push=0.0, render_path=None):
    import gymnasium as gym
    env = gym.make("Walker2d-v5", exclude_current_positions_from_observation=False)
    obs, info = env.reset(seed=0)
    m, d = env.unwrapped.model, env.unwrapped.data
    nv = m.nv
    traj = []
    for step in range(n_steps):
        dqpos = np.zeros(nv)
        mujoco.mj_differentiatePos(m, dqpos, 1.0, qpos0, d.qpos)
        x = np.concatenate([dqpos, d.qvel])
        u = ctrl0 - K @ x
        u = np.clip(u, -1.0, 1.0)
        if step == push_at:
            d.qvel[0] += push
        traj.append((d.qpos.copy(), d.qvel.copy()))
        obs, r, term, trunc, info = env.step(u)
        if term:
            env.close()
            return step, traj
    env.close()
    return n_steps, traj


if __name__ == "__main__":
    import sys
    import gymnasium as gym
    env = gym.make("Walker2d-v5", exclude_current_positions_from_observation=False)
    env.reset(seed=0)
    K, qpos0, ctrl0, A, B = build_lqr(env)
    env.close()
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    push_at = int(sys.argv[2]) if len(sys.argv) > 2 else -1
    push = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    steps, traj = run(K, qpos0, ctrl0, n_steps=n, push_at=push_at, push=push)
    print(f"survived {steps}/{n} steps ({steps*0.008:.2f}s)" + (" (full duration)" if steps == n else " (fell)"))
