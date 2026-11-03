#!/usr/bin/env python3
"""
Stage 1 + 2: a coupled, contact-rich, multi-joint body (MuJoCo Walker2d: 6
actuated joints, two legs, ground contact), controlled by a CLASSICAL
closed-loop controller - no RL, no learning, no training time. This is
the "act and react" capability path_to_full_Robotics.md says doesn't
need MW or RL for a system whose equations are known (MuJoCo IS the
known equations here; we just need a control law, not a learned one).

Design: a hand-authored walking gait (hip/knee/ankle reference angles as
a function of a cycling phase variable, alternating legs), tracked by a
PD controller every step, PLUS a balance-correction term that adjusts
hip references based on current torso pitch and its rate - a simple,
textbook inverted-pendulum-style stabilizer. This is what gives real
reactive behavior: push the walker, the balance term reacts immediately,
every single frame, because it's reading the ACTUAL current state, not
interpolating toward a pre-known target.
"""
import numpy as np


class WalkGaitController:
    def __init__(self, stride_period: float = 1.0, dt: float = 0.008):
        self.stride_period = stride_period
        self.dt = dt
        self.t = 0.0
        # PD gains per joint type (thigh/hip, leg/knee, foot/ankle)
        self.kp = np.array([4.0, 4.0, 2.0, 4.0, 4.0, 2.0])
        self.kd = np.array([0.5, 0.5, 0.2, 0.5, 0.5, 0.2])
        self.balance_kp = 3.0   # how hard to react to torso pitch error
        self.balance_kd = 0.8   # how hard to react to torso pitch rate

    def reference(self, phase: float):
        """phase in [0,1). Returns 6 reference angles: thigh,leg,foot,
        thigh_left,leg_left,foot_left. Right leg leads; left is phase+0.5."""
        def leg_ref(p):
            p = p % 1.0
            # swing (p<0.5): lift knee, bring thigh forward
            # stance (p>=0.5): extend, push back
            thigh = 0.5 * np.sin(2 * np.pi * p)          # forward(+)/back(-) swing
            knee = -0.6 * max(0.0, np.sin(2 * np.pi * p))  # bend only during swing
            foot = 0.1 * np.cos(2 * np.pi * p)
            return thigh, knee, foot

        t_r, k_r, f_r = leg_ref(phase)
        t_l, k_l, f_l = leg_ref(phase + 0.5)
        return np.array([t_r, k_r, f_r, t_l, k_l, f_l])

    def act(self, obs, qpos, qvel, dt=None):
        """obs: the 17-d Walker2d observation (qpos[1:], qvel), used for
        joint angles/velocities and torso pitch. qpos/qvel passed directly
        from env.unwrapped.data for exact values (obs excludes rootx by
        default but we want it for diagnostics only, not control)."""
        self.t += (dt or self.dt)
        phase = (self.t / self.stride_period) % 1.0
        ref = self.reference(phase)

        joint_angles = qpos[3:9]     # thigh,leg,foot,thigh_left,leg_left,foot_left
        joint_vel = qvel[3:9]
        torso_pitch = qpos[2]        # rooty
        torso_pitch_rate = qvel[2]

        # balance correction: if leaning forward (pitch<0 means tipping fwd
        # in this model's convention - sign confirmed empirically below),
        # push both hips to catch the fall by extending the reference.
        balance_term = self.balance_kp * (-torso_pitch) + self.balance_kd * (-torso_pitch_rate)
        ref_corrected = ref.copy()
        ref_corrected[0] += balance_term   # right thigh
        ref_corrected[3] += balance_term   # left thigh

        torque = self.kp * (ref_corrected - joint_angles) - self.kd * joint_vel
        return np.clip(torque, -1.0, 1.0)
