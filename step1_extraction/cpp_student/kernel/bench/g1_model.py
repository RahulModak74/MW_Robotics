"""g1_model.py - a stand-in for ProtoMotions' MuJoCo G1 simulator (the model FlyByWire copies), for benchmarking without
ProtoMotions' runtime. Replicates MujocoSimulator's model setup: g1_holo_compat.xml, visual meshes dropped (contype 0, density 0:
no physics effect), floor, sensors removed, robot friction 1.0, 1 kHz physics, decimation 20, passive stiffness/damping zeroed,
armature from the robot config, gear 1, position (implicit PD) actuators with the config's kp/kd/effort limits."""
import os, sys
import numpy as np
import mujoco

PM = os.environ.get("PM", os.path.expanduser("~/ProtoMotions"))        # your ProtoMotions clone
sys.path.insert(0, PM)
XML = os.path.join(PM, "protomotions/data/assets/mjcf/g1_holo_compat.xml")


def build_model():
    import protomotions.robot_configs.g1 as g
    cfg = [getattr(g, n) for n in dir(g) if n.startswith("G1") and n.endswith("RobotConfig")][0]()
    ci = cfg.control.control_info
    s = mujoco.MjSpec.from_file(XML)
    for gm in list(s.geoms):
        if gm.type == mujoco.mjtGeom.mjGEOM_MESH:
            s.delete(gm)
    for me in list(s.meshes):
        s.delete(me)
    for se in list(s.sensors):
        s.delete(se)
    f = s.worldbody.add_geom(); f.name = "floor"; f.type = mujoco.mjtGeom.mjGEOM_PLANE; f.size = [0, 0, .05]
    m = s.compile()
    robot = [b for b in range(1, m.nbody)]
    gid = np.flatnonzero(np.isin(m.geom_bodyid, robot)); m.geom_friction[gid, 0] = 1.0
    m.opt.timestep = 0.001
    m.jnt_stiffness[:] = 0; m.dof_damping[:] = 0
    for j in range(m.njnt):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        if n in ci and ci[n].armature is not None:
            m.dof_armature[m.jnt_dofadr[j]] = ci[n].armature
    for a in range(m.nu):
        gr = m.actuator_gear[a, 0]
        m.actuator_ctrlrange[a] *= gr; m.actuator_forcerange[a] *= gr
        m.actuator_gear[a] = 0; m.actuator_gear[a, 0] = 1
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[a, 0]); info = ci[n]
        kp, kd, eff = info.require("stiffness"), info.require("damping"), info.require("effort_limit")
        m.actuator_gainprm[a, 0] = kp; m.actuator_biastype[a] = 1
        m.actuator_biasprm[a, :3] = (0, -kp, -kd)
        m.actuator_forcerange[a] = (-eff, eff); m.actuator_ctrllimited[a] = 0; m.actuator_forcelimited[a] = 1
    mujoco.mj_setConst(m, mujoco.MjData(m))
    return m


class FakeSim:
    """what FlyByWire reads from ProtoMotions' simulator: model, data, decimation."""
    def __init__(self):
        self.model = build_model(); self.data = mujoco.MjData(self.model); self.decimation = 20
