"""g1_kin.py - differentiable Unitree-G1 kinematics in torch (pelvis frame), built from ProtoMotions' MJCF via MuJoCo.

    kin = G1Kin.from_mjcf()                     # once (needs mujoco); then pure torch, batched, differentiable, CPU or GPU
    out = kin(q)                                # q [B, 29] joint angles in ProtoMotions dof order
    out["com"] [B,3]  whole-body centre of mass;  out["feet"] [B,2,3]  left/right ankle-roll body origins;  out["pos"] [B,nb,3]
All positions are in the pelvis frame (pelvis at the origin, identity orientation). Combine with the projected gravity vector from the
observation to get world-vertical / horizontal components. Checked against MuJoCo's own forward kinematics (check() below).
"""
import os
import numpy as np
import torch

XML = os.path.expanduser("~/ProtoMotions/protomotions/data/assets/mjcf/g1_holo_compat.xml")
DOF_NAMES = ["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint",
             "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
             "right_ankle_pitch_joint", "right_ankle_roll_joint", "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
             "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
             "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint", "right_shoulder_pitch_joint",
             "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
             "right_wrist_pitch_joint", "right_wrist_yaw_joint"]          # = deployment contract joint order (unified_pipeline.yaml)
FEET = ("left_ankle_roll_link", "right_ankle_roll_link")


def load_mj(xml=XML):
    import mujoco
    s = mujoco.MjSpec.from_file(xml)
    for g in list(s.geoms):                       # meshes are LFS pointers in some clones; inertials are explicit in the XML
        if g.type == mujoco.mjtGeom.mjGEOM_MESH:
            s.delete(g)
    for m in list(s.meshes):
        s.delete(m)
    f = s.worldbody.add_geom(); f.name = "floor"; f.type = mujoco.mjtGeom.mjGEOM_PLANE; f.size = [10, 10, .1]
    return s.compile()


def quat2mat(q):                                   # wxyz
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def axis_angle(axis, ang):                         # axis [3], ang [B] -> [B,3,3] (Rodrigues)
    K = torch.tensor([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]], dtype=ang.dtype, device=ang.device)
    s, c = torch.sin(ang)[:, None, None], torch.cos(ang)[:, None, None]
    return torch.eye(3, dtype=ang.dtype, device=ang.device) + s * K + (1 - c) * (K @ K)


class G1Kin(torch.nn.Module):
    def __init__(self, parent, pos, rot, ipos, mass, jaxis, janchor, jdof, foot_ids, names):
        super().__init__()
        self.parent, self.jdof, self.foot_ids, self.names = parent, jdof, foot_ids, names
        self.jaxis = [None if a is None else tuple(float(v) for v in a) for a in jaxis]
        for n, v in (("pos", pos), ("rot", rot), ("ipos", ipos), ("mass", mass), ("janchor", janchor)):
            self.register_buffer(n, torch.tensor(np.asarray(v), dtype=torch.float32))
        self.total_mass = float(np.sum(mass))

    @classmethod
    def from_mjcf(cls, xml=XML):
        import mujoco
        m = load_mj(xml)
        root = m.body("pelvis").id
        bodies = [b for b in range(m.nbody) if b == root or _descends(m, b, root)]
        idx = {b: i for i, b in enumerate(bodies)}
        parent = [-1 if b == root else idx[int(m.body_parentid[b])] for b in bodies]
        jaxis, janchor, jdof = [], [], []
        for b in bodies:
            js = [j for j in range(m.njnt) if m.jnt_bodyid[j] == b and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
            assert len(js) <= 1, "one hinge per body expected"
            if js:
                j = js[0]; nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
                jaxis.append(m.jnt_axis[j].copy()); janchor.append(m.jnt_pos[j].copy()); jdof.append(DOF_NAMES.index(nm))
            else:
                jaxis.append(None); janchor.append(np.zeros(3)); jdof.append(-1)
        pos = np.array([np.zeros(3) if b == root else m.body_pos[b] for b in bodies])
        rot = np.array([np.eye(3) if b == root else quat2mat(m.body_quat[b]) for b in bodies])
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in bodies]
        return cls(parent, pos, rot, m.body_ipos[bodies], m.body_mass[bodies], jaxis, janchor, jdof,
                   [names.index(f) for f in FEET], names)

    def forward(self, q):
        B = q.shape[0]
        R = [None] * len(self.parent); P = [None] * len(self.parent)
        eye = torch.eye(3, dtype=q.dtype, device=q.device).expand(B, 3, 3)
        for i, p in enumerate(self.parent):
            Rp, Pp = (eye, torch.zeros(B, 3, dtype=q.dtype, device=q.device)) if p < 0 else (R[p], P[p])
            Ri = Rp @ self.rot[i]
            Pi = Pp + (Rp @ self.pos[i])
            if self.jaxis[i] is not None:
                Rj = axis_angle(self.jaxis[i], q[:, self.jdof[i]])
                c = self.janchor[i]
                Pi = Pi + (Ri @ (c - (Rj @ c)).unsqueeze(-1)).squeeze(-1)
                Ri = Ri @ Rj
            R[i], P[i] = Ri, Pi
        Pt = torch.stack(P, 1); Rt = torch.stack(R, 1)
        cpos = Pt + (Rt @ self.ipos.unsqueeze(-1)).squeeze(-1)
        com = (cpos * self.mass[None, :, None]).sum(1) / self.total_mass
        return {"pos": Pt, "com": com, "feet": Pt[:, list(self.foot_ids)]}


def _descends(m, b, root):
    while b != 0:
        if b == root:
            return True
        b = int(m.body_parentid[b])
    return False


def check(n=20, seed=0):
    """Compare with MuJoCo forward kinematics on random joint angles (pelvis at origin, identity)."""
    import mujoco
    m = load_mj(); d = mujoco.MjData(m); kin = G1Kin.from_mjcf()
    rng = np.random.RandomState(seed)
    qadr = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j): m.jnt_qposadr[j] for j in range(m.njnt)}
    lo = np.array([m.jnt_range[m.joint(nm).id][0] for nm in DOF_NAMES]); hi = np.array([m.jnt_range[m.joint(nm).id][1] for nm in DOF_NAMES])
    Q = rng.uniform(lo, hi, size=(n, 29)).astype(np.float64)
    out = kin.double()(torch.tensor(Q))
    err_com, err_feet = [], []
    root = m.body("pelvis").id
    for k in range(n):
        d.qpos[:] = 0; d.qpos[3] = 1.0
        for i, nm in enumerate(DOF_NAMES):
            d.qpos[qadr[nm]] = Q[k, i]
        mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
        com_mj = d.subtree_com[root] - d.xpos[root]
        feet_mj = np.array([d.xpos[m.body(f).id] - d.xpos[root] for f in FEET])
        err_com.append(np.abs(out["com"][k].numpy() - com_mj).max()); err_feet.append(np.abs(out["feet"][k].numpy() - feet_mj).max())
    return max(err_com), max(err_feet), kin.total_mass


if __name__ == "__main__":
    ec, ef, M = check()
    print(f"max |com - mujoco| {ec:.2e} m, max |feet - mujoco| {ef:.2e} m, mass {M:.3f} kg")
