#!/usr/bin/env python3
"""render_dance.py - side-by-side MP4 (and a JPEG contact sheet) of recorded G1 runs, from MuJoCo qpos saved by
pm_eval_student.py --qpos-out. Uses ProtoMotions' g1_holo_compat.xml with the real G1 meshes (from the G1 Racer bundle).

    ~/racer_venv/bin/python render_dance.py out.mp4 "MW student (1.6 M params, CPU)=dance_mw.npz" "Teacher=dance_teacher.npz"
"""
import os, sys
import numpy as np, mujoco, imageio
from PIL import Image, ImageDraw, ImageFont

os.environ.setdefault("MUJOCO_GL", "egl")
XML = os.path.join(os.environ.get("PM", os.path.expanduser("~/ProtoMotions")), "protomotions/data/assets/mjcf/g1_holo_compat.xml")
MESHES = os.environ.get("G1_MESHES", os.path.expanduser("~/g1-racer/meshes"))   # real G1 STL meshes (e.g. from the G1 Racer bundle)
W, H, FPS = 640, 720, 50


def model():
    s = mujoco.MjSpec.from_file(XML)
    s.meshdir = MESHES
    s.visual.global_.offwidth, s.visual.global_.offheight = W, H
    t = s.add_texture(); t.name = "grid"; t.type = mujoco.mjtTexture.mjTEXTURE_2D; t.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
    t.rgb1, t.rgb2, t.width, t.height = [0.82, 0.84, 0.87], [0.70, 0.72, 0.76], 512, 512
    m = s.add_material(); m.name = "grid"; m.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"; m.texrepeat = [8, 8]
    f = s.worldbody.add_geom(); f.name = "floor"; f.type = mujoco.mjtGeom.mjGEOM_PLANE; f.size = [20, 20, 0.1]; f.material = "grid"
    for pos in ([0, 0, 6], [3, -3, 4]):
        l = s.worldbody.add_light(); l.pos = pos; l.dir = [-p for p in pos]; l.diffuse = [0.6, 0.6, 0.6]
    return s.compile()


def frames(m, path, label):
    z = np.load(path); q = z["qpos"][:, :7 + 29].astype(np.float64)
    q[:, :2] -= q[0, :2]                                                # start at the origin
    d = mujoco.MjData(m); r = mujoco.Renderer(m, H, W)
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE; cam.distance, cam.azimuth, cam.elevation = 3.2, 150, -12
    font = ImageFont.load_default(size=26) if hasattr(ImageFont, "load_default") else None
    out = []
    for t, qt in enumerate(q):
        d.qpos[:36] = qt; mujoco.mj_forward(m, d)
        cam.lookat[:] = [d.qpos[0], d.qpos[1], 0.75]
        r.update_scene(d, cam); img = Image.fromarray(r.render())
        g = ImageDraw.Draw(img); g.text((16, 14), label, fill=(255, 255, 255), font=font); g.text((16, H - 40), f"t = {t * 0.02:4.2f} s", fill=(40, 40, 40), font=font)
        out.append(np.asarray(img))
    return out


def main():
    out, specs = sys.argv[1], [a.split("=", 1) for a in sys.argv[2:]]
    m = model()
    base = os.path.dirname(os.path.abspath(out))
    cols = [frames(m, p if os.path.isabs(p) else os.path.join(base, p), lab) for lab, p in specs]
    n = min(len(c) for c in cols)
    video = [np.concatenate([c[i] for c in cols], 1) for i in range(n)]
    imageio.mimwrite(out, video, fps=FPS, codec="libx264", quality=8, macro_block_size=8)
    idx = np.linspace(0, n - 1, 6).astype(int)                          # contact sheet: 6 moments, one row per run
    sheet = np.concatenate([np.concatenate([Image.fromarray(c[i]).resize((W // 2, H // 2)) for i in idx], 1) for c in cols], 0)
    Image.fromarray(sheet).save(out.replace(".mp4", ".jpg"), quality=90)
    print(f"{out} ({n} frames, {n / FPS:.1f} s) and {out.replace('.mp4', '.jpg')}")


if __name__ == "__main__":
    main()
