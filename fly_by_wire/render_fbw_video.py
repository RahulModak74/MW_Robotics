#!/usr/bin/env python3
"""render_fbw_video.py - side-by-side MP4: the student ALONE (left) vs the same student + fly-by-wire config D (right), on several
motions played one after another. Input: <dir>/<motion>_alone_qpos.npz, <motion>_fbwD_qpos.npz (pm_eval_student.py --qpos-out),
plus <motion>_alone.npz / _fbwD.npz (fail_step) and <motion>_fbwD_fbw.npz (per-step FBW interventions). Reuses render_dance.model().

    ~/racer_venv/bin/python render_fbw_video.py out.mp4 <dir> motion1 motion2 ...
"""
import os, sys
import numpy as np, mujoco, imageio
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../for_distilation/protomotions_g1/scripts"))
from render_dance import model, W, H, FPS

FONT = ImageFont.load_default(size=24)
SMALL = ImageFont.load_default(size=20)


def load(d, m, arm):
    q = np.load(f"{d}/{m}_{arm}_qpos.npz")["qpos"][:, :36].astype(np.float64)
    fail = int(np.load(f"{d}/{m}_{arm}.npz", allow_pickle=True)["fail_step"])
    inter = np.load(f"{d}/{m}_{arm}_fbw.npz")["inter"] if arm == "fbwD" else None
    if fail >= 0:
        q = q[:fail + 25]                                                    # show the fall, then hold
    return q, fail, inter


def render(m, r, q, fail, inter, label, n, motion):
    d = mujoco.MjData(m); cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.azimuth, cam.elevation = 3.4, 150, -12
    q = q.copy(); q[:, :2] -= q[0, :2]
    out = []
    for t in range(n):
        qt = q[min(t, len(q) - 1)]
        d.qpos[:36] = qt; mujoco.mj_forward(m, d); cam.lookat[:] = [qt[0], qt[1], 0.75]
        r.update_scene(d, cam); img = Image.fromarray(r.render()); g = ImageDraw.Draw(img)
        g.text((16, 12), label, fill=(255, 255, 255), font=FONT)
        g.text((16, 44), motion, fill=(235, 235, 235), font=SMALL)
        g.text((16, H - 36), f"t = {t / FPS:4.1f} s", fill=(40, 40, 40), font=SMALL)
        if fail >= 0 and t >= fail:
            g.text((W // 2 - 60, H // 2 - 200), "FELL", fill=(220, 30, 30), font=ImageFont.load_default(size=56))
        if inter is not None and t < len(inter):
            on = bool(inter[t]); g.rectangle([W - 210, 12, W - 12, 46], fill=(200, 120, 0) if on else (40, 140, 60))
            g.text((W - 200, 16), "FBW correcting" if on else "pilot unchanged", fill=(255, 255, 255), font=SMALL)
        out.append(np.asarray(img))
    return out


def main():
    out, d, motions = sys.argv[1], sys.argv[2], sys.argv[3:]
    m = model(); r = mujoco.Renderer(m, H, W); video = []
    for mo in motions:
        a, fa, _ = load(d, mo, "alone"); b, fb, ib = load(d, mo, "fbwD")
        n = max(len(a), len(b))
        L = render(m, r, a, fa, None, "Student alone", n, mo)
        Rr = render(m, r, b, fb, ib, "Student + fly-by-wire (D)", n, mo)
        video += [np.concatenate([x, y], 1) for x, y in zip(L, Rr)]
        print(f"{mo}: alone {'fell at %.1f s' % (fa / FPS) if fa >= 0 else 'ok'}, FBW {'fell at %.1f s' % (fb / FPS) if fb >= 0 else 'ok'}"
              f", FBW corrected {ib.mean() * 100:.0f} % of steps, {n / FPS:.1f} s")
    imageio.mimwrite(out, video, fps=FPS, codec="libx264", quality=8, macro_block_size=8)
    print(f"{out} ({len(video)} frames, {len(video) / FPS:.1f} s)")


if __name__ == "__main__":
    main()
