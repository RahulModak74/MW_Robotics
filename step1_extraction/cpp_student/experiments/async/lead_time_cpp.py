#!/usr/bin/env python3
"""lead_time_cpp.py — the 18-rescue async pass-rate with the C++ kernel worker.

Same decisive criterion as ../ASYNC_APPROACH/lead_time_analysis.py:

    a rescue is SAVED  iff  first_solve_latency < warning_lead
    warning_lead = (student_alone_fall_step - first_warning_step) * 20 ms   [student dynamics, FIXED]

The ONLY thing C++ changes is the solve latency. The recorded solve_ms in the fbw2 npz is the PYTHON
fbw solve; the C++ kernel is bit-identical but faster by a factor measured at 4.3-4.5x on real trigger
states (smoke_student.py), so  cpp_solve = python_solve / SPEEDUP, per rescue. This is the faithful
per-state estimate: the expensive and cheap solves scale by the same measured factor.

Lower bound: SPEEDUP=4.5 assumes NO extra gain from the exact noise-collapse shortcut. If the
deployment look-ahead is deterministic (FBW_LA_NOISE=0), the costly S=4 solves collapse 4x further,
so the true rescue count is >= what this prints.
"""
import argparse, glob, json, os
import numpy as np

FALL = {
    ("dance", "n0.1", "s0", 0): 160, ("dance", "n0.2", "s0", 0): 160,
    ("dance", "n0.2", "s1", 0): 207, ("dance", "n0.2", "s2", 0): 99,
    ("heldout_kdance", "n0", "s0", 4): 198, ("heldout_kdance", "n0", "s0", 5): 173,
    ("heldout_kdance", "n0", "s0", 10): 354, ("heldout_kdance", "n0", "s0", 12): 190,
    ("heldout_kdance", "n0", "s0", 16): 250, ("heldout_kdance", "n0", "s0", 18): 165,
    ("heldout_kdance", "n0", "s0", 19): 211, ("heldout_kdance", "n0", "s0", 20): 255,
    ("heldout_kdance", "n0", "s0", 25): 147, ("heldout_kdance", "n0", "s0", 29): 231,
    ("heldout_kfamiliar", "n0", "s0", 5): 89, ("heldout_kfamiliar", "n0", "s0", 6): 252,
    ("heldout_kfamiliar", "n0", "s0", 15): 274, ("heldout_kfamiliar", "n0", "s0", 19): 134,
}
DT_MS = 20.0


def parse(fn):
    b = os.path.basename(fn).replace("_fbw2.npz", ""); p = b.split("__"); return p[1], p[2], p[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", default="eval_prob/fbw2")
    ap.add_argument("--speedup", type=float, default=4.5, help="measured C++/python solve speedup (smoke_student.py)")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.logdir, "*_fbw2.npz")))
    thr = json.loads(str(np.load(files[0], allow_pickle=True)["cfg"])).get("fbw2_thr", 0.0136)
    print(f"thr={thr}  SPEEDUP={a.speedup}x  save iff cpp_solve_ms < lead_ms   (cpp_solve = python_solve / SPEEDUP)\n")
    print(f"{'motion':20s}{'ep':>3s}{'fall':>6s}{'lead_ms':>8s}{'py_solve':>9s}{'cpp_solve':>10s}  py -> cpp")
    nsave_py = nsave_cpp = 0
    for fn in files:
        z = np.load(fn, allow_pickle=True); mot, noise, seed = parse(fn)
        ep, sig, gate, t = z["episode"], z["sig"], z["gate"], z["t_fbw"] * 1e3
        for e in np.unique(ep):
            key = (mot, noise, seed, int(e))
            if key not in FALL:
                continue
            s = np.where(ep == e)[0]
            wi = np.where(sig[s] > thr)[0]; gi = np.where(gate[s])[0]
            fw = int(wi[0]) if len(wi) else -1
            solve = float(t[s][gi[0]]) if len(gi) else float("nan")
            fall = FALL[key]; lead_ms = (fall - fw) * DT_MS
            cpp = solve / a.speedup
            sp = fw >= 0 and solve < lead_ms; sc = fw >= 0 and cpp < lead_ms
            nsave_py += sp; nsave_cpp += sc
            print(f"{mot[:18]+'/'+noise:20s}{e:>3d}{fall:>6d}{lead_ms:>8.0f}{solve:>9.0f}{cpp:>10.0f}  "
                  f"{'SAVE' if sp else 'late':4s} -> {'SAVE' if sc else 'late'}")
    print(f"\n=> python worker:  {nsave_py}/18 saved  -> {40+nsave_py}/58")
    print(f"=> C++ worker:     {nsave_cpp}/18 saved  -> {40+nsave_cpp}/58   (main loop <20ms throughout)")
    print("   SAVE = correction LANDS before the fall (necessary). Lower bound: noise-collapse would raise it.")


if __name__ == "__main__":
    main()
