"""fbw_2.py - MW-gated fly-by-wire: the MW student's own fall warning decides WHEN the flight computer acts.

fbw.py (config D, 58/58) never reads the MW student. It rolls the pilot's plan out in exact physics on EVERY step (S draws x H steps)
just to find out whether the plan is safe, then searches only when it is not. That safety check is the expensive part on hands-off
steps, and it duplicates what the MW student already reports for free: the Bayesian last layer's epistemic uncertainty ("epi",
mean over joints of sigma_epi / action sd) warns of a fall 0-25 steps ahead with AUROC 0.987 on r7c's 58 tests.

fbw_2 = MW warning as the gate, fbw.py's D search as the action:
  epi(x) <= thr and no recent alarm  -> the pilot's command passes UNCHANGED; zero physics (one student forward pass, already paid
                                        to get the action)
  epi(x) >  thr                      -> the flight computer takes over; on each such step it runs fbw.FlyByWire.filter exactly as
                                        config D (same look-ahead, envelope, search, clip), so any difference vs 58/58 is the gate.
How long it keeps control after a warning (FBW2_RELEASE):
  0 (fixed latch)   : `latch` steps after the last warning (default 10 = 0.2 s), then hands back.
  1 (follow-through): stays on until the robot is stable again: for `calm` consecutive steps (default 10) D's own look-ahead finds the
                      pilot's plan inside the envelope (no correction needed) AND the MW warning is quiet; never before `latch` steps.
                      This is how fly-by-wire returns authority: only once the aircraft is back in the envelope, not on a timer.
The threshold is set on the student's TRAINING states (95th percentile of epi on r7c's 100 training motions = 0.0136), never on the
exam. fbw.py is imported, not modified.

Env: FBW2_SIG (epi | ale | mw, default epi), FBW2_THR (default 0.0136), FBW2_LATCH (default 10), FBW2_RELEASE (0|1, default 0),
FBW2_CALM (default 10), plus every FBW_* of fbw.py.
FBW_CPP=1 runs the same search in the C++ kernel (../kernel/fbw_fast.FlyByWireCpp; build it first with ../kernel/build.sh). In exact
mode the kernel reproduces fbw.py's costs to 1e-9, so decisions are the same and only the time per gated step changes.
"""
import os, time
import numpy as np
from fbw import FlyByWire

CFG2 = dict(sig=os.environ.get("FBW2_SIG", "epi"), thr=float(os.environ.get("FBW2_THR", 0.0136)),
            latch=int(os.environ.get("FBW2_LATCH", 10)), release=int(os.environ.get("FBW2_RELEASE", 0)),
            calm=int(os.environ.get("FBW2_CALM", 10)))


class GatedFBW:
    def __init__(self, sim, dof_names, feet, cfg2=None, student=None):
        self.c2 = dict(CFG2, **(cfg2 or {}))
        if os.environ.get("FBW_CPP") == "1":                                    # same search, look-ahead in C++ (student weights run in the kernel)
            from fbw_fast import FlyByWireCpp
            self.fbw = FlyByWireCpp(sim, dof_names, feet, student=student)
        else:
            self.fbw = FlyByWire(sim, dof_names, feet)
        self.c = dict(self.fbw.c, **{"fbw2_" + k: v for k, v in self.c2.items()})
        self.hold, self.active, self.calm = 0, False, 0
        self.log = dict(sig=[], gate=[], inter=[], viol=[], t_fbw=[])

    def filter(self, d, x, T_pilot, pilot, sig):
        """sig = the MW student's warning signal at this state. Returns (targets, gate_open, intervened)."""
        warn = sig > self.c2["thr"]
        if warn:
            self.hold, self.active, self.calm = self.c2["latch"], True, 0
        gate = self.active if self.c2["release"] else self.hold > 0
        t0 = time.perf_counter()
        if gate:
            self.hold -= 1
            T, inter = self.fbw.filter(d, x, T_pilot, pilot)
            viol = self.fbw.stats["pred_viol_pilot"][-1]
            if self.c2["release"]:                                              # follow-through: hand back only once stable
                self.calm = self.calm + 1 if (not inter and not warn) else 0
                if self.calm >= self.c2["calm"] and self.hold <= 0:
                    self.active = False
        else:
            self.fbw.prev *= 0.5                                                # same warm-start decay as fbw.py's hands-off path
            T, inter, viol = T_pilot, False, np.nan
        self.log["t_fbw"].append(time.perf_counter() - t0)
        self.log["sig"].append(float(sig)); self.log["gate"].append(gate); self.log["inter"].append(bool(inter)); self.log["viol"].append(viol)
        return T, gate, inter

    def reset(self):
        self.hold, self.active, self.calm = 0, False, 0
        self.fbw.reset()
