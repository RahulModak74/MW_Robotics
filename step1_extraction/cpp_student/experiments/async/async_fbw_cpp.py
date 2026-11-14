#!/usr/bin/env python3
"""async_fbw_cpp.py — the deployment-side async supervisor, C++ worker.

Same contract as results/ASYNC_APPROACH/async_fbw.py, but the worker runs FlyByWireCpp
(cpp/fbw_fast.py), whose `_score` H-loop is the native fbw_kernel. The kernel releases the GIL for
the whole rollout (fbw_kernel.cpp: `py::gil_scoped_release`), so the worker's correction search runs
TRULY PARALLEL to the Python control heartbeat — not merely interleaved under the GIL.

Why this matters (measured, full clock, i5-12450H `performance`):
  * exact correction search: ~260 ms idle / ~374 ms under a live 20 ms heartbeat (6 threads),
    vs ~3.7 s for the Python fbw worker -> first-solve latency drops ~185 control steps -> ~19.
  * the heartbeat NEVER exceeds ~3 ms while the worker runs (async_cpp_proof.py).
  * 19-step (~0.4 s) latency sits inside most of the 18 rescue warning leads (26-200 steps),
    where the 185-step Python latency did not -> the async correction now lands in time.

Contract (identical to AsyncFBW):
    sup = AsyncFBWCpp(sim, dof_names, feet, student=student); sup.set_pilot(student_fn)
    # each control step (main loop, <20 ms):
    sup.submit(d, obs, T_pilot)          # non-blocking; snapshots state for the worker
    delta, w = sup.latest(max_age_s=0.4) # non-blocking; w decays 1->0 over max_age_s
    T = clip(T_act + w*delta)            # act_track heartbeat, biased by the (stale) correction

Env: FBW_THREADS (worker rollout threads; keep modest — the quiet tick is single-rollout and does
not parallelize, and oversubscription vs the main thread hurts). PYTHONPATH must include cpp/.
"""
import os
import threading
import time
import numpy as np
import mujoco
from fbw_fast import FlyByWireCpp


class AsyncFBWCpp:
    def __init__(self, sim, dof_names, feet, cfg=None, student=None):
        if student is None:
            raise ValueError("AsyncFBWCpp needs the student (its weights run inside the kernel)")
        self.fbw = FlyByWireCpp(sim, dof_names, feet, cfg, student=student)
        self.m = self.fbw.m
        self.spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
        self.ns = mujoco.mj_stateSize(self.m, self.spec)
        self._cv = threading.Condition(threading.Lock())
        self._pending = None          # (state, obs, T_pilot)
        self._result = None           # (delta[29], t_done, solve_s, intervened)
        self._busy = False
        self._pilot_fn = None
        self._stop = False
        self._thr = threading.Thread(target=self._run, daemon=True); self._thr.start()

    def set_pilot(self, fn): self._pilot_fn = fn

    @property
    def busy(self):
        with self._cv:
            return self._busy or self._pending is not None

    def submit(self, d, obs, T_pilot):
        """Non-blocking. Snapshot the live state on the calling thread; queue one job (replacing any
        un-started job — only the freshest warning state is wanted)."""
        s = np.empty(self.ns); mujoco.mj_getState(self.m, d, s, self.spec)
        with self._cv:
            self._pending = (s, np.asarray(obs).copy(), np.asarray(T_pilot).copy())
            self._cv.notify()

    def latest(self, max_age_s=0.4):
        """Non-blocking. (delta, weight); weight decays linearly to 0 over max_age_s, then (0, 0)."""
        with self._cv:
            r = self._result
        if r is None:
            return np.zeros(29), 0.0
        delta, t_done, _, _ = r
        age = time.perf_counter() - t_done
        if age >= max_age_s:
            return np.zeros(29), 0.0
        return delta, 1.0 - age / max_age_s

    def _run(self):
        while True:
            with self._cv:
                while self._pending is None and not self._stop:
                    self._cv.wait()
                if self._stop:
                    return
                job = self._pending; self._pending = None; self._busy = True
            s, obs, T_pilot = job
            d = mujoco.MjData(self.m); mujoco.mj_setState(self.m, d, s, self.spec); mujoco.mj_forward(self.m, d)
            t0 = time.perf_counter()
            try:
                T_fbw, inter = self.fbw.filter(d, obs, T_pilot, self._pilot_fn)
                delta = (T_fbw - T_pilot) if inter else np.zeros(29)
            except Exception:
                import traceback; traceback.print_exc(); delta = np.zeros(29); inter = False
            with self._cv:
                self._result = (delta, time.perf_counter(), time.perf_counter() - t0, bool(inter)); self._busy = False

    def reset(self):
        with self._cv:
            self._pending = None; self._result = None
        self.fbw.reset()

    def close(self):
        with self._cv:
            self._stop = True; self._cv.notify()
