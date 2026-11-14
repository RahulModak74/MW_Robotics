**Headline: r7c + FBW_2 with follow-through = 58/58, with the physics look-ahead on 21 % of steps and a 2.6 ms median decision on CPU (D: 58/58, look-ahead every step, 3.1 s median).**

# FBW_2: the MW student's own fall warning decides when the flight computer acts (10 Nov 2026)

Code: [`../gate/fbw_2.py`](../gate/fbw_2.py), [`../gate/pm_eval_fbw2.py`](../gate/pm_eval_fbw2.py), [`../scripts/run_gated_58.sh`](../scripts/run_gated_58.sh). Timings in this file are the Python search.
Pilot `r7c_mwv3ctl_s100`; the search inside the gate is config D unchanged (`fbw.py` untouched). CPU MuJoCo only.

## The gate

- **Signal:** the student's epistemic uncertainty `epi`, from the Bayesian last layer.
- **Threshold:** `epi` > 0.0136, the 95th percentile on r7c's **training** states, so it was not set on the exam.
- **Quality of the signal on r7c's 58 tests:** AUROC 0.987 for "fall within 25 steps". At the gate threshold it flags 99 % of the steps before falls. For comparison, `mw` distance scores 0.915 and `ale` 0.898.

## Result: 58 tests

| Version | Score | Notes |
|---|---|---|
| Student alone | 40/58 | |
| Config D (physics check every step) | 58/58 | |
| **FBW_2, 10-step latch** | **57/58** | Only Kimodo #5 (`dance_009_00`) fell, at step 368; alone it falls at step 173 |
| **FBW_2 + follow-through release** (owner's fix, found mid-run) | **58/58** (full rerun of all 58 with the fix) | Familiar 21/21, Kimodo 30/30 (incl. #5), dance 7/7. Gate open 21.3 %; median 2.6 ms; 78.7 % of steps within 20 ms |

**What follow-through does:** once MW warns, the flight computer keeps control until D's look-ahead needs no correction **and** MW is quiet, for 10 consecutive steps.

## Speed (CPU, all 58 tests, 21,703 steps)

| Measure | Value |
|---|---|
| Gate open | 16.9 % of steps, so the physics look-ahead is skipped on 83 % of steps (D runs it on 100 %) |
| Median decision time | 2.6 ms |
| Steps decided within 20 ms | 83 % |
| Gated steps | mean 3.2 s; p90 3.4 s and p99 7.7 s over all steps |
| Familiar library | gate open 11.8 % |
| Dance at noise 0.2 | gate open 82–92 %, so there is little saving there |

**What this means:** on a CPU, FBW_2 is real time whenever the warning is quiet, which is most steps. The steps where the warning fires are not real time yet. Next steps:
- `S=1` at noise 0: the 4 noise draws are identical there, so this gives the same decisions at a quarter of the cost.
- A smaller N and H on gated steps.
- Spreading the search over the warning's 0.2–0.8 s lead time.


## Calibration: D always on, same laptop (dance, noise 0, 248 steps)

| | Median per step | Mean per step | Steps within 20 ms |
|---|---|---|---|
| D (gate always open) | 3,125 ms | 1,856 ms | 0 % |
| FBW_2, all 58 tests | **2.6 ms** | **540 ms** | **83 %** |

That is about **1,200× faster on the typical step** and about **3.4× faster on average** (wall clock).
