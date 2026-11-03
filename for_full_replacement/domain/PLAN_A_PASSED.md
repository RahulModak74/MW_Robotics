rahul@rahul-LOQ-15IRH8:~/MW_Robotics/domain$ ./run_plan_a.sh --generations 30 --pop 80
2026-11-01T20:38:41Z [plan-a] repo root : /home/rahul/MW_Robotics
2026-11-01T20:38:41Z [plan-a] log file  : /home/rahul/MW_Robotics/domain/logs/plan_a_20261101_203841.log
2026-11-01T20:38:41Z [plan-a] config    : pop=80 gens=30 max_steps=400 restarts=1 held_out=20
2026-11-01T20:38:51Z [plan-a] Python + deps present
2026-11-01T20:38:51Z [plan-a] MW_DEVICE=cpu, CUDA_VISIBLE_DEVICES='' (CPU-only by design)
2026-11-01T20:38:51Z [plan-a] step 1/4 — CEM search (1 restart(s)) + held-out eval
2026-11-01T20:38:51Z [plan-a]             this is the long step (~90s CPU, serial)

[search] pop=80 gens=30 max_steps=400 restarts=1 held_out=20 dim=96

[restart 1/1] running CEM (seed=0)…
  gen  0  pop_mean=  16.1  pop_best=  36.0  std_norm=4.497
  gen  1  pop_mean=  24.9  pop_best=  95.0  std_norm=4.177
  gen  2  pop_mean=  41.2  pop_best= 138.5  std_norm=3.929
  gen  3  pop_mean=  55.8  pop_best= 170.5  std_norm=3.655
  gen  4  pop_mean=  83.2  pop_best= 228.5  std_norm=3.510
  gen  5  pop_mean=  96.1  pop_best= 205.5  std_norm=3.544
  gen  6  pop_mean= 125.3  pop_best= 305.5  std_norm=3.399
  gen  7  pop_mean= 150.0  pop_best= 333.0  std_norm=3.097
  gen  8  pop_mean= 159.2  pop_best= 300.5  std_norm=2.908
  gen  9  pop_mean= 165.8  pop_best= 366.5  std_norm=2.789
  gen 10  pop_mean= 194.6  pop_best= 400.0  std_norm=2.717
  gen 11  pop_mean= 195.2  pop_best= 400.0  std_norm=2.614
  gen 12  pop_mean= 196.7  pop_best= 323.0  std_norm=2.373
  gen 13  pop_mean= 201.0  pop_best= 400.0  std_norm=2.236
  gen 14  pop_mean= 241.2  pop_best= 400.0  std_norm=2.229
  gen 15  pop_mean= 248.8  pop_best= 400.0  std_norm=2.077
  gen 16  pop_mean= 272.8  pop_best= 400.0  std_norm=2.090
  gen 17  pop_mean= 316.9  pop_best= 400.0  std_norm=1.960
  gen 18  pop_mean= 310.9  pop_best= 400.0  std_norm=1.857
  gen 19  pop_mean= 326.1  pop_best= 400.0  std_norm=1.900
  gen 20  pop_mean= 315.3  pop_best= 400.0  std_norm=1.741
  gen 21  pop_mean= 310.9  pop_best= 400.0  std_norm=1.612
  gen 22  pop_mean= 321.3  pop_best= 400.0  std_norm=1.460
  gen 23  pop_mean= 334.9  pop_best= 400.0  std_norm=1.450
  gen 24  pop_mean= 349.8  pop_best= 400.0  std_norm=1.363
  gen 25  pop_mean= 332.6  pop_best= 400.0  std_norm=1.212
  gen 26  pop_mean= 358.9  pop_best= 400.0  std_norm=1.125
  gen 27  pop_mean= 358.7  pop_best= 400.0  std_norm=1.101
  gen 28  pop_mean= 360.2  pop_best= 400.0  std_norm=1.068
  gen 29  pop_mean= 347.6  pop_best= 400.0  std_norm=1.003
[restart 1] search 1126.4s, |theta|=4.401
[restart 1] held-out mean (20 seeds) = 920.9 steps

[save] best restart = #1 (held-out 920.9), saved -> /home/rahul/MW_Robotics/cem_best_theta.npy
[random seed=999] held-out mean = 14.3 steps
[random seed=1000] held-out mean = 6.0 steps
[random seed=1001] held-out mean = 7.0 steps

[summary] wrote /home/rahul/MW_Robotics/cem_summary.json
2026-11-01T20:57:49Z [plan-a] step 2/4 — reading summary
2026-11-01T20:57:49Z [plan-a] CEM best held-out: 920.9 steps
2026-11-01T20:57:49Z [plan-a] Random baseline : 9.116666666666667 steps
2026-11-01T20:57:49Z [plan-a] step 3/4 — thresholds: pass>=210.4, strong>=249.8, reference_repro=[187.5, 229.1]
2026-11-01T20:57:49Z [plan-a] step 4/4 — result

════════════════════════════════════════════════════════════════
  PLAN A — RESULT: PASS
════════════════════════════════════════════════════════════════
  CEM best held-out  : 920.9 steps
  Random baseline    : 9.116666666666667 steps
  PD baseline        : 263 steps
  Reference (earlier): 208.3 steps
  Reproducibility    : outside 10% of reference (run more restarts to diagnose)

  STRONG — CEM matches or beats 95% of PD baseline.
  This is the result to compare V10 against. Plan B's fixed
  variant must beat THIS number to be worth shipping.

  Saved theta   : /home/rahul/MW_Robotics/cem_best_theta.npy
  Full summary  : /home/rahul/MW_Robotics/cem_summary.json
  Full log      : /home/rahul/MW_Robotics/domain/logs/plan_a_20261101_203841.log
════════════════════════════════════════════════════════════════

