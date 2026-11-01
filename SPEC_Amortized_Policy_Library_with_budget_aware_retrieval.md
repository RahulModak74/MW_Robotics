Spec an agent can implement directly. It is written to be **tunable**, not final — defaults are marked `DEFAULT` and every one is a config knob. The two honest constraints are stated up front because they shape the architecture: (1) online refinement needs an **evaluator** (a simulator), and (2) CEM/ES cost scales with **parameter count**, so the library must store **low-dimensional** policies and the refinement must be **local**.
#Rahul changed Nov 2
---

# SPEC: Amortized Policy Library with Budget-Aware Retrieve-and-Refine (APL-RR)

## 0. Scope and non-goals

**Goal.** Ship a runtime that, given a *context* (body + task + environment), produces a policy by:
- **CPU tier:** nearest-neighbor retrieval from a precomputed policy library, executed as-is.
- **GPU tier:** retrieval followed by a bounded CEM/ES refinement against a local simulator, then deploy.

**Non-goals.**
- Not a posterior sampler. The stored Gaussian is a **search distribution**, not calibrated uncertainty.
- Not training from scratch at runtime. Refinement is **local**, bounded, and abortable.
- Not hardware-in-the-loop by default (see §9, Mode H is gated).

**Non-negotiable design rule.** All library policies are **low-dimensional** (linear or ≤ ~1k params). This is what makes CPU-tier retrieval viable and keeps refinement's population small. Do not allow a 10⁶-weight net into the library; it breaks both tiers.

---

## 1. Glossary and symbols

| Symbol | Meaning |
|---|---|
| `c` | context vector (normalized body/task/env descriptor) |
| `θ` | policy parameter vector (flattened) |
| `μ, σ` | CEM search mean and per-dimension std (diagonal) |
| `J(θ)` | scalar score = mean survival steps over `R` fixed seeds |
| `n, k, ρ` | population size, elite count, elite fraction (`k=⌈ρn⌉`) |
| `B` | refinement budget, expressed in **rollouts** (not seconds) |
| `rps` | measured rollouts/sec on the host (calibration, §7) |

Budget is always counted in **rollouts**, then converted to wall-clock via `rps`. Never budget in seconds directly — it hides host variance.

---

## 2. Library artifact format

### 2.1 Container

One directory per library version. One file per entry.

```
lib/<lib_id>/
  manifest.json
  entries/
    <entry_id>.pol        # binary: header + float32 blobs
    <entry_id>.json       # metadata sidecar (also embedded in .pol header)
  INDEX.faiss / INDEX.npz # prebuilt context index
  PROVENANCE.json
```

### 2.2 `.pol` header (JSON, then raw arrays)

Required fields:

```json
{
  "entry_id": "str",
  "lib_id": "str",
  "schema": {"obs": "hash", "act": "hash", "body": "hash", "task": "hash"},
  "policy_family": "linear" | "mlp",
  "arch": {"layers": [96, 32, 6], "activation": "tanh", "out_scale": 1.0},
  "param_count": 96,
  "params": {"offset": 0, "dtype": "float32", "shape": [96]},
  "context": {"offset": 0, "dtype": "float32", "shape": [D_c]},
  "search_dist": {
    "mu": {"offset": 96, "shape": [96]},
    "sigma": {"offset": 192, "shape": [96]},
    "sigma_floor": 1e-3,
    "generations_run": 14,
    "converged": false
  },
  "expected_score": {"J_mean": 208.3, "J_p10": 171.0, "episodes": 20, "cap": 1000},
  "eval_protocol": {"seeds": [...], "sim_commit": "git:abc123", "cap": 1000},
  "sha256": "..."
}
```

**Rule L1.** `search_dist.mu/sigma` is **mandatory**. Storing the final CEM Gaussian is the whole point: it gives refinement a warm start *and* gives retrieval a Mahalanobis metric and a spread-based confidence. Do not store only the argmax.

**Rule L2.** `schema` hashes must match exactly at retrieval time. A schema mismatch is a hard reject, not a warning.

**Rule L3.** `eval_protocol.sim_commit` must match the runtime simulator's commit, or the entry is usable **only** for retrieval, never as a refinement warm start (drifted dynamics make warm-started σ meaningless).

---

## 3. Context vector `c`

### 3.1 Definition

`c` is a fixed-length, normalized descriptor of the query situation. Minimum fields:

```python
c = concat([
    body_signature,        # joint count, link masses (log), lengths, damping, gear  -> normalized
    task_descriptor,       # target velocity, terrain class one-hot, horizon, reward weights
    env_descriptor,        # gravity, friction, latency estimate, actuator model id hash-embed
])
```

### 3.2 Normalization

Fit `mean_c, std_c` on the library's own contexts. **Freeze** them into the library. Runtime uses the frozen stats. Never refit at runtime.

**Rule C1.** `c` must not contain anything the runtime cannot observe before the first action. No leakage from the episode outcome.

**Rule C2.** Context dimensionality `D_c` ≤ 64. If you exceed this, you are encoding too much; use a hash-embedding for categoricals and drop the rest.

---

## 4. Retrieval module

```python
def retrieve(c_query, lib, k=8, metric="mahalanobis") -> RetrievalResult
```

### 4.1 Distance

- `DEFAULT metric = "mahalanobis"` on context: `d² = (c_q − c_i)ᵀ M⁻¹ (c_q − c_i)`, with `M` the context covariance estimated from the library (regularized: `M + λI`, `λ = 1e-3·trace(M)/D_c`).
- Fallback `metric="l2"` after standardization.
- **Tie-break** by higher `expected_score.J_p10` (prefer robust entries, not just high-mean ones).

### 4.2 Output

```python
RetrievalResult = {
  "entry": <pol header + params>,
  "distance": float,
  "confidence": "high" | "low",   # see C3
  "alternates": [entry_ids],      # k-1 nearest, for fallback
}
```

**Rule C3 (confidence gate).** `confidence = "high"` iff `d ≤ d_hi` **and** the top-1 entry's `expected_score.J_p10` exceeds `J_min_safe`. Otherwise `"low"`. `d_hi` is the 90th percentile of leave-one-out nearest-neighbor distances within the library. A `low` retrieval **must not** be executed on hardware (see §9) and must be flagged in telemetry.

**Rule C4 (no silent nearest-neighbor).** If `confidence == "low"` and no refinement budget exists, fall back to `fallback_controller` (the vetted baseline, e.g. the hand-tuned PD), and log `reason="low_conf_no_budget"`. Never ship a low-confidence nearest-neighbor as if it were fine.

---

## 5. Refinement module (CEM/ES)

```python
def refine(init: RetrievalResult, evaluator, budget_rollouts: int, cfg: RefineCfg) -> RefineResult
```

### 5.1 Initialization (warm start — this is the core trick)

```python
mu    = init.entry.search_dist.mu          # or init.params if mu absent
sigma = clip(alpha * init.entry.search_dist.sigma, sigma_min, sigma_max)
# DEFAULT alpha = 0.5   (shrink stored spread; the retrieved policy is already locally good)
# DEFAULT sigma_min = 1e-3, sigma_max = 0.5 * (per-dim scale of mu)
```

Rationale: the stored `sigma` is the width the offline search needed at convergence. Halving it gives a **local** search that respects the offline result instead of re-exploring globally. This is why refinement is cheap.

If `confidence == "low"`, use `alpha = 1.0` and `sigma = sigma_default = 0.1 * ones` (broader, because you're in an unvalidated region) — but only if the caller explicitly allows it (`allow_broad_refine=True`).

### 5.2 Loop

```python
DEFAULT: n = 32, rho = 0.2, k = ceil(rho*n) = 7, R_seeds = 3, cap = 1000,
         sigma_smooth = 0.7, sigma_floor = 1e-3, patience = 3, min_gen = 2

best = None; best_J = -inf; stall = 0
while rollouts_used + n*R_seeds <= budget_rollouts:
    eps = randn(n, D)
    pop = mu + sigma * eps                       # diagonal; see 5.3 for rank-1 option
    Js  = [mean_survival(evaluator, theta, seeds=R_seeds, cap=cap) for theta in pop]
    rollouts_used += n*R_seeds

    elites = top_k(pop, Js, k)
    mu_new    = mean(elites, axis=0)
    sigma_new = std(elites, axis=0)
    sigma = clip(sigma_smooth*sigma + (1-sigma_smooth)*sigma_new, sigma_floor, sigma_max)
    mu = mu_new

    if max(Js) > best_J + eps_improve: best_J, best = max(Js), pop[argmax(Js)]; stall = 0
    else: stall += 1

    if gen >= min_gen and (stall >= patience or max(sigma) <= sigma_floor): break
    gen += 1

# ALWAYS re-evaluate the winner with more seeds before returning.
final = mean_survival(evaluator, best, seeds=10*R_seeds, cap=cap)
```

### 5.3 Covariance policy

- `DEFAULT: diagonal only.` Full covariance costs `O(D²)` samples; at `D=96` that's already 4× the diagonal population for marginal gain. Do not enable full covariance unless `D ≤ 32`.
- Optional `rank-1` update (sep-CMA-lite) behind a flag; measure before adopting.

### 5.4 Budget floor

**Rule B1.** Refinement is only attempted if `budget_rollouts ≥ min_budget = 8 * n * R_seeds` (DEFAULT ≈ 768). Below that, retrieval-only. A half-run CEM is worse than no CEM — it can return a policy that was merely lucky on 3 seeds.

**Rule B2.** Always reserve `10*R_seeds` rollouts for the final re-evaluation. Don't let the loop consume the whole budget.

### 5.5 Output

```python
RefineResult = {
  "params": best, "J_final": final, "J_init": evaluated_init_score,
  "gain": final - J_init, "rollouts_used": int, "generations": int,
  "converged": bool, "sigma_final": float, "aborted": bool, "abort_reason": str|None
}
```

**Rule B3 (accept gate).** Adopt refined params **only if** `gain ≥ g_min` (DEFAULT `g_min = 5% of J_init`, and `≥ 10 steps` absolute) **and** `J_final ≥ J_init`. Otherwise discard and keep the retrieved policy, logging `reason="no_gain"`. This prevents CEM noise from overwriting a good offline policy.

---

## 6. Runtime tier scheduler

```python
def decide_tier(host_profile, budget_wallclock_s, retrieval) -> Plan
```

### 6.1 Tiers

| Tier | Hardware | Default behavior | Budget source |
|---|---|---|---|
| T0 | CPU-only, no sim | retrieval only | none |
| T1 | CPU-only, sim available | retrieval + **tiny** CEM (`n=16`, `budget ≤ 512 rollouts`) | leftover CPU time |
| T2 | ≥1 GPU, sim on GPU | retrieval + full CEM per §5 | `budget_wallclock_s` |
| T3 | many GPUs / batch | retrieval + **parallel** CEM (population across workers), multiple seeds | queued |

### 6.2 Decision rule

```python
rps = calibrate_rollouts_per_sec(host_profile, evaluator)      # §7
budget_rollouts = floor(budget_wallclock_s * rps * safety_margin)  # DEFAULT safety_margin = 0.8
if budget_rollouts < min_budget:            plan = RETRIEVE_ONLY
elif tier in {T1}:                          plan = REFINE(n=16, budget=budget_rollouts)
else:                                       plan = REFINE(n=32, budget=budget_rollouts)
if retrieval.confidence == "low" and plan != RETRIEVE_ONLY: plan.alpha = 1.0
```

**Rule S1.** The scheduler must be able to answer "what would retrieval-only have scored?" without running refinement — i.e. `J_init` comes from the library's `expected_score`, **not** from a fresh roll-out, when no sim is available. When a sim *is* available, measure `J_init` once (cheap) and use the measured value.

**Rule S2.** Refinement must be **interruptible**: checkpoint `(mu, sigma, gen, best)` every generation so a preempted job resumes rather than restarts.

---

## 7. Calibration

```python
def calibrate_rollouts_per_sec(host, evaluator, n_probe=20) -> float
```

Run `n_probe` rollouts of a fixed policy with `cap=1000`, measure wall-clock, return median `rps`. Cache per `(host_hash, sim_commit, cap)`. Re-measure if `> 24h` old or sim commit changed. **Do not** hardcode `rps`.

---

## 8. Evaluation protocol (must be identical across library build and runtime)

**Rule E1.** Scoring uses `K` fixed evaluation seeds, disjoint from any training seed, `DEFAULT K=20`, `cap=1000`. Report `J_mean` and `J_p10` (10th percentile across seeds). Never report a single-seed number.

**Rule E2.** Held-out context split: build the library on contexts `C_train`, report retrieval quality on `C_heldout`. No entry may be indexed by a context it was tested on.

**Rule E3.** Any number in a README/table must cite `sim_commit` and `eval_protocol`. The 208.3 vs 263 vs 15.9 table is only meaningful with those attached.

---

## 9. Execution modes (safety)

| Mode | Evaluator | Allowed? |
|---|---|---|
| **S (sim)** | local simulator | **DEFAULT.** Refine here, deploy weights. |
| **O (offline/batch)** | queued GPU sim | allowed; deploy next session |
| **H (hardware-in-loop)** | the real body | **gated.** Requires: (a) `confidence=="high"`, (b) a fallback controller in a supervisory loop that can take over within one control period, (c) a bounded action-space governor, (d) `n ≤ 8`, `sigma ≤ 0.02·scale(mu)`, `cap` short, (e) explicit operator enable flag. No broad refinement on hardware, ever. |

**Rule H1.** Hardware refinement may only make **small local perturbations** and must be able to revert to the retrieved policy at any time. If the safety governor trips, revert and abort the job.

---

## 10. The gating experiment (run this before building the runtime)

This is the dose-response curve that decides the library's max parameter count.

**E-GATE.** On Walker2d (or the available contact-rich body), sweep policy size:

| Variant | Params (approx) |
|---|---|
| linear | ~96–150 |
| MLP 1×32 tanh | ~250 |
| MLP 1×64 | ~500 |
| MLP 2×64 | ~1000 |
| MLP 2×256 | ~4000 |

For each: fixed `n`, `rho`, `R_seeds`, `cap`; run CEM to a **rollout budget** `B_max` (DEFAULT 200k rollouts); record (a) rollouts-to-threshold, (b) final `J_mean`, (c) whether the population mean was still climbing at `B_max`, (d) variance across 5 seeds.

**Decision rule (write it down before running):**
- If `J_mean` holds within 90% of the linear result up to **1000 params** → library limit = 1000; T1 CPU refinement is viable.
- If it degrades beyond **~400 params** → library limit = 400; T1 is retrieval-only; refinement becomes GPU-only.
- If it degrades beyond **~150 params** → this family is a simple-body tool; reconsider before committing to a 20+-joint target.

Publish the curve. It's the single most informative artifact here, more than any single policy's score.

---

## 11. Acceptance tests (CI)

| ID | Test | Pass criterion |
|---|---|---|
| A1 | Schema mismatch rejected | hard reject, no execution |
| A2 | Retrieval determinism | same `c`, same lib → identical entry |
| A3 | Leave-one-out retrieval recall | retrieved `J_mean` ≥ 80% of the held-out entry's own `J_mean`, median over `C_heldout` |
| A4 | Refinement gain | on `C_heldout`, median `gain ≥ 5%`, and `gain > 0` in ≥ 70% of cases |
| A5 | No-gain gate | when injected with a deliberately bad init, gate discards and keeps init ≥ 95% of runs |
| A6 | Budget accounting | `rollouts_used ≤ budget` always; final re-eval reserved |
| A7 | Interrupt/resume | kill mid-run, resume reproduces same `(mu,sigma)` within fp tolerance |
| A8 | Fallback path | `confidence=="low"` + no budget → fallback controller, logged |
| A9 | Mode H lockout | without enable flag, any hardware path raises |
| A10 | `rps` calibration | scheduler never uses a stale/hardcoded `rps` |

---

## 12. Deliverables and order of work

1. **`lib/` schema + writer/reader** (§2), with schema hashing and `search_dist` round-trip. *(blocks everything)*
2. **Evaluator contract** (§8, E1) + `rps` calibration (§7).
3. **Retrieval** (§3–4) + leave-one-out recall harness (A3).
4. **Refinement** (§5) with warm start, accept gate, checkpointing (A4–A7).
5. **Scheduler + tiers** (§6) + fallback (A8).
6. **E-GATE** (§10) — run and publish; use the result to set `param_count` limits in §0 and §2.
7. **Mode H safety governor** (§9) — only after 1–6 are green.

---

## 13. Two things to keep straight

- **The stored `sigma` is not uncertainty.** It's the search width. Using it as a Mahalanobis metric and as a warm-start scale is legitimate engineering; reporting it as a posterior is not. If you want calibration, you need a separate held-out reliability estimate — `J_p10` over `K` seeds is the honest proxy, not `sigma`.
- **Refinement is local or it is nothing.** The whole reason this architecture is cheap is that the library has already paid the global search cost and stored the result *as a distribution*. If refinement starts from a broad σ, it's re-doing the offline job at runtime with a worse budget. Guard `alpha` and `sigma_max` accordingly.

If you want, I can turn §2–§5 into actual module stubs (`library.py`, `retrieve.py`, `cem.py`, `scheduler.py`) with the interfaces above wired and the acceptance tests as pytest skeletons.
