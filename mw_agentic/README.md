# mw_agentic — a standalone agentic layer for the robot demo

Mirrors `~/DASHBOARD_FACTORY/analytics-harness-1.7`'s design (read in full before building this -
`AGENTS.md`, `DESIGN.md`, `ah/agent.py`, `ah/rl.py`), as a **standalone** package inside
`MW_Robotics` - no dependency on, or changes to, the live harness or its auth/secrets.

## The mapping

| analytics-harness-1.7 | mw_agentic | Role |
|---|---|---|
| `ah/agent.py` (~140 lines) | `agent.py` | the loop: query -> parse one fenced block -> validate -> retry |
| `agents/*.md` (front matter + sections) | `agents/choreographer.md` | one json-action agent: draft a candidate, nothing else |
| `tools/*.md` | the schema block in `choreographer.md`'s System section | what the model is told it can do, and the valid ranges |
| `bi.py check` (the json agent's `submit` validator) | `choreography.py::validate()` | deterministic, exit-code-shaped accept/reject |
| `ah/rl.py::Policy` (Thompson sampling, Beta bandit) | `policy.py::Policy` | same math, ported close to unchanged |
| `domains/*/recipes/*.md` (`fixes:`, hand-written) | `recipes/*.md` (`keywords:`, hand-written) | a developer's template for a known request shape |
| a repair auto-recorded after a successful fix | `recipes/learned/<signature>.json` (auto-saved) | a request the LLM solved once, replayed verbatim next time |
| `PolicyModel` (no-LLM driver, tries ranked recipes first) | `orchestrate.py::choreograph()` | the same default: recipe first, LLM only when nothing works |

## What "minimum work for the LLM" means here, concretely

`orchestrate.choreograph(task)`:
1. Compute a task **signature** (sorted bag-of-words - the cheap, honest baseline; a real
  embedding-based near-duplicate match is the natural upgrade, not built here).
2. If any recipe (hand-written or previously learned) is relevant, **Thompson-sample a ranking**
  over them and try the top one: render it, validate it. **Zero model calls.** This is not gated
  behind a minimum number of past successes - the harness's own `PolicyModel` does not gate
  either; it always tries the best-ranked candidate first, the way a flat `Beta(1,1)` prior
  naturally explores on the first try.
3. If that fails (or nothing matched), try the next-ranked candidate, and so on.
4. Only once every candidate is exhausted does it call a model (`model.pick_model()`:
  the `claude` CLI first - zero config on this machine or any Claude-Code-based run - falling
  back to a raw `ANTHROPIC_API_KEY` call for portability).
5. If the model's draft validates, it is **saved as a new learned recipe**, keyed by the task
  signature. The next identical (or bag-of-words-identical) request is then answered by step 2,
  with no model call at all - proven end to end: see `workspace/rl/policy.json` and
  `recipes/learned/` after running the two commands below.

## Try it

```bash
cd MW_Robotics
python3 -m mw_agentic "create a dancing robot pair"      # matches recipes/mirror_dance.md - 0 model calls
python3 -m mw_agentic "robot a does a high five with robot b"   # no recipe matches - one model call,
                                                                 # then saved; run the SAME command again
                                                                 # and it costs 0 model calls the second time
```

## What is real here, and what is a stated simplification

- **The validator is the real safety gate.** Every parameter range in `choreography.py` matches
  what the corresponding `mw_framework/domains/*.py` was actually trained on - a request the
  model (or a hand-written recipe) produces outside that range is rejected before it ever
  reaches a manifold, the same role `bi.py check` plays for a dashboard layout.
- **The bandit is the harness's own math**, not a simplified stand-in: `policy.py::Policy` is a
  close port of `ah/rl.py::Policy` (same Beta-bandit, same `rank()`/`update()`/`preventive()`
  shape).
- **The task-signature matching is deliberately simple** (bag-of-words), not semantic. Two
  genuinely different requests that happen to share the same words will collide; two near-
  identical requests phrased very differently will not match. Documented, not hidden - the
  harness's own exact-signature matching for error codes has the same deliberate simplicity for
  the same reason (cheap, deterministic, auditable).
- **No `pause`/`sign` step.** The harness's domain flows include a human-review pause before
  anything ships to a client. This demo has no such step - a validated choreography plays
  immediately. Adding one (review the resolved frames before they animate) is the natural next
  piece if this stops being a demo.
- **Recipe parameter filling is defaults only**, not NLP extraction from the task text (e.g.
  "dance for 20 seconds" does not currently change `duration_s`). Noted as the clear next
  extension, not built here.

## Files

```
agentmd.py      parse .md (front matter + sections) + {{var}} rendering - a small subset of
                the harness's ah/agentmd.py
choreography.py  the schema and the deterministic validator
model.py          ClaudeCodeModel (claude -p) and AnthropicAPIModel (ANTHROPIC_API_KEY), tried in that order
agent.py          the json-agent loop: draft -> validate -> retry (bounded)
policy.py         the Thompson-sampling bandit, ported from ah/rl.py
recipes.py        recipe lookup (hand-written + auto-learned) and rendering
orchestrate.py    choreograph(task): the recipe-first, LLM-as-fallback decision
cli.py / __main__.py   python -m mw_agentic "<task>"
agents/choreographer.md   the one agent doc
recipes/*.md               hand-written recipe templates (mirror_dance, wave_hello)
recipes/learned/*.json    auto-saved, git-ignored - a fresh clone starts with nothing learned
workspace/rl/             policy.json + experience.jsonl - git-ignored, local learning state
```
