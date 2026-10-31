"""
mw_agentic.orchestrate
=========================
choreograph(task): the "minimum work for the LLM" decision, mirroring
analytics-harness-1.7's default (no-LLM) repair loop as closely as
possible - NOT gated behind a minimum observation count. The harness's
PolicyModel always Thompson-samples a ranking over whatever recipes exist
for the current error signature and tries the top one; an LLM is only
needed when there is nothing left to try. Applied here:

  1. Any recipe (hand-written or previously auto-learned) for this kind of
     request? Thompson-sample a ranking over them, try the top one -
     render + validate, ZERO model calls. If it validates: done, reward
     the policy, return.
  2. If it does not validate (or no recipe exists at all): try the next
     ranked candidate the same way; if all are exhausted, fall through.
  3. Only then run the choreographer json-agent (one real model call,
     bounded retries, validated deterministically).
  4. If that validates, SAVE it as a new learned recipe and reward the
     policy - so the NEXT identical or near-identical request is answered
     by step 1 with no model call at all.
"""
from __future__ import annotations

from pathlib import Path

from . import agent, choreography, model as model_module, recipes
from .policy import Policy

WORKSPACE = Path(__file__).resolve().parent / "workspace"


def choreograph(task: str, workspace: Path | None = None) -> dict:
    ws = Path(workspace) if workspace else WORKSPACE
    pol = Policy(ws / "rl")
    sig = recipes.task_signature(task)
    candidates = recipes.list_candidates(task, sig)
    tried = []

    if candidates:
        ranked = pol.rank(sig, candidates)
        for name in ranked:
            try:
                spec = recipes.fill(name, task)
                errors = choreography.validate(spec)
            except Exception as e:
                spec, errors = None, [f"recipe {name!r} failed to render: {e}"]
            tried.append(name)
            if not errors:
                pol.update(sig, name, 1.0)
                pol.log(task=task, signature=sig, path="recipe", recipe=name, tried=tried, llm_calls=0, ok=True)
                return {"ok": True, "spec": spec, "path": "recipe", "recipe": name, "tried": tried,
                       "llm_calls": 0, "errors": [], "signature": sig}
            pol.update(sig, name, 0.0)
            pol.log(task=task, signature=sig, path="recipe", recipe=name, llm_calls=0, ok=False, errors=errors)

    # no recipe worked (or none existed) - fall through to the model
    pol.observe_signal(sig)
    try:
        chosen_model = model_module.pick_model()
    except model_module.ModelUnavailable as e:
        return {"ok": False, "spec": None, "path": "llm", "tried": tried, "llm_calls": 0,
               "errors": [str(e)], "signature": sig}

    result = agent.run_choreographer(chosen_model, task)
    if result["ok"]:
        name = recipes.save_learned(sig, result["spec"])
        pol.update(sig, name, 1.0)
        pol.log(task=task, signature=sig, path="llm", recipe=name, llm_calls=result["steps"],
               ok=True, model=result["model"])
    else:
        pol.log(task=task, signature=sig, path="llm", llm_calls=result["steps"], ok=False,
               errors=result["errors"], model=result.get("model"))
    return {**result, "path": "llm", "tried": tried, "llm_calls": result["steps"], "signature": sig}
