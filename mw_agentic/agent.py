"""
mw_agentic.agent
===================
The json-agent loop: query the model, extract exactly one ```json block,
validate it deterministically, retry on failure with the validator's own
error text (same shape as analytics-harness-1.7's json-agent pattern:
"the environment is the validator; exit 0 accepts, else the message goes
back to the model"). No bash, no tool-calling here - the choreographer's
only action is "submit a candidate".
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from . import agentmd, choreography

HERE = Path(__file__).resolve().parent
JSON_BLOCK = re.compile(r"```json\s*\n(.*?)```", re.S)


def run_choreographer(model, task: str, step_limit: int = 4) -> dict:
    """Returns {"ok": bool, "spec": dict|None, "errors": list, "steps": int,
    "model": str}. Never raises - a model or validation failure is reported,
    not an exception the caller has to catch."""
    doc = agentmd.load(HERE / "agents" / "choreographer.md")
    system = agentmd.render(doc.section("system"), task=task)
    user = agentmd.render(doc.section("task"), task=task)
    limit = int(doc.meta.get("step_limit", step_limit))

    history_note = ""
    for step in range(1, limit + 1):
        prompt = user if step == 1 else f"{user}\n\n{history_note}"
        try:
            reply = model.query(system, prompt)
        except Exception as e:                      # model unavailable or errored mid-run
            return {"ok": False, "spec": None, "errors": [f"model error: {e}"], "steps": step, "model": model.name}

        m = JSON_BLOCK.search(reply)
        if not m:
            history_note = ("VALIDATION_ERROR: your reply did not contain exactly one ```json block. "
                            "Reply with ONLY the JSON object in one ```json fenced block.")
            continue
        try:
            spec = json.loads(m.group(1))
        except json.JSONDecodeError as e:
            history_note = f"VALIDATION_ERROR: not valid JSON ({e}). Fix and resend the whole object."
            continue

        errors = choreography.validate(spec)
        if not errors:
            return {"ok": True, "spec": spec, "errors": [], "steps": step, "model": model.name}
        history_note = "VALIDATION_ERROR:\n" + "\n".join(f"- {e}" for e in errors) + "\nFix and resend the whole object."

    return {"ok": False, "spec": None, "errors": [history_note or "step limit reached"],
           "steps": limit, "model": model.name}
