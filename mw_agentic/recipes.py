"""
mw_agentic.recipes
=====================
Two kinds of recipe, both matched against a task by simple keyword overlap
(no embeddings, no LLM - deliberately as simple as the harness's own exact
signature matching):

  hand-written    recipes/*.md - a developer writes these after seeing a
                  pattern repeat, same spirit as the harness's recipes/*.md
                  (fixes:/run:). A small {{var}} template filled from
                  declared defaults (param extraction from the task text is
                  a natural extension, not built here - see the README).
  auto-learned    recipes/learned/<signature>.json - saved automatically
                  the first time the LLM path produces a spec that
                  validates for a given task signature. Replaying it next
                  time is the literal "once an acceptable result exists,
                  follow a deterministic approach" behaviour.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from . import agentmd

HERE = Path(__file__).resolve().parent
HAND_DIR = HERE / "recipes"
LEARNED_DIR = HERE / "recipes" / "learned"


def task_signature(task: str) -> str:
    """A simple, deterministic signature: the sorted set of words in the
    task, lowercased. Good enough to recognise a near-identical repeat
    request without an LLM or an embedding model - a real extension would
    cluster near-duplicates, this is the honest, cheap baseline."""
    words = sorted(set(re.findall(r"[a-z]+", task.lower())))
    return " ".join(words)


def _hand_written() -> list:
    return [agentmd.load(p) for p in sorted(HAND_DIR.glob("*.md"))]


def list_candidates(task: str, signature: str) -> list:
    """Names of every recipe (hand-written or learned) plausibly relevant
    to this task: hand-written ones by keyword overlap, learned ones tagged
    with the EXACT same signature (an exact-repeat request)."""
    words = set(re.findall(r"[a-z]+", task.lower()))
    names = []
    for doc in _hand_written():
        kws = set(doc.meta.get("keywords") or [])
        if kws & words:
            names.append(doc.meta["name"])
    sig_file = LEARNED_DIR / f"{_safe(signature)}.json"
    if sig_file.is_file():
        names.append(f"learned:{signature}")
    return names


def fill(name: str, task: str) -> dict:
    """Render a recipe into a concrete choreography spec."""
    if name.startswith("learned:"):
        sig = name[len("learned:"):]
        return json.loads((LEARNED_DIR / f"{_safe(sig)}.json").read_text())
    doc = next((d for d in _hand_written() if d.meta.get("name") == name), None)
    if doc is None:
        raise KeyError(f"no recipe named {name!r}")
    params = dict(doc.meta.get("params") or {})
    rendered = agentmd.render(doc.section("template"), **params)
    return json.loads(rendered)


def save_learned(signature: str, spec: dict) -> str:
    LEARNED_DIR.mkdir(parents=True, exist_ok=True)
    (LEARNED_DIR / f"{_safe(signature)}.json").write_text(json.dumps(spec, indent=1))
    return f"learned:{signature}"


def _safe(signature: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", signature)[:120] or "empty"
