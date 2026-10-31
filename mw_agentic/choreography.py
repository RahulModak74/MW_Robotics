"""
mw_agentic.choreography
==========================
The schema and the deterministic validator for a choreography spec - the
"artifact" in this harness-shaped design, equivalent to the layout.json a
json agent drafts and bi.py check validates in analytics-harness-1.7.

Schema:
    {"title": str, "duration_s": float,
     "actions": [{"robot": "a"|"b", "t": float>=0, "tool": <name>, "params": {...}}, ...]}

Tool params and their valid ranges match what each trained MW domain was
actually shown during training (see ../mw_framework/domains/*.py and
../mw_framework/datasets.py) - an out-of-range request is rejected here
BEFORE it ever reaches a manifold, the same role bi.py check plays for a
dashboard layout.
"""
from __future__ import annotations

TOOL_SCHEMA = {
    "walk":  {"dx": (-3.0, 3.0)},
    "reach": {"x": (-1.8, 1.8), "y": (-1.8, 1.8)},     # also checked: hypot(x,y) <= 1.8 (L1+L2)
    "hop":   {"height": (0.1, 1.5)},
    "lift":  {"deflection": (0.1, 0.9)},
    "joint": {"theta": (-1.4, 1.4)},
}


def validate(spec: dict) -> list:
    """Returns a list of error strings; empty = valid. Never raises - a
    malformed spec is just a very long error list."""
    errors = []
    if not isinstance(spec, dict):
        return ["spec must be a JSON object"]
    if not isinstance(spec.get("title"), str) or not spec["title"].strip():
        errors.append("title: missing or empty")
    actions = spec.get("actions")
    if not isinstance(actions, list) or not actions:
        errors.append("actions: missing or empty list")
        return errors

    per_robot_times = {"a": [], "b": []}
    max_t = 0.0
    for i, a in enumerate(actions):
        tag = f"actions[{i}]"
        if not isinstance(a, dict):
            errors.append(f"{tag}: must be an object")
            continue
        robot = a.get("robot")
        if robot not in ("a", "b"):
            errors.append(f"{tag}.robot: must be 'a' or 'b', got {robot!r}")
            robot = None
        t = a.get("t")
        if not isinstance(t, (int, float)) or t < 0:
            errors.append(f"{tag}.t: must be a number >= 0, got {t!r}")
            t = 0.0
        tool = a.get("tool")
        if tool not in TOOL_SCHEMA:
            errors.append(f"{tag}.tool: must be one of {sorted(TOOL_SCHEMA)}, got {tool!r}")
            continue
        params = a.get("params")
        if not isinstance(params, dict):
            errors.append(f"{tag}.params: must be an object")
            continue
        for p, (lo, hi) in TOOL_SCHEMA[tool].items():
            v = params.get(p)
            if not isinstance(v, (int, float)):
                errors.append(f"{tag}.params.{p}: missing or not a number")
            elif not (lo <= v <= hi):
                errors.append(f"{tag}.params.{p}: {v} out of trained range [{lo}, {hi}]")
        if tool == "reach" and isinstance(params.get("x"), (int, float)) and isinstance(params.get("y"), (int, float)):
            r = (params["x"] ** 2 + params["y"] ** 2) ** 0.5
            if r > 1.8:
                errors.append(f"{tag}.params: (x,y) is {r:.2f} from the shoulder, beyond the arm's 1.8 reach")
        if robot:
            per_robot_times[robot].append(t)
        max_t = max(max_t, t)

    for robot, times in per_robot_times.items():
        if len(times) != len(set(times)):
            errors.append(f"robot {robot}: two actions scheduled at the same t - it can only do one thing at a time")

    duration_s = spec.get("duration_s")
    if not isinstance(duration_s, (int, float)) or duration_s < max_t:
        errors.append(f"duration_s: must be a number >= the latest action time ({max_t})")

    return errors
