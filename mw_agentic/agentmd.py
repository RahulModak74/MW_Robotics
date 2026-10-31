"""
mw_agentic.agentmd
=====================
Parse a .md agent/recipe doc: YAML front matter + named sections
(`# System`, `# Task`, ...), with `{{var}}` substitution. A deliberately
small subset of analytics-harness-1.7's ah/agentmd.py (front matter +
sections + {{var}} rendering) - enough for one json-action agent, not the
full bash-agent/tool-doc machinery that harness also has.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None

_FRONT = re.compile(r"\A---\s*\n(.*?)\n---\s*\n(.*)\Z", re.S)
_SECTION = re.compile(r"^#\s+(\w+)\s*$", re.M)
_VAR = re.compile(r"\{\{\s*([a-zA-Z0-9_.]+)\s*\}\}")


@dataclass
class MdDoc:
    meta: dict = field(default_factory=dict)
    sections: dict = field(default_factory=dict)
    path: Path | None = None

    def section(self, name: str, default: str = "") -> str:
        return self.sections.get(name, default)


def _simple_yaml(text: str) -> dict:
    """A tiny YAML-ish fallback (key: value, key: [a, b, c]) if PyYAML is not
    installed - front matter here is simple enough not to need the real thing."""
    out: dict[str, Any] = {}
    for line in text.splitlines():
        if not line.strip() or line.strip().startswith("#"):
            continue
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        k, v = k.strip(), v.strip()
        if v.startswith("[") and v.endswith("]"):
            out[k] = [s.strip().strip('"\'') for s in v[1:-1].split(",") if s.strip()]
        elif v.lower() in ("true", "false"):
            out[k] = v.lower() == "true"
        else:
            try:
                out[k] = int(v)
            except ValueError:
                try:
                    out[k] = float(v)
                except ValueError:
                    out[k] = v.strip('"\'')
    return out


def load(path: str | Path) -> MdDoc:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    m = _FRONT.match(text)
    if not m:
        return MdDoc(meta={}, sections={"body": text}, path=path)
    meta_text, body = m.group(1), m.group(2)
    meta = yaml.safe_load(meta_text) if yaml else _simple_yaml(meta_text)
    sections: dict[str, str] = {}
    matches = list(_SECTION.finditer(body))
    for i, mm in enumerate(matches):
        start = mm.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        sections[mm.group(1).lower()] = body[start:end].strip()
    return MdDoc(meta=meta or {}, sections=sections, path=path)


def render(template: str, **vars: Any) -> str:
    def sub(m: re.Match) -> str:
        key = m.group(1)
        cur: Any = vars
        for part in key.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                return m.group(0)       # leave unresolved placeholders visible, not silently blank
        return str(cur)
    return _VAR.sub(sub, template)
