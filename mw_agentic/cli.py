#!/usr/bin/env python3
"""
python -m mw_agentic "create a dancing robot pair"
Prints the resulting choreography.json to stdout, and a one-line summary
(path taken, LLM calls, recipe used if any) to stderr.
"""
from __future__ import annotations

import json
import sys

from .orchestrate import choreograph


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print("usage: python -m mw_agentic \"<task>\"", file=sys.stderr)
        return 2
    task = " ".join(argv)
    result = choreograph(task)
    print(json.dumps(result.get("spec"), indent=1))
    summary = (f"path={result['path']} llm_calls={result.get('llm_calls', 0)} "
              f"recipe={result.get('recipe', '-')} ok={result['ok']}")
    if not result["ok"]:
        summary += f" errors={result.get('errors')}"
    print(summary, file=sys.stderr)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
