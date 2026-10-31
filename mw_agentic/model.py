"""
mw_agentic.model
===================
Two ways to query a model for the choreographer agent, mirroring
analytics-harness-1.7's "model string" idea (ah/models.py) but scoped to
just what this standalone demo needs:

  ClaudeCodeModel   shells out to the `claude` CLI in --print mode. Works
                    on this machine, and anywhere Claude Code is already
                    set up, with NO separate API key to manage.
  AnthropicAPIModel  a direct HTTPS call to api.anthropic.com, reading
                    ANTHROPIC_API_KEY from the environment at runtime (the
                    same convention the rest of this owner's code uses -
                    see e.g. FINAL_PROD_VERSION_BATTERY_ANALYTICS/can_convert.sh).
                    Portable to a machine without Claude Code installed.

pick_model() tries ClaudeCodeModel first (nothing to configure on this
machine or any Claude-Code-based run), falls back to AnthropicAPIModel if
an API key is set, and raises a clear error otherwise - never silently
guesses or fabricates a response.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.error
import urllib.request


class ModelUnavailable(RuntimeError):
    pass


class ClaudeCodeModel:
    name = "claude-code-cli"

    def __init__(self, timeout: int = 90):
        self.timeout = timeout
        if not shutil.which("claude"):
            raise ModelUnavailable("the 'claude' CLI is not on PATH")

    def query(self, system: str, user: str) -> str:
        cmd = ["claude", "-p", user, "--system-prompt", system,
              "--output-format", "text", "--restricted"]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
        except subprocess.TimeoutExpired as e:
            raise ModelUnavailable(f"claude -p timed out after {self.timeout}s") from e
        if r.returncode != 0:
            raise ModelUnavailable(f"claude -p exited {r.returncode}: {(r.stderr or r.stdout)[:400]}")
        return r.stdout.strip()


class AnthropicAPIModel:
    name = "anthropic-api"

    def __init__(self, model: str = "claude-sonnet-5-5", max_tokens: int = 2000, timeout: int = 60):
        self.key = os.environ.get("ANTHROPIC_API_KEY")
        if not self.key:
            raise ModelUnavailable("ANTHROPIC_API_KEY is not set")
        self.model, self.max_tokens, self.timeout = model, max_tokens, timeout
        self.base_url = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")

    def query(self, system: str, user: str) -> str:
        body = json.dumps({"model": self.model, "max_tokens": self.max_tokens,
                          "system": system, "messages": [{"role": "user", "content": user}]}).encode()
        req = urllib.request.Request(self.base_url + "/v1/messages", data=body, headers={
            "content-type": "application/json", "x-api-key": self.key, "anthropic-version": "2023-06-01"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise ModelUnavailable(f"Anthropic API HTTP {e.code}: {e.read()[:300]!r}") from e
        return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")


def pick_model():
    """ClaudeCodeModel first (zero config here), then a raw API key, else
    raise with a clear message naming both options - never fabricate a
    choreography without a real model behind it."""
    errors = []
    for cls in (ClaudeCodeModel, AnthropicAPIModel):
        try:
            return cls()
        except ModelUnavailable as e:
            errors.append(f"{cls.__name__}: {e}")
    raise ModelUnavailable("no model available - " + "; ".join(errors) +
                           ". Install/auth the claude CLI, or set ANTHROPIC_API_KEY.")
