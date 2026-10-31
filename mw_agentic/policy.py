"""
mw_agentic.policy
====================
Thompson-sampling bandit over choreography RECIPES, ported from
analytics-harness-1.7's ah/rl.py::Policy almost unchanged (same Beta-bandit
math, same preventive()/preflight idea - the piece that turns "ask the LLM"
into "just apply what has worked before").

State      a task signature (a normalised string describing the request)
Action     a recipe name (a saved, parameterised choreography template)
Reward     1.0 if the recipe's filled-in choreography validates and (when
          a human reviewed it) was approved; 0.0 otherwise
Policy     Thompson sampling, Beta(successes+1, failures+1) per
          (signature, recipe) - identical shape to the harness's bandit

preventive(): once a (signature, recipe) pair's success rate crosses a
threshold with enough observations, choreograph() applies it WITHOUT
calling the LLM at all - this is "once an acceptable result exists, the
system should follow a deterministic approach" from the harness's own
preflight mechanism, carried over unchanged in spirit.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

CAP = 60.0


def _cap(ab: list, cap: float | None) -> None:
    n = ab[0] + ab[1]
    if cap and n > cap:
        ab[0], ab[1] = ab[0] * cap / n, ab[1] * cap / n


class Policy:
    def __init__(self, rl_dir: str | Path, seed: int | None = None, cap: float | None = CAP):
        self.cap = cap
        self.dir = Path(rl_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "policy.json"
        self.data = json.loads(self.path.read_text()) if self.path.is_file() else {"arms": {}, "signals": {}}
        self.rng = random.Random(seed)

    def _arm(self, sig: str, arm: str, prior=None) -> list:
        return self.data["arms"].setdefault(sig, {}).setdefault(arm, list(map(float, prior or [1, 1])))

    def belief(self, sig: str, arm: str, prior=None) -> tuple:
        a, b = self._arm(sig, arm, prior)
        return a, b

    def rank(self, sig: str, arms: list, priors: dict | None = None) -> list:
        """Thompson draw per arm: sample a success rate from each recipe's
        learned Beta distribution, return arms sorted best-draw-first."""
        if not arms:
            return []
        draws = {a: self.rng.betavariate(*self.belief(sig, a, (priors or {}).get(a))) for a in arms}
        return sorted(arms, key=lambda a: -draws[a])

    def mean(self, sig: str, arm: str) -> float:
        a, b = self.belief(sig, arm)
        return a / (a + b)

    def n_obs(self, sig: str, arm: str) -> float:
        a, b = self.belief(sig, arm)
        return a + b - 2.0

    def update(self, sig: str, arm: str, reward: float) -> None:
        self._sync()
        ab = self._arm(sig, arm)
        ab[0] += reward
        ab[1] += 1 - reward
        _cap(ab, self.cap)
        self.save()

    def observe_signal(self, sig: str) -> None:
        """Count that this task signature has been seen (independent of which
        recipe was tried) - used to decide whether a signature has enough
        history to trust a recipe's success rate at all."""
        self._sync()
        s = self.data["signals"].setdefault(sig, [1.0, 1.0])
        s[0] += 1
        self.save()

    def preventive(self, sig: str, min_rate: float = 0.7, min_obs: int = 2) -> str | None:
        """The 'go deterministic' decision: is there a recipe for this
        signature that has succeeded often enough, with enough tries, to
        trust without asking the LLM again? Returns its name, or None."""
        arms = self.data["arms"].get(sig, {})
        candidates = [(name, self.mean(sig, name)) for name in arms if self.n_obs(sig, name) >= min_obs]
        candidates = [(n, m) for n, m in candidates if m >= min_rate]
        if not candidates:
            return None
        return max(candidates, key=lambda nm: nm[1])[0]

    def _sync(self) -> None:
        if self.path.is_file():
            try:
                disk = json.loads(self.path.read_text())
                for k, v in disk.items():
                    if isinstance(v, dict) and isinstance(self.data.get(k), dict):
                        self.data[k].update(v)
                    else:
                        self.data[k] = v
            except ValueError:
                pass

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1))
        tmp.replace(self.path)

    def log(self, **event) -> None:
        with (self.dir / "experience.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), **event}) + "\n")
