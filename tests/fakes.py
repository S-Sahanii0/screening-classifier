"""A fake chat client so tests never touch the network."""
from __future__ import annotations

import json


class FakeClient:
    name = "fake:test"

    def __init__(self, verdict: str = "B", reply: str | None = None):
        self.calls: list[tuple[str, str]] = []
        self._reply = reply
        self._verdict = verdict

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if self._reply is not None:
            return self._reply
        return json.dumps({
            "original_model": "no", "clinical_outcome": "no",
            "fairness_in_results": "no", "verdict": self._verdict,
            "reason": "fake",
        })
