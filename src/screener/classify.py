"""Turn a paper into a Prediction using a chat client and a prompt variant."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass

from .llm import ChatClient
from .prompts import build_system_prompt, build_user_prompt

VALID_VERDICTS = {"A", "B", "needs_check"}
VALID_CHECKS = {"yes", "no", "unclear"}


@dataclass
class Prediction:
    verdict: str  # "A", "B", "needs_check", or "parse_error"
    original_model: str = "unclear"
    clinical_outcome: str = "unclear"
    fairness_in_results: str = "unclear"
    reason: str = ""
    raw: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("raw")
        return d


def _extract_json(text: str) -> dict | None:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _norm_verdict(v: object) -> str | None:
    s = str(v).strip().lower().replace(" ", "_").replace("-", "_")
    return {"a": "A", "b": "B", "group_a": "A", "group_b": "B",
            "needs_check": "needs_check", "needscheck": "needs_check"}.get(s)


def derive_verdict(checks: list[str]) -> str:
    """The rubric's own verdict rule, applied to the three checks."""
    if "no" in checks:
        return "B"
    if "unclear" in checks:
        return "needs_check"
    return "A"


def parse_response(text: str) -> Prediction:
    obj = _extract_json(text)
    if obj is None:
        return Prediction(verdict="parse_error", raw=text)
    checks = {}
    for key in ("original_model", "clinical_outcome", "fairness_in_results"):
        val = str(obj.get(key, "unclear")).strip().lower()
        checks[key] = val if val in VALID_CHECKS else "unclear"
    verdict = _norm_verdict(obj.get("verdict"))
    if verdict is None:
        return Prediction(verdict="parse_error", raw=text, **checks)
    return Prediction(
        verdict=verdict,
        reason=str(obj.get("reason", "")).strip(),
        raw=text,
        **checks,
    )


class Classifier:
    def __init__(self, client: ChatClient, variant: str = "rules",
                 fewshot_papers: list[tuple[int, str, str]] | None = None):
        self.client = client
        self.variant = variant
        self.system = build_system_prompt(variant, fewshot_papers)

    def classify(self, title: str, abstract: str | None) -> Prediction:
        user = build_user_prompt(self.variant, title, abstract)
        return parse_response(self.client.complete(self.system, user))
