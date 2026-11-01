"""Prompt variants used in the ablation.

title_only : full rules, but the model only sees the title
zero_shot  : one-paragraph description of the two groups, no checks
rules      : the three-check rubric (the main prompt)
few_shot   : rubric + six worked examples taken from the labeled set

Nothing here may mention a specific paper from the evaluation set, otherwise the
rubric would leak labels. The few-shot examples are the one deliberate exception and
those papers are excluded from scoring (see data.FEWSHOT_IDS).
"""
from __future__ import annotations

import json

VARIANTS = ("title_only", "zero_shot", "rules", "few_shot")

OUTPUT_FORMAT = """Reply with one JSON object and nothing else, in this form:
{
  "original_model": "yes" | "no" | "unclear",
  "clinical_outcome": "yes" | "no" | "unclear",
  "fairness_in_results": "yes" | "no" | "unclear",
  "verdict": "A" | "B" | "needs_check",
  "reason": "one short sentence"
}"""

CONTEXT = (
    "You help with a systematic review on fairness-aware, explainable machine learning "
    "for clinical prediction. Every paper you will see was already included in the review. "
    "Your job is to sort each included paper into Group A or Group B, the way the "
    "human reviewer did."
)

RUBRIC = """Group A: prediction-model studies, appraised with PROBAST.
A paper is Group A only if ALL three checks are "yes":
1. original_model. The paper develops or trains its own predictive model and claims the
   model (or its training method) as a contribution. Applying, auditing or benchmarking
   existing models does not count. If the authors present the work as an audit or
   benchmark and reuse standard architectures, the answer is "no" even if they trained
   the weights themselves.
2. clinical_outcome. The model predicts a real patient-level clinical outcome, such as
   mortality, length of stay, readmission, decompensation, sepsis onset or similar.
   Diagnostic image classification alone, text classification, predicting demographics,
   or any non-clinical target does not count.
3. fairness_in_results. Fairness or bias metrics are reported as results, broken down by
   a demographic or socioeconomic subgroup (race, sex, age, insurance status, income
   proxy). Fairness mentioned only as motivation or as future work does not count.
   Technical variation across hospitals or sites does not count.

Group B: fairness, audit, review and framework papers. A paper is Group B if at least
one check is "no". Typical patterns: reviews and scoping papers with no original model;
audits of existing models; bias studies on non-outcome tasks (clinical notes, AI-generated
images, demographic prediction from scans); diagnostic image classification; fairness
reported only as motivation.

Verdict rule:
- "A" if all three checks are "yes".
- "B" if at least one check is "no".
- "needs_check" only if no check is "no" but at least one is "unclear".

You only see the title and abstract, not the full text. Judge from what is written.
If the abstract does not say whether something is true, use "unclear" for that check.
Do not guess in favour of A."""

ZERO_SHOT = (
    "Group A papers build their own machine learning model that predicts a real clinical "
    "outcome (for example mortality, length of stay or readmission) and report fairness "
    "or bias results across demographic or socioeconomic subgroups. Group B papers are "
    "everything else: reviews, audits of existing models, bias studies on non-outcome "
    "tasks such as imaging or text, and papers where fairness is only mentioned as "
    "motivation. Use \"needs_check\" if you cannot tell."
)


def _fmt_paper(title: str, abstract: str | None) -> str:
    if abstract is None:
        return f"Title: {title}"
    return f"Title: {title}\n\nAbstract: {abstract}"


# Hand-written answers for the few-shot papers, keyed by workbook number.
FEWSHOT_ANSWERS: dict[int, dict] = {
    10: dict(original_model="yes", clinical_outcome="yes", fairness_in_results="yes", verdict="A",
             reason="Trains its own XGBoost model for ICU in-hospital mortality and assesses bias between racial groups."),
    46: dict(original_model="yes", clinical_outcome="yes", fairness_in_results="yes", verdict="A",
             reason="Introduces a post-processing algorithm on ICU mortality predictors and reports sensitivity gaps across subgroups."),
    15: dict(original_model="yes", clinical_outcome="yes", fairness_in_results="yes", verdict="A",
             reason="Trains random forest models for readmission and mortality and reports fairness gaps by race and sex over time."),
    4: dict(original_model="no", clinical_outcome="no", fairness_in_results="no", verdict="B",
            reason="Scoping review with no model; it examines dataset origins and author demographics."),
    39: dict(original_model="no", clinical_outcome="no", fairness_in_results="yes", verdict="B",
             reason="Audit of chest X-ray label classifiers; diagnostic imaging, not a patient-level outcome."),
    24: dict(original_model="yes", clinical_outcome="yes", fairness_in_results="no", verdict="B",
             reason="Builds a frailty model but says subgroup fairness assessment is still needed, so no fairness results."),
}


def build_system_prompt(variant: str, fewshot_papers: list[tuple[int, str, str]] | None = None) -> str:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}, choose from {VARIANTS}")
    if variant == "zero_shot":
        return f"{CONTEXT}\n\n{ZERO_SHOT}\n\n{OUTPUT_FORMAT}"
    parts = [CONTEXT, RUBRIC]
    if variant == "few_shot":
        if not fewshot_papers:
            raise ValueError("few_shot needs the example papers (see data.fewshot_papers)")
        blocks = []
        for i, (num, title, abstract) in enumerate(fewshot_papers, 1):
            answer = json.dumps(FEWSHOT_ANSWERS[num], indent=2)
            blocks.append(f"Example {i}\n{_fmt_paper(title, abstract)}\n\nAnswer:\n{answer}")
        parts.append("Worked examples:\n\n" + "\n\n".join(blocks))
    parts.append(OUTPUT_FORMAT)
    return "\n\n".join(parts)


def build_user_prompt(variant: str, title: str, abstract: str | None) -> str:
    if variant == "title_only":
        return _fmt_paper(title, None)
    return _fmt_paper(title, abstract or "")
