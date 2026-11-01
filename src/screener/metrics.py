"""Metrics for the A/B screening task. Pure Python, no numpy needed.

Group A is the positive class. Recall on A is the headline number: a missed A paper
skips PROBAST appraisal, which is the costly mistake. "needs_check" counts as wrong in
strict accuracy but is reported separately, because it just sends the paper to a human.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

LABELS = ("A", "B")
PRED_LABELS = ("A", "B", "needs_check", "parse_error")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _safe(a: int, b: int) -> float:
    return a / b if b else 0.0


@dataclass
class Report:
    n: int
    confusion: dict  # confusion[true][pred] -> count
    accuracy: float
    accuracy_ci: tuple[float, float]
    recall_a: float
    recall_a_ci: tuple[float, float]
    precision_a: float
    f1_a: float
    recall_b: float
    precision_b: float
    f1_b: float
    balanced_accuracy: float
    needs_check_rate: float
    parse_error_rate: float
    review_load: float  # share of papers a human must still look at (pred A or needs_check)


def evaluate(y_true: list[str], y_pred: list[str]) -> Report:
    assert len(y_true) == len(y_pred)
    n = len(y_true)
    conf = {t: {p: 0 for p in PRED_LABELS} for t in LABELS}
    for t, p in zip(y_true, y_pred):
        conf[t][p if p in PRED_LABELS else "parse_error"] += 1

    tp_a = conf["A"]["A"]
    fn_a = sum(conf["A"].values()) - tp_a
    fp_a = conf["B"]["A"]
    tp_b = conf["B"]["B"]
    fn_b = sum(conf["B"].values()) - tp_b
    fp_b = conf["A"]["B"]

    prec_a, rec_a = _safe(tp_a, tp_a + fp_a), _safe(tp_a, tp_a + fn_a)
    prec_b, rec_b = _safe(tp_b, tp_b + fp_b), _safe(tp_b, tp_b + fn_b)
    f1 = lambda p, r: _safe(2 * p * r, p + r)  # noqa: E731
    correct = tp_a + tp_b
    n_a = tp_a + fn_a
    flagged = sum(conf[t]["A"] + conf[t]["needs_check"] for t in LABELS)

    return Report(
        n=n,
        confusion=conf,
        accuracy=_safe(correct, n),
        accuracy_ci=wilson(correct, n),
        recall_a=rec_a,
        recall_a_ci=wilson(tp_a, n_a),
        precision_a=prec_a,
        f1_a=f1(prec_a, rec_a),
        recall_b=rec_b,
        precision_b=prec_b,
        f1_b=f1(prec_b, rec_b),
        balanced_accuracy=(rec_a + rec_b) / 2,
        needs_check_rate=_safe(sum(conf[t]["needs_check"] for t in LABELS), n),
        parse_error_rate=_safe(sum(conf[t]["parse_error"] for t in LABELS), n),
        review_load=_safe(flagged, n),
    )
