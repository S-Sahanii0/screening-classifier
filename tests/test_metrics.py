import math

from screener.metrics import evaluate, wilson


def test_perfect_predictions():
    r = evaluate(["A", "A", "B", "B"], ["A", "A", "B", "B"])
    assert r.accuracy == 1 and r.recall_a == 1 and r.precision_a == 1 and r.balanced_accuracy == 1


def test_known_confusion():
    # 3 true A: 2 caught, 1 called B.  3 true B: 2 correct, 1 called A.
    r = evaluate(["A", "A", "A", "B", "B", "B"], ["A", "A", "B", "B", "B", "A"])
    assert r.confusion["A"]["A"] == 2 and r.confusion["A"]["B"] == 1
    assert math.isclose(r.recall_a, 2 / 3)
    assert math.isclose(r.precision_a, 2 / 3)
    assert math.isclose(r.accuracy, 4 / 6)


def test_needs_check_counts_wrong_but_goes_to_human():
    r = evaluate(["A", "B"], ["needs_check", "B"])
    assert r.accuracy == 0.5 and r.needs_check_rate == 0.5
    assert r.review_load == 0.5  # the needs_check paper still gets read


def test_parse_errors_are_counted():
    r = evaluate(["A", "B"], ["parse_error", "B"])
    assert r.parse_error_rate == 0.5 and r.accuracy == 0.5


def test_wilson_interval_sane():
    lo, hi = wilson(8, 10)
    assert 0 < lo < 0.8 < hi < 1
    assert wilson(0, 0) == (0.0, 0.0)
    lo, hi = wilson(10, 10)
    assert hi == 1.0 and lo > 0.6
