import csv

from screener.data import FEWSHOT_IDS, eval_set, fewshot_papers, load_records
from screener.evaluate import (disagreements_markdown, report_for, run_variant,
                               summary_markdown)

from fakes import FakeClient

FIELDS = ["id", "workbook_no", "title", "authors", "abstract", "group", "holdout", "my_reason"]


def make_csv(path, n=30, missing=(8,), holdout=(9,)):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for i in range(1, n + 1):
            w.writerow({
                "id": f"P{i}", "workbook_no": i, "title": f"Title {i}",
                "authors": "X", "abstract": "" if i in missing else f"Abstract {i}",
                "group": "A" if i % 2 else "B",
                "holdout": "yes" if i in holdout else "",
                "my_reason": f"PRIVATE-REASON-{i}",
            })
    return str(path)


def test_eval_set_excludes_holdout_fewshot_and_missing(tmp_path):
    recs = load_records(make_csv(tmp_path / "r.csv"))
    nos = {r.workbook_no for r in eval_set(recs)}
    assert 9 not in nos and 8 not in nos
    assert not nos & set(FEWSHOT_IDS)
    # title_only does not need an abstract, so #8 is allowed there
    assert 8 in {r.workbook_no for r in eval_set(recs, need_abstract=False)}


def test_fewshot_papers_error_when_missing(tmp_path):
    recs = load_records(make_csv(tmp_path / "r.csv", n=3))
    try:
        fewshot_papers(recs)
    except ValueError as e:
        assert "missing" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_my_reason_is_never_sent_to_the_model(tmp_path):
    recs = load_records(make_csv(tmp_path / "r.csv", n=50))  # few-shot ids go up to 46
    client = FakeClient("B")
    for variant in ("title_only", "zero_shot", "rules", "few_shot"):
        run_variant(client, variant, recs, verbose=False)
    assert client.calls, "fake client was never called"
    for system, user in client.calls:
        assert "PRIVATE-REASON" not in system and "PRIVATE-REASON" not in user


def test_run_variant_and_reports(tmp_path):
    recs = load_records(make_csv(tmp_path / "r.csv"))
    rows = run_variant(FakeClient("B"), "rules", recs, verbose=False)
    assert rows and all(r["pred"] == "B" for r in rows)
    rep = report_for(rows)
    assert rep.recall_a == 0 and rep.recall_b == 1
    md = summary_markdown({"rules": rep}, "fake:test")
    assert "Recall on A" in md and "rules" in md
    dis = disagreements_markdown(rows, "rules")
    assert "My reason: PRIVATE-REASON" in dis  # allowed here: error analysis only


def test_limit_argument(tmp_path):
    recs = load_records(make_csv(tmp_path / "r.csv"))
    rows = run_variant(FakeClient("A"), "rules", recs, limit=3, verbose=False)
    assert len(rows) == 3
