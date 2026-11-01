"""Load the labeled records and decide which ones are scored."""
from __future__ import annotations

import csv
from dataclasses import dataclass

# Workbook numbers used as few-shot examples. They are never scored.
FEWSHOT_IDS = (10, 46, 15, 4, 39, 24)


@dataclass
class Record:
    workbook_no: int
    id: str
    title: str
    abstract: str
    group: str
    holdout: bool
    my_reason: str  # for error analysis only. NEVER passed to the model.


def load_records(path: str) -> list[Record]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [
        Record(
            workbook_no=int(r["workbook_no"]),
            id=r["id"],
            title=r["title"],
            abstract=(r.get("abstract") or "").strip(),
            group=r["group"].strip(),
            holdout=bool(r.get("holdout", "").strip()),
            my_reason=r.get("my_reason", ""),
        )
        for r in rows
    ]


def eval_set(records: list[Record], need_abstract: bool = True) -> list[Record]:
    """Records that are scored: not held out, not a few-shot example, has an abstract."""
    out = []
    for r in records:
        if r.holdout or r.workbook_no in FEWSHOT_IDS:
            continue
        if need_abstract and not r.abstract:
            continue
        out.append(r)
    return out


def fewshot_papers(records: list[Record]) -> list[tuple[int, str, str]]:
    by_no = {r.workbook_no: r for r in records}
    missing = [n for n in FEWSHOT_IDS if n not in by_no or not by_no[n].abstract]
    if missing:
        raise ValueError(f"few-shot papers missing from the data file: {missing}")
    return [(n, by_no[n].title, by_no[n].abstract) for n in FEWSHOT_IDS]
