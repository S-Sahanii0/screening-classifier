"""Run prompt variants over the labeled set and write results files."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

from .classify import Classifier
from .data import Record, eval_set, fewshot_papers
from .llm import ChatClient, LLMError
from .metrics import Report, evaluate


def run_variant(client: ChatClient, variant: str, all_records: list[Record],
                limit: int | None = None, verbose: bool = True) -> list[dict]:
    fs = fewshot_papers(all_records) if variant == "few_shot" else None
    clf = Classifier(client, variant, fs)
    rows = []
    # Same papers for every variant so the ablation compares like with like.
    todo = eval_set(all_records, need_abstract=True)
    if limit:
        todo = todo[:limit]
    for i, r in enumerate(todo, 1):
        try:
            pred = clf.classify(r.title, r.abstract)
        except LLMError as e:
            print(f"  [{variant}] #{r.workbook_no} failed: {e}", file=sys.stderr)
            continue
        rows.append({
            "workbook_no": r.workbook_no,
            "title": r.title,
            "true": r.group,
            "pred": pred.verdict,
            **{k: v for k, v in pred.to_dict().items() if k != "verdict"},
            "my_reason": r.my_reason,  # error analysis only, never sent to the model
        })
        if verbose:
            mark = "ok " if pred.verdict == r.group else "ERR"
            print(f"  [{variant}] {i}/{len(todo)} #{r.workbook_no} true={r.group} pred={pred.verdict} {mark}")
    return rows


def write_predictions(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def report_for(rows: list[dict]) -> Report:
    return evaluate([r["true"] for r in rows], [r["pred"] for r in rows])


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def _ci(ci: tuple[float, float]) -> str:
    return f"{_pct(ci[0])} to {_pct(ci[1])}"


def summary_markdown(reports: dict[str, Report], model_name: str) -> str:
    lines = [
        f"# Results ({model_name})",
        "",
        "Positive class is Group A. 95% intervals are Wilson intervals. "
        "Strict accuracy counts `needs_check` as wrong.",
        "",
        "| Variant | n | Strict accuracy | Recall on A | Precision on A | Recall on B | needs_check | Parse errors | Human review load |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, r in reports.items():
        lines.append(
            f"| {name} | {r.n} | {_pct(r.accuracy)} ({_ci(r.accuracy_ci)}) | "
            f"{_pct(r.recall_a)} ({_ci(r.recall_a_ci)}) | {_pct(r.precision_a)} | "
            f"{_pct(r.recall_b)} | {_pct(r.needs_check_rate)} | {_pct(r.parse_error_rate)} | "
            f"{_pct(r.review_load)} |"
        )
    lines += ["", "## Confusion matrices (rows = my label, columns = model)", ""]
    for name, r in reports.items():
        lines += [f"**{name}**", "", "| | A | B | needs_check | parse_error |", "|---|---|---|---|---|"]
        for t in ("A", "B"):
            c = r.confusion[t]
            lines.append(f"| {t} | {c['A']} | {c['B']} | {c['needs_check']} | {c['parse_error']} |")
        lines.append("")
    lines += [
        "Human review load = share of papers the model labels A or needs_check, "
        "i.e. the papers a person would still read. Everything labeled B is skipped.",
    ]
    return "\n".join(lines) + "\n"


def disagreements_markdown(rows: list[dict], variant: str) -> str:
    wrong = [r for r in rows if r["pred"] != r["true"]]
    out = [f"# Disagreements: {variant}", "",
           f"{len(wrong)} of {len(rows)} papers differ from my label.", ""]
    for r in wrong:
        out += [
            f"## #{r['workbook_no']} {r['title']}",
            f"- My label: {r['true']}. Model: {r['pred']}",
            f"- Checks (original model / clinical outcome / fairness in results): "
            f"{r['original_model']} / {r['clinical_outcome']} / {r['fairness_in_results']}",
            f"- Model reason: {r['reason']}",
            f"- My reason: {r['my_reason']}",
            "",
        ]
    return "\n".join(out)
