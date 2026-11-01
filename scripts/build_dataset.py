"""Merge a records CSV (labels) with a markdown file of abstracts into data/records.csv.

Usage:
    python scripts/build_dataset.py --records records_2.csv \
        --abstracts "Screening papers - abstract only.md" --out data/records.csv

The abstracts file is expected to look like:
    12. **Title | Authors (2024)** Abstract: text ...
Records are joined on the number in front of each entry (workbook_no).
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

ENTRY = re.compile(r"^\s*(\d+)\.\s+\*\*", re.M)
NOISE = re.compile(r"Here is the [^\n]*?formatted as requested:?", re.I)

# Papers that should not be scored (label changed after first screening).
HOLDOUT = {9}


def parse_abstracts(text: str) -> dict[int, str]:
    text = NOISE.sub("", text)
    matches = list(ENTRY.finditer(text))
    out: dict[int, str] = {}
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[m.end() - 2 : end]  # keep the leading ** for splitting
        num = int(m.group(1))
        if "NOT FOUND" in chunk[:40]:
            out[num] = ""
            continue
        _, _, abstract = chunk.partition("Abstract:")
        abstract = abstract.replace("**", "")
        abstract = re.sub(r"\\([<>_\[\]()#*~|-])", r"\1", abstract)  # unescape markdown
        abstract = re.sub(r"\s+", " ", abstract).strip()
        out[num] = abstract
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--abstracts", required=True)
    ap.add_argument("--out", default="data/records.csv")
    ap.add_argument("--overrides", default=None,
                    help="optional markdown file in the same format; its entries replace "
                         "or fill in abstracts from the main file")
    args = ap.parse_args()

    abstracts = parse_abstracts(Path(args.abstracts).read_text(encoding="utf-8"))
    if args.overrides:
        fixes = parse_abstracts(Path(args.overrides).read_text(encoding="utf-8"))
        abstracts.update({n: a for n, a in fixes.items() if a})
        print(f"applied overrides for: {sorted(fixes)}")
    with open(args.records, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    fields = ["id", "workbook_no", "title", "authors", "abstract", "group",
              "holdout", "my_reason"]
    missing, short = [], []
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            n = int(r["workbook_no"])
            abstract = abstracts.get(n, "")
            if not abstract:
                missing.append(n)
            elif len(abstract) < 400:
                short.append(n)
            w.writerow({
                "id": r["id"],
                "workbook_no": n,
                "title": r["title"],
                "authors": r["authors"],
                "abstract": abstract,
                "group": r["group"],
                "holdout": "yes" if n in HOLDOUT else "",
                "my_reason": r["my_reason"],
            })
    print(f"wrote {len(rows)} rows to {args.out}")
    print(f"missing abstract: {missing}")
    print(f"very short abstract (<400 chars, check by hand): {short}")


if __name__ == "__main__":
    main()
