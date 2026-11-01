# Results (gemini:gemini-3.5-flash-lite)

Positive class is Group A. 95% intervals are Wilson intervals. Strict accuracy counts `needs_check` as wrong.

| Variant | n | Strict accuracy | Recall on A | Precision on A | Recall on B | needs_check | Parse errors | Human review load |
|---|---|---|---|---|---|---|---|---|
| title_only | 64 | 70% (58% to 80%) | 35% (19% to 54%) | 82% | 95% | 0% | 0% | 17% |
| zero_shot | 64 | 81% (70% to 89%) | 92% (76% to 98%) | 71% | 74% | 0% | 0% | 53% |
| rules | 64 | 86% (75% to 92%) | 81% (62% to 91%) | 91% | 89% | 5% | 0% | 41% |
| few_shot | 64 | 81% (70% to 89%) | 69% (50% to 83%) | 82% | 89% | 0% | 0% | 34% |

## Confusion matrices (rows = my label, columns = model)

**title_only**

| | A | B | needs_check | parse_error |
|---|---|---|---|---|
| A | 9 | 17 | 0 | 0 |
| B | 2 | 36 | 0 | 0 |

**zero_shot**

| | A | B | needs_check | parse_error |
|---|---|---|---|---|
| A | 24 | 2 | 0 | 0 |
| B | 10 | 28 | 0 | 0 |

**rules**

| | A | B | needs_check | parse_error |
|---|---|---|---|---|
| A | 21 | 4 | 1 | 0 |
| B | 2 | 34 | 2 | 0 |

**few_shot**

| | A | B | needs_check | parse_error |
|---|---|---|---|---|
| A | 18 | 8 | 0 | 0 |
| B | 4 | 34 | 0 | 0 |

Human review load = share of papers the model labels A or needs_check, i.e. the papers a person would still read. Everything labeled B is skipped.
