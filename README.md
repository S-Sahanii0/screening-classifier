# Screening classifier

An LLM that sorts papers into **Group A** or **Group B** the way I did by hand during my systematic review.

## Why I built this

I did the screening for my Master's thesis systematic review by hand, and it was slow and repetitive. I wanted to find out whether an LLM could apply my written rubric the way I did, and where it would disagree with me.

A tool like this could save a reviewer a lot of time, but only if it rarely misses a Group A paper, so the main metric is **recall on A**, not plain accuracy. I also wanted to measure how well it works: my own labels on real papers, confidence intervals, and a close look at the errors.

## Background

The review looks at fairness-aware, explainable machine learning for clinical prediction (part of my Master's thesis). Every paper in it was already included. The question for each one is how it gets appraised:

- **Group A**: the paper builds its own model, predicts a real clinical outcome (mortality, length of stay, readmission), and reports fairness results by subgroup. These get a full PROBAST risk-of-bias appraisal.
- **Group B**: everything else (reviews, audits, imaging and text studies, papers that only mention fairness as motivation). These are handled in the narrative.

The labels are my own decisions on 71 real papers, so the evaluation uses real ground truth instead of synthetic data.

## How it works

```
title + abstract --> prompt (rubric) --> LLM --> JSON --> verdict + reason
```

The model answers three yes / no / unclear checks (original model, clinical outcome, fairness in results) and gives a verdict: `A`, `B` or `needs_check`. `needs_check` is allowed on purpose, because a paper the model cannot decide on should go to a human, not be forced into a group.

Four prompt variants are compared:

| Variant | What the model sees |
|---|---|
| `title_only` | Rubric, but only the title |
| `zero_shot` | One paragraph describing the groups, title + abstract |
| `rules` | Full three-check rubric, title + abstract |
| `few_shot` | Rubric plus six worked examples |

## Evaluation setup

- 64 papers are scored (26 A, 38 B): the 71 labeled papers minus one held out (label changed after full-text review) and the six few-shot examples. All four variants are scored on the same 64.
- Positive class is A. Reported: recall on A, precision on A, recall on B, strict accuracy, `needs_check` rate, parse errors, and a **human review load** (share of papers the model labels A or `needs_check`, which a person would still read).
- 95% Wilson confidence intervals. With 64 papers the intervals are wide, and I report them so the numbers are not over-read.
- `my_reason` (my written explanation for each label) is never sent to the model. It is only used in the disagreement report. A test checks this.
- The rubric text does not name any paper in the evaluation set. A test checks this too.

## Results

Model: `gemini-3.5-flash-lite` (Google free tier), temperature 0. 64 papers scored (26 Group A, 38 Group B). Positive class is A. Intervals are 95% Wilson intervals.

| Variant | Recall on A | Precision on A | Strict accuracy | needs_check | Papers a human still reads |
|---|---|---|---|---|---|
| title_only | 35% (19% to 54%) | 82% | 70% | 0% | 17% |
| zero_shot | 92% (76% to 98%) | 71% | 81% | 0% | 53% |
| rules | 81% (62% to 91%) | 91% | 86% | 5% | 41% |
| few_shot | 69% (50% to 83%) | 82% | 81% | 0% | 34% |

Strict accuracy counts `needs_check` as wrong, even though that answer just sends the paper to a person. Full tables and confusion matrices are in `results/summary.md`.

### What I take from this

- **The abstract matters.** With only the title, the model found 9 of 26 Group A papers. With the abstract and the rules it found 21.
- **Writing the rules helped overall.** `rules` had the best accuracy (86%) and the best precision (91%). `zero_shot` caught more Group A papers (24 of 26) but also labeled 10 Group B papers as A, so a person would still read over half the papers.
- **Examples did not help.** `few_shot` found fewer Group A papers than `rules` (18 vs 21). With six examples and 64 papers I would not read much into this, because the intervals overlap a lot.
- **It is not safe to use as a filter.** Even the best run for recall missed 2 of 26 Group A papers, and `rules` missed 4. If I skipped everything it labeled B, I would lose real Group A papers. I would use it as a second opinion, not to skip papers.
- With 64 papers, a gap of three or four papers (about 5 points) is within noise.

### Where it disagreed with me (`rules`)

9 of 64 papers differed from my label (the list is in `results/public/disagreements_rules.md`). Counting `needs_check` as a flag for a person, 6 were real misses and 3 were flags.

- **Called B, I said A (4 papers).** Three were the "audit versus own model" problem. These papers train or retrain models but frame the work as an audit, so the model decided it was not an original model. For one of them (#61) I had already marked myself unsure on that exact check. This looks like a gap in my rule more than a model mistake. The fourth (#56) was an abstract limit: the model said the abstract does not report fairness results, while the paper's Results section does.
- **Called A, I said B (2 papers).** For #62 the predicted outcome is unclear, which I had noted myself. #65 predicts blood oxygen saturation, and my own notes do not clearly say why I labeled it B, so I need to recheck it.
- **needs_check (3 papers).** The model said it could not tell from the abstract, which is the behavior I wanted. Two were Group B papers where fairness is only partly reported.

Several of these disagreements pointed to unclear spots in my own rules or labels, which I would not have noticed otherwise.

## Quick start

Needs Python 3.10+. The core uses only the standard library.

```bash
pip install -e .

# pick ONE free provider
cp .env.example .env        # then put a key in it
export $(grep -v '^#' .env | xargs)

# classify one paper
screener classify --title "..." --abstract "..."

# run the full comparison (slow on free tiers, so space out the calls)
screener eval --min-interval 5
```

Providers (all through one OpenAI-compatible client in `src/screener/llm.py`):

| Provider | Flag | Needs |
|---|---|---|
| Google Gemini | `--provider gemini` | `GEMINI_API_KEY` (free tier) |
| Groq | `--provider groq` | `GROQ_API_KEY` (free tier) |
| Ollama | `--provider ollama` | A local model, no key |

Free-tier limits and model names change. Override with `--model` or `SCREENER_MODEL`.

Responses are cached on disk in `.cache/llm`, so re-running costs nothing and a crash halfway does not lose work. Use `--no-cache` to turn that off.

## The data

The real dataset is **not in this repo**. It contains publisher abstracts and my own review notes, so I keep it local (`data/records.csv` is git-ignored). For the same reason, the per-paper outputs in `results/` (`predictions_*.csv` and `disagreements_*.md`, which include my notes) are git-ignored too. Shared instead: `results/summary.md`, and `results/public/disagreements_*.md`, which list each disagreement with the model's reason but without my notes. `data/example_records.csv` has four clearly synthetic rows so the format is visible.

To build the real file from your own labels and abstracts:

```bash
python scripts/build_dataset.py --records my_labels.csv --abstracts abstracts.md --out data/records.csv
```

The script prints any paper with a missing or very short abstract. Fix those by putting the correct abstracts in a second file (same format) and passing `--overrides fixes.md`.

## API

```bash
pip install -e ".[api]"
uvicorn screener.api:app --reload
curl -X POST localhost:8000/classify -H "Content-Type: application/json" \
  -d '{"title": "...", "abstract": "..."}'
```

Responses are marked `provisional: true` because the call is made from title and abstract only, not the full text.

Docker:

```bash
docker build -t screening-classifier .
docker run -p 8000:8000 --env-file .env screening-classifier
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```

Tests use a fake model client, so they run offline. They cover response parsing, the metrics, the evaluation split, the cache, and the API.

## Limitations

- **One reviewer.** The labels are mine. There was no second human reviewer, so there is no human agreement figure to compare against.
- **AI-assisted labels.** Some of my screening decisions were first suggested by an LLM and then reviewed by me. Agreement with an LLM could therefore be higher than it would be against fully independent human labels.
- **Small sample.** 64 scored papers. Treat differences of a few points between variants as noise.
- **Abstract only.** Some checks (especially "fairness in results") are easier to judge in the full text. The model is told to say `unclear` rather than guess.
- **One domain.** The rubric is specific to this review. It is a demonstration of the method, not a general screening tool.

## Project layout

```
src/screener/   llm.py (client), prompts.py, classify.py, metrics.py, evaluate.py, cli.py, api.py
scripts/        build_dataset.py
tests/          offline tests with a fake client
data/           example_records.csv (real data stays local)
results/        summary.md is shared; per-paper predictions and full disagreement reports are written here but stay local
results/public/ disagreement reports with my notes removed (safe to share)
```
