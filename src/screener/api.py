"""FastAPI wrapper. Install with `pip install -e ".[api]"` and run:

    uvicorn screener.api:app --reload
"""
from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .classify import Classifier
from .data import fewshot_papers, load_records
from .llm import LLMClient, LLMConfig, LLMError
from .prompts import VARIANTS

app = FastAPI(title="Screening classifier", version="0.1.0")


class ClassifyRequest(BaseModel):
    title: str = Field(min_length=3)
    abstract: str = Field(min_length=20)
    variant: str = "rules"


class ClassifyResponse(BaseModel):
    verdict: str
    original_model: str
    clinical_outcome: str
    fairness_in_results: str
    reason: str
    provisional: bool = True  # decided from title and abstract only


@lru_cache(maxsize=1)
def get_client() -> LLMClient:
    return LLMClient(LLMConfig.from_env())


def get_classifier(variant: str) -> Classifier:
    if variant not in VARIANTS or variant == "title_only":
        raise HTTPException(422, f"variant must be one of {[v for v in VARIANTS if v != 'title_only']}")
    fs = None
    if variant == "few_shot":
        try:
            fs = fewshot_papers(load_records("data/records.csv"))
        except (FileNotFoundError, ValueError) as e:
            raise HTTPException(503, f"few_shot needs data/records.csv: {e}")
    return Classifier(get_client(), variant, fs)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest) -> ClassifyResponse:
    clf = get_classifier(req.variant)
    try:
        pred = clf.classify(req.title, req.abstract)
    except LLMError as e:
        raise HTTPException(502, f"model call failed: {e}")
    if pred.verdict == "parse_error":
        raise HTTPException(502, "model returned an unreadable answer")
    return ClassifyResponse(**pred.to_dict())
