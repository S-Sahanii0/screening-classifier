"""Provider-agnostic chat client.

Gemini, Groq and Ollama all expose an OpenAI-compatible /chat/completions endpoint,
so one small client covers all three. Standard library only (urllib).

Free tiers and model names change often. Override the defaults with env vars:
    SCREENER_PROVIDER   gemini | groq | ollama   (default: gemini)
    SCREENER_MODEL      model name for that provider
    SCREENER_BASE_URL   override the endpoint
    SCREENER_API_KEY    override the key (otherwise GEMINI_API_KEY / GROQ_API_KEY)
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

PROVIDERS = {
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "key_env": "GEMINI_API_KEY",
        "model": "gemini-3.8-flash",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "model": "llama-3.3-70b-versatile",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "key_env": None,
        "model": "llama3.1:8b",
    },
}


class ChatClient(Protocol):
    """Anything with this method can be used by the classifier (real client or test fake)."""

    name: str

    def complete(self, system: str, user: str) -> str: ...


@dataclass
class LLMConfig:
    provider: str = "gemini"
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = 0.0
    timeout: float = 120.0
    max_retries: int = 6
    min_interval: float = 0.0  # seconds between calls, for free-tier rate limits
    cache_dir: str | None = ".cache/llm"

    @classmethod
    def from_env(cls, **overrides) -> "LLMConfig":
        provider = overrides.pop("provider", None) or os.getenv("SCREENER_PROVIDER", "gemini")
        if provider not in PROVIDERS:
            raise ValueError(f"unknown provider {provider!r}, choose from {sorted(PROVIDERS)}")
        preset = PROVIDERS[provider]
        key = os.getenv("SCREENER_API_KEY") or (
            os.getenv(preset["key_env"]) if preset["key_env"] else None
        )
        cfg = cls(
            provider=provider,
            model=os.getenv("SCREENER_MODEL") or preset["model"],
            base_url=os.getenv("SCREENER_BASE_URL") or preset["base_url"],
            api_key=key,
        )
        for k, v in overrides.items():
            if v is not None:
                setattr(cfg, k, v)
        return cfg


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.name = f"{config.provider}:{config.model}"
        self._last_call = 0.0
        if config.provider != "ollama" and not config.api_key:
            preset = PROVIDERS[config.provider]
            raise LLMError(
                f"No API key for {config.provider}. Set {preset['key_env']} "
                "(or SCREENER_API_KEY), or use --provider ollama for a local model."
            )

    # -- caching ---------------------------------------------------------
    def _cache_path(self, system: str, user: str) -> Path | None:
        if not self.config.cache_dir:
            return None
        raw = json.dumps(
            [self.name, self.config.temperature, system, user], ensure_ascii=False
        ).encode()
        return Path(self.config.cache_dir) / (hashlib.sha256(raw).hexdigest() + ".json")

    # -- main call -------------------------------------------------------
    def complete(self, system: str, user: str) -> str:
        path = self._cache_path(system, user)
        if path and path.exists():
            return json.loads(path.read_text(encoding="utf-8"))["text"]

        text = self._request(system, user)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"text": text}, ensure_ascii=False), encoding="utf-8")
        return text

    def _request(self, system: str, user: str) -> str:
        cfg = self.config
        body = json.dumps({
            "model": cfg.model,
            "temperature": cfg.temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }).encode()
        headers = {"Content-Type": "application/json", "User-Agent": "screening-classifier/0.1"}
        if cfg.api_key:
            headers["Authorization"] = f"Bearer {cfg.api_key}"
        url = cfg.base_url.rstrip("/") + "/chat/completions"

        delay = 4.0
        last_err = ""
        for attempt in range(cfg.max_retries):
            wait = cfg.min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=cfg.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"] or ""
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:300]
                last_err = f"HTTP {e.code}: {detail}"
                if e.code in (429, 500, 502, 503, 504):
                    retry_after = e.headers.get("Retry-After")
                    sleep_for = float(retry_after) if retry_after and retry_after.isdigit() else delay
                    time.sleep(min(sleep_for, 90))
                    delay = min(delay * 2, 90)
                    continue
                raise LLMError(last_err) from e
            except (urllib.error.URLError, TimeoutError) as e:
                last_err = f"network error: {e}"
                time.sleep(delay)
                delay = min(delay * 2, 90)
        raise LLMError(f"gave up after {cfg.max_retries} attempts. Last error: {last_err}")
