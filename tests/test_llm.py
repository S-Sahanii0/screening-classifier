from unittest.mock import patch

from screener.llm import LLMClient, LLMConfig, LLMError


def test_cache_avoids_second_call(tmp_path):
    cfg = LLMConfig(provider="ollama", model="m", base_url="http://x", cache_dir=str(tmp_path))
    client = LLMClient(cfg)
    with patch.object(LLMClient, "_request", return_value="hello") as req:
        assert client.complete("sys", "user") == "hello"
        assert client.complete("sys", "user") == "hello"
        assert req.call_count == 1
        client.complete("sys", "different user")
        assert req.call_count == 2


def test_cache_can_be_disabled(tmp_path):
    cfg = LLMConfig(provider="ollama", model="m", base_url="http://x", cache_dir=None)
    client = LLMClient(cfg)
    with patch.object(LLMClient, "_request", return_value="hi") as req:
        client.complete("s", "u")
        client.complete("s", "u")
        assert req.call_count == 2


def test_missing_key_gives_clear_error():
    cfg = LLMConfig(provider="gemini", model="m", base_url="http://x", api_key=None)
    try:
        LLMClient(cfg)
    except LLMError as e:
        assert "GEMINI_API_KEY" in str(e)
    else:
        raise AssertionError("expected LLMError")


def test_from_env_presets(monkeypatch=None):
    import os
    old = {k: os.environ.pop(k, None) for k in ("SCREENER_PROVIDER", "SCREENER_MODEL", "SCREENER_BASE_URL", "SCREENER_API_KEY")}
    try:
        cfg = LLMConfig.from_env(provider="groq")
        assert cfg.provider == "groq" and "groq" in cfg.base_url
        cfg = LLMConfig.from_env(provider="ollama", min_interval=2.5)
        assert cfg.min_interval == 2.5 and cfg.base_url.startswith("http://localhost")
    finally:
        for k, v in old.items():
            if v is not None:
                os.environ[k] = v


def test_unknown_provider_rejected():
    try:
        LLMConfig.from_env(provider="nope")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
