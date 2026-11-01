import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from screener import api  # noqa: E402

from fakes import FakeClient  # noqa: E402


@pytest.fixture()
def client():
    api.get_client.cache_clear()
    api.get_client = lambda: FakeClient("A")  # type: ignore[assignment]
    return TestClient(api.app)


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_classify_ok(client):
    r = client.post("/classify", json={
        "title": "A paper about fairness", "abstract": "We train a model and report subgroup results."})
    assert r.status_code == 200
    body = r.json()
    assert body["verdict"] == "A" and body["provisional"] is True


def test_rejects_short_abstract(client):
    r = client.post("/classify", json={"title": "Title here", "abstract": "short"})
    assert r.status_code == 422


def test_rejects_unknown_variant(client):
    r = client.post("/classify", json={
        "title": "A paper about fairness", "abstract": "We train a model and report subgroup results.",
        "variant": "nope"})
    assert r.status_code == 422
