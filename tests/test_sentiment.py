import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    # Use context manager so FastAPI lifespan runs (loads VADER + dataset)
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "vader" in data["model"].lower()


def test_positive_headline(client):
    r = client.post("/api/v1/analyze", json={"headline": "Company reports record quarterly profits"})
    assert r.status_code == 200
    data = r.json()
    assert data["sentiment"] == "positive"
    assert 0.0 < data["confidence"] <= 1.0
    assert set(data["scores"].keys()) == {"positive", "negative", "neutral"}


def test_negative_headline(client):
    r = client.post(
        "/api/v1/analyze",
        json={"headline": "Firm files for bankruptcy amid mounting debt crisis"},
    )
    assert r.status_code == 200
    assert r.json()["sentiment"] == "negative"


def test_neutral_headline(client):
    r = client.post(
        "/api/v1/analyze",
        json={"headline": "The board of directors will hold its annual general meeting next month"},
    )
    assert r.status_code == 200
    assert r.json()["sentiment"] == "neutral"


def test_get_endpoint(client):
    r = client.get("/api/v1/analyze", params={"headline": "Earnings beat analyst forecasts"})
    assert r.status_code == 200
    assert r.json()["sentiment"] in {"positive", "negative", "neutral"}


def test_empty_headline_rejected(client):
    r = client.post("/api/v1/analyze", json={"headline": "   "})
    assert r.status_code == 422


def test_batch_analysis(client):
    headlines = [
        "Profits soar as company delivers outstanding record results",
        "CEO arrested on fraud charges",
        "Company schedules investor day",
    ]
    r = client.post("/api/v1/analyze/batch", json={"headlines": headlines})
    assert r.status_code == 200
    data = r.json()
    assert len(data["results"]) == 3
    assert "vader" in data["model_used"].lower()
    sentiments = [item["sentiment"] for item in data["results"]]
    assert sentiments[0] == "positive"
    assert sentiments[1] == "negative"


def test_batch_limit_enforced(client):
    r = client.post("/api/v1/analyze/batch", json={"headlines": ["x"] * 51})
    assert r.status_code == 422


def test_batch_empty_rejected(client):
    r = client.post("/api/v1/analyze/batch", json={"headlines": []})
    assert r.status_code == 422


def test_samples_endpoint(client):
    r = client.get("/api/v1/samples", params={"count": 5})
    assert r.status_code == 200
    data = r.json()
    assert len(data["samples"]) == 5
    assert data["total_available"] > 0
    for sample in data["samples"]:
        assert sample["label"] in {"positive", "negative", "neutral"}


def test_samples_filter_by_sentiment(client):
    for sentiment in ("positive", "negative", "neutral"):
        r = client.get("/api/v1/samples", params={"count": 3, "sentiment": sentiment})
        assert r.status_code == 200
        for sample in r.json()["samples"]:
            assert sample["label"] == sentiment


def test_samples_invalid_sentiment_rejected(client):
    r = client.get("/api/v1/samples", params={"sentiment": "bullish"})
    assert r.status_code == 422
