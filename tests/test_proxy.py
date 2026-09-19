"""End-to-end proxy behaviour: headers, latency, metrics and purging."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from cacheflow.cache import LexicalEmbedder, SemanticCache
from cacheflow.client import LLMClient
from cacheflow.proxy import CacheFlowEngine, create_app

HIT_LATENCY_BUDGET_MS = 15.0


@pytest.fixture
def client(engine: CacheFlowEngine) -> TestClient:
    return TestClient(create_app(engine))


def ask(client: TestClient, query: str, **kwargs: object) -> dict[str, object]:
    response = client.post("/v1/chat", json={"query": query, **kwargs})
    assert response.status_code == 200, response.text
    body: dict[str, object] = response.json()
    return body


def test_cold_query_misses_then_paraphrase_hits(client: TestClient) -> None:
    first = ask(client, "How do I reset my password?")
    assert first["cached"] is False
    assert first["similarity_score"] is None

    second = ask(client, "How can I reset my password?")
    assert second["cached"] is True
    assert second["response"] == first["response"]
    assert float(second["similarity_score"]) >= 0.94  # type: ignore[arg-type]


def test_response_headers_report_cache_state(client: TestClient) -> None:
    cold = client.post("/v1/chat", json={"query": "What are your support hours?"})
    assert cold.headers["X-CacheFlow-Cache"] == "MISS"
    assert cold.headers["X-CacheFlow-Model"] == "gemini-2.0-flash"
    assert "X-CacheFlow-Similarity" not in cold.headers
    assert float(cold.headers["X-CacheFlow-Latency-Ms"]) >= 0.0

    warm = client.post("/v1/chat", json={"query": "What are the support hours?"})
    assert warm.headers["X-CacheFlow-Cache"] == "HIT"
    assert float(warm.headers["X-CacheFlow-Similarity"]) >= 0.94


def test_cache_hit_is_served_within_latency_budget(client: TestClient) -> None:
    ask(client, "What is the refund window for online orders?")
    started = time.perf_counter()
    hit = ask(client, "What is the refund window on online orders?")
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    assert hit["cached"] is True
    assert float(hit["latency_ms"]) < HIT_LATENCY_BUDGET_MS  # type: ignore[arg-type]
    assert elapsed_ms < 200.0  # includes HTTP/serialization overhead


def test_complex_query_routes_to_heavy_tier(client: TestClient) -> None:
    result = ask(client, "Implement a thread-safe LRU cache in Rust with generics")
    assert result["model_selected"] == "gemini-2.5-pro"
    assert "tier_heavy" in str(result["rationale"])


def test_force_refresh_bypasses_the_cache(client: TestClient) -> None:
    ask(client, "How do I cancel my subscription?")
    refreshed = ask(client, "How do I cancel my subscription?", force_refresh=True)
    assert refreshed["cached"] is False


def test_namespaces_do_not_share_entries(client: TestClient) -> None:
    ask(client, "How do I reset my password?", namespace="tenant-a")
    other = ask(client, "How do I reset my password?", namespace="tenant-b")
    assert other["cached"] is False


def test_metrics_accumulate_across_requests(client: TestClient) -> None:
    empty = client.get("/metrics").json()
    assert empty["total_requests"] == 0
    assert empty["hit_rate"] == 0.0

    ask(client, "How do I reset my password?")
    ask(client, "How can I reset my password?")
    ask(client, "How do I reset the password?")
    ask(client, "Implement a distributed rate limiter")

    metrics = client.get("/metrics").json()
    assert metrics["total_requests"] == 4
    assert metrics["cache_hits"] == 2
    assert metrics["cache_misses"] == 2
    assert metrics["hit_rate"] == pytest.approx(0.5)
    assert metrics["routed_cheap"] == 1
    assert metrics["routed_heavy"] == 1
    assert metrics["entries"] == 2
    assert metrics["total_cost_saved_usd"] > 0.0


def test_latency_saved_accrues_against_real_model_latency() -> None:
    """With simulated model latency, a hit must bank measurable time."""
    engine = CacheFlowEngine(
        cache=SemanticCache(embedder=LexicalEmbedder()),
        client=LLMClient(api_key="", simulate_latency=True),
    )
    client = TestClient(create_app(engine))
    ask(client, "What are your support hours?")
    ask(client, "What are the support hours?")

    metrics = client.get("/metrics").json()
    assert metrics["cache_hits"] == 1
    assert metrics["total_latency_saved_ms"] > 40.0
    assert metrics["avg_hit_latency_ms"] < HIT_LATENCY_BUDGET_MS


def test_purge_single_namespace(client: TestClient) -> None:
    ask(client, "How do I reset my password?", namespace="support")
    ask(client, "How do I reset my password?", namespace="billing")

    purged = client.post("/cache/purge", json={"namespace": "support"}).json()
    assert purged["purged"] == 1
    assert purged["remaining"] == 1

    assert ask(client, "How do I reset my password?", namespace="support")["cached"] is False
    assert ask(client, "How do I reset my password?", namespace="billing")["cached"] is True


def test_purge_everything(client: TestClient) -> None:
    ask(client, "How do I reset my password?")
    ask(client, "What are your support hours?")

    purged = client.post("/cache/purge", json={}).json()
    assert purged["purged"] == 2
    assert purged["remaining"] == 0
    assert client.get("/metrics").json()["entries"] == 0


def test_health_reports_configuration(client: TestClient) -> None:
    health = client.get("/health").json()
    assert health["status"] == "ok"
    assert health["embedder"] == "lexical-hash-512"
    assert health["live_llm"] is False
    assert health["threshold"] == 0.94


def test_empty_query_is_rejected(client: TestClient) -> None:
    assert client.post("/v1/chat", json={"query": ""}).status_code == 422


def test_invalid_max_tokens_is_rejected(client: TestClient) -> None:
    response = client.post("/v1/chat", json={"query": "hello", "max_tokens": 0})
    assert response.status_code == 422
