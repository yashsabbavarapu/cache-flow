"""Semantic cache: retrieval, the 0.94 boundary, TTL and LRU eviction."""

from __future__ import annotations

import math
import time

import pytest

from cacheflow.cache import (
    DEFAULT_THRESHOLD,
    GeminiEmbedder,
    LexicalEmbedder,
    MemoizingEmbedder,
    SemanticCache,
    build_embedder,
    cosine_similarity,
    dig,
)
from tests.conftest import StubEmbedder, make_entry


def test_exact_repeat_hits(cache: SemanticCache) -> None:
    cache.put(make_entry("How do I reset my password?", response="Use the reset link."))
    match = cache.lookup("How do I reset my password?")
    assert match is not None
    entry, score = match
    assert entry.response == "Use the reset link."
    assert score == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize(
    "paraphrase",
    [
        "How can I reset my password?",
        "How do I reset the password?",
        "How do I reset my password",
        "How do I reset my password???",
        "HOW DO I RESET MY PASSWORD?",
    ],
)
def test_paraphrase_hits(cache: SemanticCache, paraphrase: str) -> None:
    """Function-word, casing and punctuation variants must reuse the entry."""
    cache.put(make_entry("How do I reset my password?"))
    assert cache.lookup(paraphrase) is not None


def test_contraction_paraphrase_hits(cache: SemanticCache) -> None:
    cache.put(make_entry("What is the refund window for online orders?"))
    assert cache.lookup("What's the refund window for online orders?") is not None


@pytest.mark.parametrize(
    "unrelated",
    [
        "Write a Rust web server with graceful shutdown",
        "What are your support hours?",
        "Explain the CAP theorem",
    ],
)
def test_dissimilar_queries_miss(cache: SemanticCache, unrelated: str) -> None:
    cache.put(make_entry("How do I reset my password?"))
    assert cache.lookup(unrelated) is None


def test_negation_does_not_hit_affirmative(cache: SemanticCache) -> None:
    """A safety property: flipping the meaning must not serve the cached answer."""
    cache.put(make_entry("Do reset my password", response="Resetting now."))
    assert cache.lookup("Don't reset my password") is None


def test_same_topic_different_intent_misses(cache: SemanticCache) -> None:
    cache.put(make_entry("How do I reset my password?"))
    assert cache.lookup("Why was my password reset without my consent?") is None


def test_threshold_boundary_093_misses_and_095_hits() -> None:
    """0.93 is below the 0.94 gate; 0.95 is above it."""
    angles = {
        "anchor": 0.0,
        "just_below": math.acos(0.93),
        "just_above": math.acos(0.95),
    }
    cache = SemanticCache(embedder=StubEmbedder(angles), threshold=DEFAULT_THRESHOLD)
    cache.put(make_entry("anchor"))

    assert cache.lookup("just_below") is None

    match = cache.lookup("just_above")
    assert match is not None
    assert match[1] == pytest.approx(0.95, abs=1e-5)


def test_threshold_is_inclusive_at_exactly_094() -> None:
    cache = SemanticCache(
        embedder=StubEmbedder({"anchor": 0.0, "exact": math.acos(DEFAULT_THRESHOLD)}),
        threshold=DEFAULT_THRESHOLD,
    )
    cache.put(make_entry("anchor"))
    match = cache.lookup("exact")
    assert match is not None
    assert match[1] == pytest.approx(DEFAULT_THRESHOLD, abs=1e-5)


def test_ttl_expiration_drops_entry(cache: SemanticCache) -> None:
    cache.ttl_seconds = 60.0
    stale = make_entry("How do I reset my password?", timestamp=time.time() - 3600)
    cache.put(stale)
    assert cache.lookup("How do I reset my password?") is None
    assert len(cache) == 0


def test_entry_within_ttl_survives(cache: SemanticCache) -> None:
    cache.ttl_seconds = 3600.0
    cache.put(make_entry("How do I reset my password?", timestamp=time.time() - 5))
    assert cache.lookup("How do I reset my password?") is not None


def test_lru_evicts_least_recently_used() -> None:
    cache = SemanticCache(embedder=LexicalEmbedder(), max_entries=2)
    cache.put(make_entry("alpha topic about invoices"))
    cache.put(make_entry("beta topic about shipping"))
    cache.put(make_entry("gamma topic about refunds"))

    assert len(cache) == 2
    assert cache.evictions == 1
    assert cache.lookup("alpha topic about invoices") is None
    assert cache.lookup("gamma topic about refunds") is not None


def test_read_refreshes_lru_recency() -> None:
    cache = SemanticCache(embedder=LexicalEmbedder(), max_entries=2)
    cache.put(make_entry("alpha topic about invoices"))
    cache.put(make_entry("beta topic about shipping"))

    # Touch alpha so beta becomes the eviction candidate.
    assert cache.lookup("alpha topic about invoices") is not None
    cache.put(make_entry("gamma topic about refunds"))

    assert cache.lookup("alpha topic about invoices") is not None
    assert cache.lookup("beta topic about shipping") is None


def test_evict_by_key(cache: SemanticCache) -> None:
    key = cache.put(make_entry("How do I reset my password?"))
    assert cache.evict(key) is True
    assert cache.evict(key) is False
    assert len(cache) == 0


def test_clear_namespace_leaves_others(cache: SemanticCache) -> None:
    cache.put(make_entry("How do I reset my password?"), namespace="support")
    cache.put(make_entry("How do I reset my password?"), namespace="billing")

    assert cache.clear("support") == 1
    assert cache.lookup("How do I reset my password?", namespace="support") is None
    assert cache.lookup("How do I reset my password?", namespace="billing") is not None


def test_clear_all(cache: SemanticCache) -> None:
    cache.put(make_entry("one question about invoices"), namespace="a")
    cache.put(make_entry("two question about shipping"), namespace="b")
    assert cache.clear(None) == 2
    assert len(cache) == 0


def test_namespaces_are_isolated(cache: SemanticCache) -> None:
    cache.put(make_entry("How do I reset my password?"), namespace="tenant-a")
    assert cache.lookup("How do I reset my password?", namespace="tenant-b") is None


def test_embeddings_are_unit_norm_and_deterministic(embedder: LexicalEmbedder) -> None:
    first = embedder.embed("How do I reset my password?")
    second = embedder.embed("How do I reset my password?")
    assert first == second
    assert math.sqrt(sum(v * v for v in first)) == pytest.approx(1.0, abs=1e-5)


def test_cosine_similarity_edge_cases() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_empty_cache_lookup_returns_none(cache: SemanticCache) -> None:
    assert cache.lookup("anything at all") is None


def test_rejects_zero_capacity() -> None:
    with pytest.raises(ValueError):
        SemanticCache(embedder=LexicalEmbedder(), max_entries=0)


class CountingEmbedder:
    """Counts how many times the expensive inner call is actually made."""

    def __init__(self) -> None:
        self.name = "counting"
        self.calls = 0
        self._inner = LexicalEmbedder()

    def embed(self, text: str) -> list[float]:
        self.calls += 1
        return self._inner.embed(text)


def test_memoizing_embedder_avoids_repeat_calls() -> None:
    """A repeated string must not cost a second (remote) embedding call."""
    inner = CountingEmbedder()
    memo = MemoizingEmbedder(inner)

    first = memo.embed("How do I reset my password?")
    second = memo.embed("How do I reset my password?")

    assert first == second
    assert inner.calls == 1
    assert (memo.hits, memo.misses) == (1, 1)


def test_memoizing_embedder_still_embeds_new_text() -> None:
    inner = CountingEmbedder()
    memo = MemoizingEmbedder(inner)
    memo.embed("first question about invoices")
    memo.embed("second question about shipping")
    assert inner.calls == 2
    assert memo.misses == 2


def test_memoizing_embedder_is_bounded_and_lru() -> None:
    inner = CountingEmbedder()
    memo = MemoizingEmbedder(inner, max_entries=2)
    memo.embed("alpha")
    memo.embed("beta")
    memo.embed("alpha")      # refresh alpha's recency
    memo.embed("gamma")      # evicts beta, not alpha

    assert len(memo._memo) == 2
    before = inner.calls
    memo.embed("alpha")
    assert inner.calls == before        # alpha survived
    memo.embed("beta")
    assert inner.calls == before + 1    # beta was evicted


def test_memoizing_embedder_preserves_inner_name() -> None:
    assert MemoizingEmbedder(CountingEmbedder()).name == "counting"


def test_gemini_embedder_sends_task_type_and_dimensions() -> None:
    """taskType is load-bearing: without it paraphrase/distinct bands overlap."""
    embedder = GeminiEmbedder("fake-key")
    assert embedder.task_type == "SEMANTIC_SIMILARITY"
    assert embedder.dimensions == 768
    assert embedder.model == "gemini-embedding-001"
    assert "gemini-embedding-001:embedContent" in embedder.ENDPOINT.format(
        model=embedder.model
    )


def test_build_embedder_is_offline_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert isinstance(build_embedder(), LexicalEmbedder)


def test_build_embedder_memoizes_live_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    embedder = build_embedder()
    assert isinstance(embedder, MemoizingEmbedder)
    assert isinstance(embedder.inner, GeminiEmbedder)


def test_dig_walks_and_rejects_bad_paths() -> None:
    payload = {"candidates": [{"content": {"parts": [{"text": "hi"}]}}]}
    assert dig(payload, "candidates", 0, "content", "parts", 0, "text") == "hi"
    with pytest.raises(ValueError):
        dig(payload, "candidates", 5)
    with pytest.raises(ValueError):
        dig(payload, "nope")
