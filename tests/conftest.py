"""Shared fixtures. Every test runs offline and deterministically."""

from __future__ import annotations

import math

import pytest

from cacheflow.cache import LexicalEmbedder, SemanticCache
from cacheflow.client import LLMClient
from cacheflow.models import CacheEntry
from cacheflow.proxy import CacheFlowEngine
from cacheflow.router import CHEAP_MODEL


class StubEmbedder:
    """Maps texts to hand-placed unit vectors so cosine scores are exact.

    Unknown texts land on an orthogonal axis, so they can never accidentally
    match the planted vectors.
    """

    def __init__(self, angles: dict[str, float]) -> None:
        self.name = "stub"
        self.angles = angles

    def embed(self, text: str) -> list[float]:
        if text not in self.angles:
            return [0.0, 0.0, 1.0]
        theta = self.angles[text]
        return [math.cos(theta), math.sin(theta), 0.0]


@pytest.fixture
def embedder() -> LexicalEmbedder:
    return LexicalEmbedder()


@pytest.fixture
def cache(embedder: LexicalEmbedder) -> SemanticCache:
    return SemanticCache(embedder=embedder)


@pytest.fixture
def offline_client() -> LLMClient:
    return LLMClient(api_key="", simulate_latency=False)


@pytest.fixture
def engine(cache: SemanticCache, offline_client: LLMClient) -> CacheFlowEngine:
    return CacheFlowEngine(cache=cache, client=offline_client)


def make_entry(
    query: str,
    response: str = "cached answer",
    model: str = CHEAP_MODEL,
    latency_ms: float = 250.0,
    tokens: int = 120,
    timestamp: float | None = None,
) -> CacheEntry:
    entry = CacheEntry(
        query=query,
        embedding=[],
        response=response,
        model_used=model,
        latency_ms=latency_ms,
        tokens=tokens,
    )
    if timestamp is not None:
        entry.timestamp = timestamp
    return entry
