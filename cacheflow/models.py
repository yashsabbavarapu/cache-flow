"""Pydantic v2 data contracts shared across the proxy, cache and router."""

from __future__ import annotations

import time
from enum import Enum

from pydantic import BaseModel, Field


class RouteTier(str, Enum):
    """Model tier a cache-miss query is dispatched to."""

    TIER_CHEAP = "tier_cheap"
    TIER_HEAVY = "tier_heavy"


class QueryRequest(BaseModel):
    """Inbound completion request handed to the proxy."""

    query: str = Field(min_length=1)
    namespace: str = "default"
    max_tokens: int = Field(default=500, gt=0)
    force_refresh: bool = False


class CacheEntry(BaseModel):
    """A stored completion plus the embedding used to retrieve it."""

    query: str
    embedding: list[float]
    response: str
    model_used: str
    latency_ms: float
    timestamp: float = Field(default_factory=time.time)
    tokens: int = 0

    def is_expired(self, ttl_seconds: float | None, now: float | None = None) -> bool:
        if ttl_seconds is None:
            return False
        return (time.time() if now is None else now) - self.timestamp > ttl_seconds


class RouteDecision(BaseModel):
    """Outcome of the complexity heuristic for a cache miss."""

    tier: RouteTier
    model: str
    complexity_score: float
    rationale: str


class ProxyResponse(BaseModel):
    """What `POST /v1/chat` returns to the caller."""

    response: str
    cached: bool
    similarity_score: float | None = None
    model_selected: str
    latency_ms: float
    estimated_savings_usd: float = 0.0
    namespace: str = "default"
    rationale: str | None = None


class CacheMetrics(BaseModel):
    """Aggregated counters exported by `GET /metrics`."""

    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    hit_rate: float = 0.0
    total_latency_saved_ms: float = 0.0
    total_cost_saved_usd: float = 0.0
    entries: int = 0
    routed_cheap: int = 0
    routed_heavy: int = 0
    avg_hit_latency_ms: float = 0.0
    avg_miss_latency_ms: float = 0.0
