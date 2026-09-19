"""Request coordinator and FastAPI surface.

``CacheFlowEngine`` holds the whole decision path (cache gate -> router ->
client) so the CLI and the HTTP app share one implementation.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import FastAPI, Response
from pydantic import BaseModel

from cacheflow.cache import DEFAULT_THRESHOLD, Embedder, SemanticCache
from cacheflow.client import LLMClient, cost_usd
from cacheflow.models import (
    CacheEntry,
    CacheMetrics,
    ProxyResponse,
    QueryRequest,
    RouteTier,
)
from cacheflow.router import ComplexityRouter


class PurgeRequest(BaseModel):
    """Body for ``POST /cache/purge``; ``namespace=None`` clears everything."""

    namespace: str | None = None


class PurgeResponse(BaseModel):
    purged: int
    namespace: str | None
    remaining: int


class CacheFlowEngine:
    """Cache gate, then complexity router, then the model client."""

    def __init__(
        self,
        cache: SemanticCache | None = None,
        router: ComplexityRouter | None = None,
        client: LLMClient | None = None,
        embedder: Embedder | None = None,
        threshold: float = DEFAULT_THRESHOLD,
        ttl_seconds: float | None = None,
        max_entries: int = 1024,
    ) -> None:
        self.cache = cache or SemanticCache(
            embedder=embedder,
            threshold=threshold,
            ttl_seconds=ttl_seconds,
            max_entries=max_entries,
        )
        self.router = router or ComplexityRouter()
        self.client = client or LLMClient()
        self._hits = 0
        self._misses = 0
        self._latency_saved_ms = 0.0
        self._cost_saved_usd = 0.0
        self._routed: dict[RouteTier, int] = {t: 0 for t in RouteTier}
        self._hit_latency_ms = 0.0
        self._miss_latency_ms = 0.0

    def _tier_of(self, model: str) -> RouteTier:
        for tier, name in self.client.models.items():
            if name == model:
                return tier
        return RouteTier.TIER_HEAVY  # unknown provenance: price conservatively

    def handle(self, request: QueryRequest) -> ProxyResponse:
        started = time.perf_counter()
        embedding = self.cache.embed(request.query)

        if not request.force_refresh:
            match = self.cache.lookup(
                request.query, request.namespace, embedding=embedding
            )
            if match is not None:
                entry, score = match
                latency_ms = (time.perf_counter() - started) * 1000.0
                saved_usd = cost_usd(entry.tokens, self._tier_of(entry.model_used))
                self._hits += 1
                self._hit_latency_ms += latency_ms
                self._latency_saved_ms += max(0.0, entry.latency_ms - latency_ms)
                self._cost_saved_usd += saved_usd
                return ProxyResponse(
                    response=entry.response,
                    cached=True,
                    similarity_score=round(score, 6),
                    model_selected=entry.model_used,
                    latency_ms=round(latency_ms, 3),
                    estimated_savings_usd=round(saved_usd, 8),
                    namespace=request.namespace,
                    rationale=f"semantic cache hit (similarity {score:.4f} "
                    f">= {self.cache.threshold})",
                )

        decision = self.router.route(request.query)
        result = self.client.complete(request.query, decision.tier, request.max_tokens)
        self._routed[decision.tier] += 1
        self._misses += 1

        self.cache.put(
            CacheEntry(
                query=request.query,
                embedding=embedding,
                response=result.text,
                model_used=result.model,
                latency_ms=result.latency_ms,
                tokens=result.tokens,
            ),
            namespace=request.namespace,
            embedding=embedding,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        self._miss_latency_ms += latency_ms
        return ProxyResponse(
            response=result.text,
            cached=False,
            similarity_score=None,
            model_selected=result.model,
            latency_ms=round(latency_ms, 3),
            estimated_savings_usd=0.0,
            namespace=request.namespace,
            rationale=f"cache miss -> {decision.tier.value} "
            f"(score {decision.complexity_score}): {decision.rationale}",
        )

    def metrics(self) -> CacheMetrics:
        total = self._hits + self._misses
        return CacheMetrics(
            total_requests=total,
            cache_hits=self._hits,
            cache_misses=self._misses,
            hit_rate=round(self._hits / total, 6) if total else 0.0,
            total_latency_saved_ms=round(self._latency_saved_ms, 3),
            total_cost_saved_usd=round(self._cost_saved_usd, 8),
            entries=len(self.cache),
            routed_cheap=self._routed[RouteTier.TIER_CHEAP],
            routed_heavy=self._routed[RouteTier.TIER_HEAVY],
            avg_hit_latency_ms=(
                round(self._hit_latency_ms / self._hits, 3) if self._hits else 0.0
            ),
            avg_miss_latency_ms=(
                round(self._miss_latency_ms / self._misses, 3) if self._misses else 0.0
            ),
        )

    def purge(self, namespace: str | None) -> int:
        return self.cache.clear(namespace)

    def reset_metrics(self) -> None:
        self._hits = self._misses = 0
        self._latency_saved_ms = self._cost_saved_usd = 0.0
        self._hit_latency_ms = self._miss_latency_ms = 0.0
        self._routed = {t: 0 for t in RouteTier}


def create_app(engine: CacheFlowEngine | None = None) -> FastAPI:
    """Build the proxy app around an engine (injectable for tests)."""
    app = FastAPI(
        title="cache-flow",
        version="0.1.0",
        summary="Semantic vector cache and cost-aware model router for LLM traffic",
    )
    app.state.engine = engine or CacheFlowEngine()

    def _engine() -> CacheFlowEngine:
        eng: CacheFlowEngine = app.state.engine
        return eng

    @app.post("/v1/chat", response_model=ProxyResponse)
    def chat(request: QueryRequest, response: Response) -> ProxyResponse:
        result = _engine().handle(request)
        response.headers["X-CacheFlow-Cache"] = "HIT" if result.cached else "MISS"
        response.headers["X-CacheFlow-Model"] = result.model_selected
        response.headers["X-CacheFlow-Latency-Ms"] = f"{result.latency_ms:.3f}"
        if result.similarity_score is not None:
            response.headers["X-CacheFlow-Similarity"] = f"{result.similarity_score:.4f}"
        return result

    @app.get("/metrics", response_model=CacheMetrics)
    def metrics() -> CacheMetrics:
        return _engine().metrics()

    @app.post("/cache/purge", response_model=PurgeResponse)
    def purge(request: PurgeRequest) -> PurgeResponse:
        eng = _engine()
        removed = eng.purge(request.namespace)
        return PurgeResponse(
            purged=removed, namespace=request.namespace, remaining=len(eng.cache)
        )

    @app.get("/health")
    def health() -> dict[str, Any]:
        eng = _engine()
        return {
            "status": "ok",
            "embedder": eng.cache.embedder.name,
            "live_llm": eng.client.live,
            "threshold": eng.cache.threshold,
            "entries": len(eng.cache),
        }

    return app


app = create_app()
