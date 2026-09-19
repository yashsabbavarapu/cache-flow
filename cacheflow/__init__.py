"""cache-flow: semantic vector cache and cost-aware model router for LLM traffic."""

from cacheflow.cache import (
    DEFAULT_THRESHOLD,
    GeminiEmbedder,
    LexicalEmbedder,
    MemoizingEmbedder,
    SemanticCache,
    build_embedder,
    cosine_similarity,
)
from cacheflow.client import PRICING_USD_PER_1K, LLMClient, cost_usd
from cacheflow.models import (
    CacheEntry,
    CacheMetrics,
    ProxyResponse,
    QueryRequest,
    RouteDecision,
    RouteTier,
)
from cacheflow.proxy import CacheFlowEngine, create_app
from cacheflow.router import ComplexityRouter

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_THRESHOLD",
    "PRICING_USD_PER_1K",
    "CacheEntry",
    "CacheFlowEngine",
    "CacheMetrics",
    "ComplexityRouter",
    "GeminiEmbedder",
    "LLMClient",
    "LexicalEmbedder",
    "MemoizingEmbedder",
    "ProxyResponse",
    "QueryRequest",
    "RouteDecision",
    "RouteTier",
    "SemanticCache",
    "build_embedder",
    "cosine_similarity",
    "cost_usd",
    "create_app",
]
