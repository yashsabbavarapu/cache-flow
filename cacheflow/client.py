"""LLM execution layer: live Gemini calls, or a deterministic offline stand-in."""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass

from cacheflow.cache import dig
from cacheflow.models import RouteTier
from cacheflow.router import CHEAP_MODEL, HEAVY_MODEL

#: USD per 1000 tokens. The heavy rate is the $0.005/1k figure the savings math
#: in `/metrics` is quoted against.
PRICING_USD_PER_1K: dict[RouteTier, float] = {
    RouteTier.TIER_CHEAP: 0.0001,
    RouteTier.TIER_HEAVY: 0.005,
}

#: Latency the offline mock simulates per tier, so benchmarks show a realistic
#: cache-hit-vs-miss gap without burning API credits. Fake by construction.
MOCK_LATENCY_MS: dict[RouteTier, float] = {
    RouteTier.TIER_CHEAP: 60.0,
    RouteTier.TIER_HEAVY: 240.0,
}

_GENERATE_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)


@dataclass(slots=True)
class CompletionResult:
    text: str
    model: str
    tier: RouteTier
    latency_ms: float
    tokens: int
    live: bool


def cost_usd(tokens: int, tier: RouteTier) -> float:
    return (tokens / 1000.0) * PRICING_USD_PER_1K[tier]


class LLMClient:
    """Unified completion interface across tiers.

    Calls Gemini when ``GEMINI_API_KEY`` is present; otherwise returns a
    deterministic synthetic completion so the whole system runs offline.
    """

    def __init__(
        self,
        api_key: str | None = None,
        cheap_model: str = CHEAP_MODEL,
        heavy_model: str = HEAVY_MODEL,
        simulate_latency: bool = True,
        timeout: float = 30.0,
    ) -> None:
        key = api_key if api_key is not None else os.environ.get("GEMINI_API_KEY", "")
        self.api_key = key.strip()
        self.models = {
            RouteTier.TIER_CHEAP: cheap_model,
            RouteTier.TIER_HEAVY: heavy_model,
        }
        self.simulate_latency = simulate_latency
        self.timeout = timeout

    @property
    def live(self) -> bool:
        return bool(self.api_key)

    def complete(
        self, query: str, tier: RouteTier, max_tokens: int = 500
    ) -> CompletionResult:
        model = self.models[tier]
        started = time.perf_counter()
        if self.live:
            try:
                text = self._call_gemini(query, model, max_tokens)
                live = True
            except Exception as exc:  # network/quota failure must not drop traffic
                text = self._mock_text(query, model, note=f"live call failed: {exc}")
                live = False
        else:
            text = self._mock_text(query, model)
            live = False
            if self.simulate_latency:
                time.sleep(MOCK_LATENCY_MS[tier] / 1000.0)

        latency_ms = (time.perf_counter() - started) * 1000.0
        tokens = estimate_completion_tokens(query, text)
        return CompletionResult(
            text=text,
            model=model,
            tier=tier,
            latency_ms=latency_ms,
            tokens=tokens,
            live=live,
        )

    def _call_gemini(self, query: str, model: str, max_tokens: int) -> str:
        import httpx

        response = httpx.post(
            _GENERATE_ENDPOINT.format(model=model),
            json={
                "contents": [{"parts": [{"text": query}]}],
                "generationConfig": {"maxOutputTokens": max_tokens},
            },
            headers={"x-goog-api-key": self.api_key},
            timeout=self.timeout,
        )
        response.raise_for_status()
        parts = dig(response.json(), "candidates", 0, "content", "parts")
        if not isinstance(parts, list):
            raise ValueError("candidate carried no parts")
        chunks = [p["text"] for p in parts if isinstance(p, dict) and "text" in p]
        if not chunks:
            raise ValueError("candidate contained no text")
        return "".join(str(c) for c in chunks).strip()

    @staticmethod
    def _mock_text(query: str, model: str, note: str | None = None) -> str:
        digest = hashlib.sha256(query.strip().lower().encode("utf-8")).hexdigest()[:12]
        suffix = f" [{note}]" if note else ""
        return (
            f"[mock:{model}] Deterministic stand-in completion for "
            f"{query.strip()!r} (id={digest}).{suffix}"
        )


def estimate_completion_tokens(prompt: str, completion: str) -> int:
    """Rough combined prompt+completion token count (~4 chars per token)."""
    return max(1, (len(prompt) + len(completion)) // 4)
