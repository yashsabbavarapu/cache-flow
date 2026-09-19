"""Cost-aware model router.

Cache misses are scored against transparent, auditable heuristics rather than a
classifier model -- routing must not itself cost a model call, and an operator
must be able to read the rationale and explain why a query went where it did.
"""

from __future__ import annotations

import re

from cacheflow.models import RouteDecision, RouteTier

CHEAP_MODEL = "gemini-2.0-flash"
HEAVY_MODEL = "gemini-2.5-pro"

#: Score at or above which a query is escalated to the heavy tier.
ESCALATION_THRESHOLD = 1.0

SHORT_QUERY_TOKENS = 50
LONG_QUERY_TOKENS = 150

_SYNTHESIS_KEYWORDS = frozenset(
    """implement refactor algorithm debug optimize architect benchmark migrate
    profile diagnose derive prove analyze critique synthesize design integrate
    troubleshoot rewrite parallelize instrument""".split()
)

_REASONING_PHRASES = (
    "step by step",
    "step-by-step",
    "compare and contrast",
    "walk me through",
    "trade-off",
    "tradeoff",
    "pros and cons",
    "root cause",
    "from first principles",
    "edge case",
    "why does",
    "explain why",
)

_STRUCTURE_PATTERNS = (
    re.compile(r"```"),                       # inline code block
    re.compile(r"^\s*\d+[.)]\s", re.M),       # numbered instruction list
    re.compile(r"\bas (?:valid )?json\b", re.I),
    re.compile(r"\b(?:markdown )?table\b", re.I),
    re.compile(r"\bschema\b", re.I),
)

_WORD_RE = re.compile(r"\S+")


def estimate_tokens(text: str) -> int:
    """Whitespace-word count scaled by the usual ~1.3 tokens-per-word ratio."""
    return max(1, round(len(_WORD_RE.findall(text)) * 1.3))


class ComplexityRouter:
    """Scores a query and picks a tier, returning the reasoning alongside it."""

    def __init__(
        self,
        cheap_model: str = CHEAP_MODEL,
        heavy_model: str = HEAVY_MODEL,
        escalation_threshold: float = ESCALATION_THRESHOLD,
    ) -> None:
        self.cheap_model = cheap_model
        self.heavy_model = heavy_model
        self.escalation_threshold = escalation_threshold

    def route(self, query: str) -> RouteDecision:
        tokens = estimate_tokens(query)
        lowered = query.lower()
        score = 0.0
        reasons: list[str] = []

        if tokens > LONG_QUERY_TOKENS:
            score += 1.5
            reasons.append(f"long prompt ({tokens} est. tokens > {LONG_QUERY_TOKENS})")

        hits = sorted(_SYNTHESIS_KEYWORDS.intersection(re.findall(r"[a-z]+", lowered)))
        if hits:
            score += 1.0 + 0.25 * (len(hits) - 1)
            reasons.append(f"synthesis keyword(s): {', '.join(hits)}")

        phrases = [p for p in _REASONING_PHRASES if p in lowered]
        if phrases:
            score += 1.0
            reasons.append(f"multi-step reasoning cue(s): {', '.join(phrases)}")

        structural = [p.pattern for p in _STRUCTURE_PATTERNS if p.search(query)]
        if structural:
            score += 0.75
            reasons.append(f"structural constraint ({len(structural)} pattern(s))")

        clauses = len([s for s in re.split(r"[.?!\n]+", query) if s.strip()])
        if clauses >= 3:
            score += 0.5
            reasons.append(f"multi-part request ({clauses} clauses)")

        # Brevity is evidence of simplicity only when nothing else fired: a short
        # "Refactor this algorithm" must still escalate.
        if score == 0.0 and tokens < SHORT_QUERY_TOKENS:
            reasons.append(f"short prompt ({tokens} est. tokens < {SHORT_QUERY_TOKENS})")

        if score >= self.escalation_threshold:
            tier, model = RouteTier.TIER_HEAVY, self.heavy_model
        else:
            tier, model = RouteTier.TIER_CHEAP, self.cheap_model
            if not reasons:
                reasons.append("no complexity signals detected")

        return RouteDecision(
            tier=tier,
            model=model,
            complexity_score=round(score, 3),
            rationale="; ".join(reasons),
        )
