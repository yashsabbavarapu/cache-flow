"""Complexity routing: cheap vs heavy tier selection and rationale."""

from __future__ import annotations

import pytest

from cacheflow.models import RouteTier
from cacheflow.router import ComplexityRouter, estimate_tokens

router = ComplexityRouter()


@pytest.mark.parametrize(
    "query",
    [
        "What is the capital of France?",
        "How do I reset my password?",
        "What are your support hours?",
        "Where do I download the desktop app?",
        "Summarize this in one sentence",
        "Write me a haiku about the sea",
        "Is the API down?",
        "Translate 'good morning' into Spanish",
    ],
)
def test_simple_queries_route_cheap(query: str) -> None:
    decision = router.route(query)
    assert decision.tier is RouteTier.TIER_CHEAP
    assert decision.model == router.cheap_model


@pytest.mark.parametrize(
    "query",
    [
        "Implement a thread-safe LRU cache in Rust with generics",
        "Refactor this module to remove the circular import",
        "Debug why my async handler deadlocks under load",
        "Explain the algorithm behind consistent hashing",
        "Optimize this query plan for a 40M row table",
        "Compare and contrast optimistic and pessimistic locking",
        "Walk me through a proof of the CAP theorem step by step",
        "What are the trade-offs between gRPC and REST for internal services?",
    ],
)
def test_complex_queries_route_heavy(query: str) -> None:
    decision = router.route(query)
    assert decision.tier is RouteTier.TIER_HEAVY
    assert decision.model == router.heavy_model


def test_long_query_escalates_on_length_alone() -> None:
    """No keywords, just bulk: length alone crosses the escalation threshold."""
    query = " ".join(["context"] * 200)
    decision = router.route(query)
    assert decision.tier is RouteTier.TIER_HEAVY
    assert "long prompt" in decision.rationale


def test_short_synthesis_query_still_escalates() -> None:
    """Brevity must not cancel a genuine complexity signal."""
    decision = router.route("Refactor the algorithm")
    assert decision.tier is RouteTier.TIER_HEAVY
    assert "synthesis keyword" in decision.rationale


def test_structural_constraints_raise_score() -> None:
    plain = router.route("List the top five users")
    structured = router.route("List the top five users as JSON matching this schema")
    assert structured.complexity_score > plain.complexity_score


def test_code_block_is_a_structural_signal() -> None:
    decision = router.route("Fix this:\n```python\nprint(1/0)\n```")
    assert "structural constraint" in decision.rationale


def test_multi_part_request_raises_score() -> None:
    decision = router.route(
        "Summarize the log. Then list the failing hosts. Finally draft an update."
    )
    assert "multi-part request" in decision.rationale


def test_every_decision_carries_a_rationale() -> None:
    for query in ["Hi there", "Implement a parser", " ".join(["x"] * 200)]:
        decision = router.route(query)
        assert decision.rationale.strip()
        assert isinstance(decision.complexity_score, float)


def test_routing_is_deterministic() -> None:
    query = "Debug the flaky integration test"
    assert router.route(query) == router.route(query)


def test_threshold_is_configurable() -> None:
    """An operator who wants to spend less can raise the escalation bar."""
    strict = ComplexityRouter(escalation_threshold=5.0)
    assert strict.route("Implement a parser").tier is RouteTier.TIER_CHEAP


def test_custom_model_names_are_used() -> None:
    custom = ComplexityRouter(cheap_model="flash-x", heavy_model="pro-x")
    assert custom.route("Hi").model == "flash-x"
    assert custom.route("Implement a parser").model == "pro-x"


def test_estimate_tokens_scales_with_length() -> None:
    assert estimate_tokens("one") == 1
    assert estimate_tokens(" ".join(["word"] * 100)) == 130
    assert estimate_tokens("") == 1
