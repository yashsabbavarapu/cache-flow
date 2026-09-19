"""Command line entry point: single queries and batch benchmarks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from cacheflow.cache import DEFAULT_THRESHOLD, LexicalEmbedder, SemanticCache, build_embedder
from cacheflow.client import LLMClient, PRICING_USD_PER_1K
from cacheflow.models import ProxyResponse, QueryRequest, RouteTier
from cacheflow.proxy import CacheFlowEngine

DEFAULT_FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "queries.json"


def load_queries(path: Path) -> list[QueryRequest]:
    """Accepts a JSON list of strings, a list of objects, or ``{"queries": [...]}``."""
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("queries", [])
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a list of queries")
    requests: list[QueryRequest] = []
    for item in raw:
        if isinstance(item, str):
            requests.append(QueryRequest(query=item))
        elif isinstance(item, dict):
            requests.append(QueryRequest.model_validate(item))
        else:
            raise ValueError(f"{path}: unsupported query entry {item!r}")
    return requests


def build_engine(args: argparse.Namespace) -> CacheFlowEngine:
    embedder = LexicalEmbedder() if args.offline else build_embedder()
    return CacheFlowEngine(
        cache=SemanticCache(
            embedder=embedder,
            threshold=args.threshold,
            ttl_seconds=args.ttl,
            max_entries=args.max_entries,
        ),
        client=LLMClient(
            api_key="" if args.offline else None,
            simulate_latency=not args.no_simulate_latency,
        ),
    )


def _print_banner(engine: CacheFlowEngine) -> None:
    mode = "live (Gemini)" if engine.client.live else "offline (deterministic mock)"
    print(
        f"cache-flow | mode={mode} | embedder={engine.cache.embedder.name} "
        f"| threshold={engine.cache.threshold}",
        file=sys.stderr,
    )


def cmd_query(args: argparse.Namespace) -> int:
    engine = build_engine(args)
    _print_banner(engine)
    result = engine.handle(
        QueryRequest(
            query=args.text, namespace=args.namespace, force_refresh=args.force_refresh
        )
    )
    if args.json:
        print(result.model_dump_json(indent=2))
    else:
        _print_result(result)
    return 0


def _print_result(result: ProxyResponse) -> None:
    label = "HIT " if result.cached else "MISS"
    sim = f"{result.similarity_score:.4f}" if result.similarity_score is not None else "-"
    print(f"[{label}] model={result.model_selected} similarity={sim} "
          f"latency={result.latency_ms:.2f}ms saved=${result.estimated_savings_usd:.6f}")
    print(f"       why: {result.rationale}")
    print(f"\n{result.response}")


def cmd_benchmark(args: argparse.Namespace) -> int:
    path = Path(args.queries)
    if not path.exists():
        print(f"error: fixture not found: {path}", file=sys.stderr)
        return 2
    requests = load_queries(path)
    engine = build_engine(args)
    _print_banner(engine)

    rows: list[tuple[int, ProxyResponse, str]] = []
    for index, request in enumerate(requests, start=1):
        result = engine.handle(request)
        rows.append((index, result, request.query))

    width = min(max((len(q) for _, _, q in rows), default=10), 52)
    header = (
        f"{'#':>3}  {'RESULT':6}  {'SIM':>6}  {'LATENCY':>9}  "
        f"{'MODEL':22}  {'QUERY':{width}}"
    )
    print(header)
    print("-" * len(header))
    for index, result, query in rows:
        sim = f"{result.similarity_score:.4f}" if result.similarity_score else "  --  "
        truncated = query if len(query) <= width else query[: width - 1] + "…"
        print(
            f"{index:>3}  {'HIT' if result.cached else 'MISS':6}  {sim:>6}  "
            f"{result.latency_ms:>7.2f}ms  {result.model_selected:22}  {truncated:{width}}"
        )

    metrics = engine.metrics()
    heavy_rate = PRICING_USD_PER_1K[RouteTier.TIER_HEAVY]
    print("\n" + "=" * len(header))
    print(f"requests           : {metrics.total_requests}")
    print(f"cache hits / misses: {metrics.cache_hits} / {metrics.cache_misses}")
    print(f"hit rate           : {metrics.hit_rate * 100:.1f}%")
    print(f"routed cheap/heavy : {metrics.routed_cheap} / {metrics.routed_heavy}")
    print(f"avg hit latency    : {metrics.avg_hit_latency_ms:.2f} ms")
    print(f"avg miss latency   : {metrics.avg_miss_latency_ms:.2f} ms")
    if metrics.avg_miss_latency_ms > 0 and metrics.cache_hits:
        reduction = 1.0 - (metrics.avg_hit_latency_ms / metrics.avg_miss_latency_ms)
        note = "" if reduction > 0 else "  (misses are not being slowed by a real model)"
        print(f"latency reduction  : {reduction * 100:.1f}% on hits{note}")
    print(f"latency saved      : {metrics.total_latency_saved_ms:.1f} ms")
    print(
        f"cost saved         : ${metrics.total_cost_saved_usd:.6f} "
        f"(heavy tier @ ${heavy_rate}/1k tokens)"
    )
    print(f"live entries       : {metrics.entries}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cacheflow", description="Semantic cache + cost-aware model router"
    )
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="cosine similarity required for a cache hit")
    parser.add_argument("--ttl", type=float, default=None,
                        help="cache entry time-to-live in seconds")
    parser.add_argument("--max-entries", type=int, default=1024,
                        help="LRU capacity")
    parser.add_argument("--offline", action="store_true",
                        help="force the mock embedder/client even if GEMINI_API_KEY is set")
    parser.add_argument("--no-simulate-latency", action="store_true",
                        help="skip the mock's simulated model latency")
    sub = parser.add_subparsers(dest="command", required=True)

    q = sub.add_parser("query", help="run a single query through the proxy")
    q.add_argument("text")
    q.add_argument("--namespace", default="default")
    q.add_argument("--force-refresh", action="store_true")
    q.add_argument("--json", action="store_true", help="emit the raw ProxyResponse")
    q.set_defaults(func=cmd_query)

    b = sub.add_parser("benchmark", help="replay a batch of queries and report savings")
    b.add_argument("--queries", default=str(DEFAULT_FIXTURE))
    b.set_defaults(func=cmd_benchmark)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    exit_code: int = args.func(args)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
