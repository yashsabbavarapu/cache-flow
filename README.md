# cache-flow

[![CI](https://github.com/yashsabbavarapu/cache-flow/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/yashsabbavarapu/cache-flow/actions/workflows/ci.yml?query=branch%3Amain)

A semantic vector cache and cost-aware model router for LLM traffic. It sits in
front of your model provider, serves semantically equivalent prompts from a
local vector store in well under a millisecond, and sends genuine cache misses
to the cheapest tier that can actually answer them.

Exact-match caches (Redis, memcached) miss almost everything in production
because users rephrase: `"What's the refund window?"` and
`"What is the refund window?"` are one cached answer, not two model calls.

```
                    ┌──────────────────────────────────────────────┐
  POST /v1/chat ──► │ 1. embed query                               │
                    │ 2. cosine scan namespace ── ≥ 0.94 ──► HIT ──┼──► cached answer
                    │                                 │            │    ~0.4 ms, $0.00
                    │                                MISS          │
                    │ 3. complexity router ──┬─ cheap ─► Flash ────┼──► answer + cache write
                    │                        └─ heavy ─► Pro ──────┤
                    └──────────────────────────────────────────────┘
```

## Quick start

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Everything runs offline by default with a deterministic mock embedder and mock
model. Set `GEMINI_API_KEY` to switch both to live Gemini calls.

```bash
# Batch benchmark over a fixture trace
python -m cacheflow.cli benchmark --queries fixtures/queries.json

# One query
python -m cacheflow.cli query "How do I reset my password?"

# Serve the proxy
uvicorn cacheflow.proxy:app --port 8000
```

```bash
curl -X POST http://127.0.0.1:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the refund window for online orders?"}'
```

## HTTP API

| Endpoint | Purpose |
| --- | --- |
| `POST /v1/chat` | `QueryRequest` in, `ProxyResponse` out. Cache gate, then router. |
| `GET /metrics` | Hit rate, latency saved, estimated USD saved, tier split. |
| `POST /cache/purge` | `{"namespace": "support"}` purges one; `{}` purges everything. |
| `GET /health` | Active embedder, threshold, live-vs-mock mode, entry count. |

Every `/v1/chat` response carries `X-CacheFlow-Cache: HIT|MISS`,
`X-CacheFlow-Model`, `X-CacheFlow-Latency-Ms`, and `X-CacheFlow-Similarity` on
hits, so you can measure cache behaviour from the edge without parsing bodies.

`QueryRequest` supports `namespace` (tenant isolation. Namespaces never share
entries), `max_tokens`, and `force_refresh` to bypass the gate and re-prime.

## Architecture

| Module | Responsibility |
| --- | --- |
| `models.py` | Pydantic v2 contracts shared by every layer |
| `cache.py` | Embedders, cosine retrieval, TTL, LRU eviction |
| `router.py` | Transparent complexity heuristics, cheap vs heavy |
| `client.py` | Gemini REST calls, deterministic offline fallback |
| `proxy.py` | `CacheFlowEngine` (cache → router → client) + FastAPI app |
| `cli.py` | Single queries and batch benchmarks |

`CacheFlowEngine` holds the entire decision path, so the CLI and the HTTP app
exercise one implementation rather than two that drift.

### Why the router is rules, not a model

Routing runs on every cache miss. A classifier call would add the latency and
cost the router exists to remove, so scoring is a handful of transparent
signals. Synthesis keywords (`implement`, `refactor`, `debug`, `algorithm`),
multi-step cues (`step by step`, `compare and contrast`, `trade-off`),
structural constraints (JSON/schema/table/code fences), clause count, and length.
Score ≥ 1.0 escalates to the heavy tier, and every decision returns the
rationale that produced it:

```
cache miss -> tier_heavy (score 1.25): synthesis keyword(s): algorithm, refactor
```

One non-obvious rule: brevity only counts as evidence of simplicity when
nothing else fires. `"Refactor the algorithm"` is four words and still needs
the heavy tier, so the short-prompt discount is applied last and only to a
query that scored zero.

## Embeddings: two backends, and the difference is the product

| Backend | When | What similarity means | Hit latency |
| --- | --- | --- | --- |
| `LexicalEmbedder` | default, offline | **Lexical overlap**, hashed word unigrams + char trigrams | ~0.3 ms |
| `GeminiEmbedder` | `GEMINI_API_KEY` set | **Semantic**, `gemini-embedding-001` | ~310 ms / ~0.1 ms memoized |

The offline backend is deterministic, dependency-free and instant, which makes
it right for tests and CI. It is **not** a semantic model. It collapses
punctuation, casing, articles, auxiliaries and contractions to one entry (the
bulk of real duplicate traffic), but a paraphrase that swaps the whole
vocabulary carries no lexical signal:

| Query pair | Lexical | Gemini |
| --- | --- | --- |
| `How do I reset my password?` / `How can I reset my password?` | 1.000 HIT | 0.998 HIT |
| `What's the refund window...` / `What is the refund window...` | 1.000 HIT | 0.998 HIT |
| `What is your refund policy?` / `How do I get my money back?` | 0.000 miss | 0.871 miss |
| `How do I reset my password?` / `Why was my password reset?` | 0.789 miss | 0.848 miss |
| `Do reset my password` / `Don't reset my password` | 0.889 miss | 0.876 miss |

Two deliberate choices in the offline normalization: **interrogatives are kept**
(`"How do I reset..."` and `"Why was my password reset..."` are different
information needs) and negations are expanded, not clipped (`don't` → `do
not`). Serving the affirmative answer to a negated question is the worst failure
a cache can have; there is a test pinning both.

### `taskType` is not optional

`gemini-embedding-001` must be called with `taskType=SEMANTIC_SIMILARITY`. Both
sides of a cache comparison are user questions, so the comparison is symmetric, not the asymmetric `RETRIEVAL_QUERY`/`RETRIEVAL_DOCUMENT` pair. Measured over 20
labelled pairs, the gap between the *lowest* paraphrase and the *highest*
distinct query:

| Configuration | Paraphrase band | Distinct band | Separation |
| --- | --- | --- | --- |
| 768 dims, no `taskType` | 0.666 – 0.842 | 0.389 – 0.764 | **−0.098** |
| 3072 dims, no `taskType` | 0.680 – 0.847 | 0.435 – 0.770 | −0.090 |
| 768 dims, `SEMANTIC_SIMILARITY` | 0.871 – 0.998 | 0.667 – 0.935 | −0.005 |
| 3072 dims, `SEMANTIC_SIMILARITY` | 0.878 – 0.998 | 0.695 – 0.938 | +0.002 |

Without it the two bands **overlap**, and no threshold separates them at any
value. 768 dimensions scores identically to 3072 while being 4× cheaper to scan,
so 768 is the default.

## Calibrating the 0.94 threshold against live embeddings

Measured on 20 labelled pairs through the shipped configuration
(`gemini-embedding-001`, 768 dims, `SEMANTIC_SIMILARITY`):

At 0.94. 18/20 correct, 0 false hits, 2 false misses.

```
  0.9977  HIT   para   How do I reset my password?        || How can I reset my password?
  0.9959  HIT   para   How much does the pro plan cost?   || What is the price of the pro plan?
  0.9839  HIT   para   Is the API down right now?         || Is there an API outage at the moment?
  0.9532  HIT   para   What are your support hours?       || When is support available?
──────────────────────────────── 0.94 gate ────────────────────────────────
  0.9346  miss  dist   How much does the pro plan cost?   || How much does the enterprise plan cost?
  0.9334  miss  dist   How do I export my data?           || How do I import my data?
  0.9192  miss  dist   refund window for online orders    || refund window for in-store orders
  0.9165  miss  dist   download the desktop app           || download the mobile app
  0.9164  miss  PARA   return policy for customers        || how many days to return items      <- false miss
  0.8708  miss  PARA   What is your refund policy?        || How do I get my money back?        <- false miss
  0.8478  miss  dist   How do I reset my password?        || Why was my password reset?
  0.6672  miss  dist   How do I reset my password?        || Write a Rust web server
```

Three things this says, none of them obvious in advance:

1. 0.94 is a good number, but only by ~0.005. The highest *distinct* pair
   sits at 0.9346. Those near-misses are all **entity swaps**. Pro/enterprise,
   export/import, online/in-store, desktop/mobile, add/remove, and they cluster
   in a wall at 0.87–0.935, directly beneath the gate. Dropping the threshold to
   0.90 to recover the two false misses would admit *all five*, i.e. quote the
   enterprise price to someone asking about pro. Do not lower this below 0.94.
2. Both errors are false misses, which is the safe direction. A false miss
   costs one model call; a false hit returns a confidently wrong answer.
3. A deep paraphrase can still miss. `"What is your refund policy?"` vs
   `"How do I get my money back?"` scores 0.871. Correct in meaning, but below
   any threshold that is safe against entity swaps. This is a real ceiling, not
   a tuning failure.

Cosine scores are **not** comparable across embedders. If you swap models,
re-run this calibration. Do not port the constant. (Comparison is inclusive
within `SCORE_EPSILON` = 1e-6: vectors are float32, and a two-decimal heuristic
should not be decided by representation noise.)

## Benchmarks

### Latency: where the sub-15 ms claim holds, and where it does not

This is the single most important operational fact about the system.

| Path | Hit latency | Why |
| --- | --- | --- |
| Offline embedder | **0.3 ms** | pure local hashing |
| Live, repeated query string | **0.14 ms** | embedding memo, no network |
| Live, novel paraphrase | **~310 ms** | one `embedContent` round trip |
| Cache miss (full generation) | ~1000–1200 ms | model call |

The sub-15 ms cache hit is real for the local embedder and for repeated query
strings. It is not achievable for a novel paraphrase against a remote embedding
API. The gate must embed the incoming query before it can compare anything, so
a first-seen string pays one network round trip no matter how fast the vector
search is. At ~310 ms it still beats a ~1100 ms generation by roughly 3.5×, and
it still costs $0.00, but it is 300 ms, not 10 ms.

`MemoizingEmbedder` wraps the remote backend to collapse byte-identical repeats
to a dict lookup (307 ms → 0.14 ms measured). If you need sub-15 ms on novel
paraphrases, the embedding model has to be local.

### Throughput (offline backend, bundled 15-query fixture)

Mock latency of 60 ms cheap / 240 ms heavy stands in for real model calls:

```
requests           : 15
cache hits / misses: 8 / 7
hit rate           : 53.3%
routed cheap/heavy : 5 / 2
avg hit latency    : 0.38 ms
avg miss latency   : 117.15 ms
latency reduction  : 99.7% on hits
latency saved      : 694.1 ms
cost saved         : $0.000316 (heavy tier @ $0.005/1k tokens)
```

Live equivalent over a 5-query trace: 40% hit rate, avg hit 380 ms vs avg miss
985 ms. A **61% latency reduction**, the honest figure once a real embedding
round trip is in the path.

### Scan ceiling

Retrieval is a linear cosine scan over the namespace (no index to corrupt, no
rebuild to schedule) but it bounds how far one process scales:

| Entries in namespace | p50 | p95 | max |
| --- | --- | --- | --- |
| 100 | 0.15 ms | 0.17 ms | 0.81 ms |
| 1 000 | 0.62 ms | 0.75 ms | 1.79 ms |
| 5 000 | 3.11 ms | 3.72 ms | 7.42 ms |
| 10 000 | 7.67 ms | 9.42 ms | **21.24 ms** |

The local-path budget holds to roughly 5 000 entries per namespace; at 10 000
the tail breaches it. Default `max_entries` is 1 024, comfortably inside the flat
region. Past ~5 000, use an ANN index (hnswlib/FAISS) rather than a longer scan.

Savings figures use estimated token counts and a static price table. The shape
is real, the absolute dollars are only as good as the `$0.005/1k` heavy-tier rate
in `client.py`.

## Eviction and invalidation

- **TTL**. `SemanticCache(ttl_seconds=...)` or `--ttl`. Expired entries are
  purged lazily on the next lookup.
- **LRU**. `max_entries` caps the store; reads refresh recency, so a hot entry
  survives a flood of one-off queries.
- **Manual**. `evict(key)` for one entry, `clear(namespace)` for a tenant,
  `clear(None)` or `POST /cache/purge {}` for everything.

## Testing

```bash
pytest -v            # 76 tests
mypy --strict cacheflow tests
```

The suite is hermetic: it passes identically with `GEMINI_API_KEY` exported and
unset (verified both ways, 0.17 s each), makes no network calls, and does not
sleep in the default path. That property is itself tested. A falsy-empty-cache
bug once let an exported key reach the engine through a *discarded* injection,
and only an environment-sensitive test run exposed it. Threshold-boundary tests use a `StubEmbedder` that
places vectors at exact angles, so `0.93` misses and `0.95` hits by
construction rather than by hoping a real embedder lands there.

## Known limitations

- Novel paraphrases cost one embedding round trip (~310 ms). Sub-15 ms hits
  require a local embedding model or a repeated query string.
- Deep vocabulary-swap paraphrases can still miss (0.871 for *refund policy*
  / *money back*) because entity-swap distractors sit at 0.87–0.935 and a safe
  threshold has to clear them.
- The store is per-process and in-memory: it does not survive a restart and is
  not shared across workers. Multi-replica deployments need a shared vector store.
- Linear scan caps a namespace at ~5 000 entries within the local latency budget.
- Cost figures are estimates from a static price table, not provider billing.
- **Model ids drift.** `text-embedding-004`, `gemini-2.0-flash`, `gemini-2.5-flash`
  and `gemini-2.5-pro` all 404 for new keys. Worse, the `/models` list advertises
  models that are not callable. `gemini-2.5-flash` is listed and still 404s. The
  defaults here were verified by calling the endpoints, not by reading the list.
  Re-verify before deploying; the client falls back to a deterministic mock on
  any provider failure (verified against a live 429) rather than dropping traffic.
