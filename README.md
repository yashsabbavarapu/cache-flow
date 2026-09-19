# cache-flow

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
hits — so you can measure cache behaviour from the edge without parsing bodies.

`QueryRequest` supports `namespace` (tenant isolation — namespaces never share
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
signals — synthesis keywords (`implement`, `refactor`, `debug`, `algorithm`),
multi-step cues (`step by step`, `compare and contrast`, `trade-off`),
structural constraints (JSON/schema/table/code fences), clause count, and length.
Score ≥ 1.0 escalates to the heavy tier, and every decision returns the
rationale that produced it:

```
cache miss -> tier_heavy (score 1.25): synthesis keyword(s): algorithm, refactor
```

One non-obvious rule: **brevity only counts as evidence of simplicity when
nothing else fires.** `"Refactor the algorithm"` is four words and still needs
the heavy tier, so the short-prompt discount is applied last and only to a
query that scored zero.

## Embeddings: read this before trusting the mock

Two backends, and the difference matters:

| Backend | When | What similarity means |
| --- | --- | --- |
| `LexicalEmbedder` | default, offline | **Lexical overlap** — hashed word unigrams + character trigrams |
| `GeminiEmbedder` | `GEMINI_API_KEY` set | **Semantic** — `text-embedding-004` |

The mock is deterministic, dependency-free and instant, which makes it right for
tests and CI. It is **not** a semantic model. Measured on the lexical backend:

| Query pair | Cosine | At 0.94 |
| --- | --- | --- |
| `How do I reset my password?` / `How can I reset my password?` | 1.000 | HIT |
| `What's the refund window...` / `What is the refund window...` | 1.000 | HIT |
| `How do I reset my password?` / `Why was my password reset?` | 0.789 | miss |
| `Do reset my password` / `Don't reset my password` | 0.889 | miss |
| `What is your refund policy?` / `How do I get my money back?` | 0.000 | **miss** |

That last row is the honest limitation: a paraphrase that swaps the entire
vocabulary carries no lexical signal, so the offline backend cannot match it.
**Vocabulary-swap paraphrases require the Gemini backend.** The rows above it
are the payoff — punctuation, casing, articles, auxiliaries and contractions all
collapse to one entry, and that is the bulk of real duplicate traffic.

Two deliberate choices in the mock's normalization:

- **Interrogatives are kept** (`how`, `what`, `why`, `when`). `"How do I reset
  my password"` and `"Why was my password reset"` are different information
  needs and must not share an answer.
- **Negations are expanded, not clipped** (`don't` → `do not`). Serving the
  affirmative answer to a negated question is the worst failure a cache can
  have; there is a test pinning it.

## The 0.94 threshold trade-off

The threshold is the entire safety/savings dial.

| Threshold | Effect |
| --- | --- |
| `> 0.97` | Near-exact only. Safe, low hit rate — barely beats Redis. |
| **`0.94`** | Default. Wording, punctuation and function-word variance collapse; distinct intents stay separate. |
| `0.90–0.93` | Higher hit rate, and the first wrong answers appear — `0.889` for an inverted negation is uncomfortably close. |
| `< 0.85` | Actively unsafe. Same-topic/different-intent queries collide. |

Two caveats worth knowing before you tune it:

1. **The right number depends on the embedder.** Cosine scores are not
   comparable across models. `text-embedding-004` puts unrelated English text
   around 0.5–0.7 rather than near 0, so 0.94 is a *strict* gate there. If you
   swap embedders, re-measure — do not port the constant.
2. **Comparison is inclusive within `SCORE_EPSILON` (1e-6).** Vectors are stored
   as float32, so a score that is mathematically exactly 0.94 can land ~1e-8
   below it. A two-decimal heuristic should not be decided by float
   representation noise.

Raise the gate per-process with `--threshold`, or `SemanticCache(threshold=...)`.

## Benchmarks

Measured on the bundled 15-query fixture (Apple Silicon, Python 3.11, offline
backend, mock latency of 60 ms cheap / 240 ms heavy standing in for real calls):

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

Retrieval is a linear cosine scan over the namespace. That is deliberate — it
has no index to corrupt and no rebuild to schedule — but it does bound how far
one process scales:

| Entries in namespace | p50 | p95 | max |
| --- | --- | --- | --- |
| 100 | 0.15 ms | 0.17 ms | 0.81 ms |
| 1 000 | 0.62 ms | 0.75 ms | 1.79 ms |
| 5 000 | 3.11 ms | 3.72 ms | 7.42 ms |
| 10 000 | 7.67 ms | 9.42 ms | **21.24 ms** |

**The sub-15 ms budget holds to roughly 5 000 entries per namespace.** At 10 000
the tail breaches it. The default `max_entries` is 1 024, comfortably inside the
flat region; past ~5 000 you want an ANN index (hnswlib/FAISS) rather than a
larger scan. Savings figures use mock token counts — the shape is real, the
absolute dollars are only as good as the `$0.005/1k` heavy-tier rate in
`client.py`.

## Eviction and invalidation

- **TTL** — `SemanticCache(ttl_seconds=...)` or `--ttl`. Expired entries are
  purged lazily on the next lookup.
- **LRU** — `max_entries` caps the store; reads refresh recency, so a hot entry
  survives a flood of one-off queries.
- **Manual** — `evict(key)` for one entry, `clear(namespace)` for a tenant,
  `clear(None)` or `POST /cache/purge {}` for everything.

## Testing

```bash
pytest -v            # 65 tests
mypy --strict cacheflow tests
```

The suite is fully offline and deterministic — no network, no API key, no
sleeps in the default path. Threshold-boundary tests use a `StubEmbedder` that
places vectors at exact angles, so `0.93` misses and `0.95` hits by
construction rather than by hoping a real embedder lands there.

## Known limitations

- Vocabulary-swap paraphrases need the Gemini backend (see above).
- The store is per-process and in-memory: it does not survive a restart and is
  not shared across workers. Multi-replica deployments need a shared vector
  store.
- Linear scan caps a namespace at ~5 000 entries within the latency budget.
- Cost figures are estimates from a static price table, not provider billing.
