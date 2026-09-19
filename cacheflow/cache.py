"""Semantic vector cache: cosine retrieval, TTL expiry and LRU eviction.

Two embedding backends are provided:

* :class:`LexicalEmbedder` -- deterministic, offline, zero-dependency. It hashes
  word unigrams and character trigrams into a fixed-width vector, so cosine
  similarity measures *lexical* overlap (wording, morphology, word order), not
  meaning. Good enough for tests, benchmarks and near-duplicate traffic.
* :class:`GeminiEmbedder` -- Google ``text-embedding-004`` over REST. Required
  for true semantic paraphrase matching ("refund policy" ~ "get my money back").
"""

from __future__ import annotations

import hashlib
import os
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from cacheflow.models import CacheEntry

Vector = npt.NDArray[np.float32]

DEFAULT_THRESHOLD = 0.94
DEFAULT_DIMENSIONS = 512
# Vectors are stored as float32 (what embedding providers emit), so a score that
# is mathematically exactly at the threshold can land ~1e-8 under it. The
# threshold is a two-decimal heuristic; float noise must not decide a hit.
SCORE_EPSILON = 1e-6
_TOKEN_RE = re.compile(r"[a-z0-9']+")
# Standard contraction expansion. Without it "what's" and "what is" hash to
# disjoint features and a pure-punctuation paraphrase misses the cache; negations
# are expanded rather than clipped so "don't" never folds into "do".
_CONTRACTIONS = {
    "n't": " not", "'re": " are", "'ve": " have", "'ll": " will",
    "'m": " am", "'d": " would", "what's": "what is", "who's": "who is",
    "how's": "how is", "where's": "where is", "that's": "that is",
    "there's": "there is", "it's": "it is", "let's": "let us",
    "can't": "can not", "won't": "will not",
}
# Function words dropped before hashing: articles, auxiliaries, modals and
# pronouns. Interrogatives (how/what/why/when/where/who) are deliberately KEPT --
# they change the information need, so they must change the cache key's vector.
_STOPWORDS = frozenset(
    "a an the is are am was were be been being do does did done have has had"
    " can could will would shall should may might must of for to in on at by"
    " with from as it its i you me my your we us our they them their and or if"
    " that this these those there here please just"
    .split()
)


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns text into a unit-norm vector."""

    name: str

    def embed(self, text: str) -> list[float]: ...


def _normalize(values: list[float]) -> list[float]:
    vec = np.asarray(values, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return [float(v) for v in vec]
    return [float(v) for v in vec / norm]


def dig(node: object, *path: str | int) -> object:
    """Walk a decoded JSON payload, raising instead of returning ``Any``.

    Shared by both REST integrations so provider responses are validated in one
    place rather than with a ladder of ``isinstance`` checks at each call site.
    """
    for key in path:
        if isinstance(key, int):
            if not isinstance(node, list) or len(node) <= key:
                raise ValueError(f"expected a list of >{key} items at {key!r}")
            node = node[key]
        else:
            if not isinstance(node, dict) or key not in node:
                raise ValueError(f"missing {key!r} in provider response")
            node = node[key]
    return node


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Cosine similarity of two equal-length vectors, clamped to [-1, 1]."""
    a = np.asarray(left, dtype=np.float32)
    b = np.asarray(right, dtype=np.float32)
    denom = float(np.linalg.norm(a)) * float(np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.clip(float(np.dot(a, b)) / denom, -1.0, 1.0))


class LexicalEmbedder:
    """Hashing bag-of-features embedder. Deterministic and offline.

    Similarity here is lexical, not semantic: paraphrases that reuse words match
    strongly, paraphrases that swap vocabulary do not. Use Gemini for the latter.
    """

    def __init__(self, dimensions: int = DEFAULT_DIMENSIONS) -> None:
        self.dimensions = dimensions
        self.name = f"lexical-hash-{dimensions}"

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        lowered = text.lower()
        for contraction, expansion in _CONTRACTIONS.items():
            lowered = lowered.replace(contraction, expansion)
        # Drop possessive clitics ("user's guide" -> "user guide").
        lowered = lowered.replace("'s ", " ").replace("'", "")
        words = [w for w in _TOKEN_RE.findall(lowered) if w not in _STOPWORDS]
        # Cheap plural/inflection folding so "items"/"item" collide.
        return [w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words]

    def _features(self, text: str) -> list[str]:
        words = self._tokenize(text)
        feats: list[str] = [f"w:{w}" for w in words]
        for word in words:
            padded = f"^{word}$"
            feats.extend(f"c:{padded[i:i + 3]}" for i in range(len(padded) - 2))
        return feats

    def _bucket(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "big") % self.dimensions
        sign = 1.0 if digest[4] & 1 else -1.0
        return index, sign

    def embed(self, text: str) -> list[float]:
        vec = np.zeros(self.dimensions, dtype=np.float32)
        counts: dict[str, int] = {}
        for feature in self._features(text):
            counts[feature] = counts.get(feature, 0) + 1
        for feature, count in counts.items():
            index, sign = self._bucket(feature)
            vec[index] += sign * (1.0 + np.log(count))
        return _normalize(vec.tolist())


#: ``text-embedding-004`` is retired on v1beta and returns 404; this is its
#: replacement. Verified against the live model list rather than assumed.
GEMINI_EMBED_MODEL = "gemini-embedding-001"

#: Both sides of a cache comparison are user questions, so similarity is
#: symmetric -- SEMANTIC_SIMILARITY, not the asymmetric RETRIEVAL_QUERY /
#: RETRIEVAL_DOCUMENT pair. This is not a tuning knob: measured on 20 labelled
#: pairs, omitting it collapses the gap between paraphrases and distinct queries
#: from +0.00 to -0.10, i.e. the two bands overlap and NO threshold separates
#: them. See README "Calibrating against live embeddings".
GEMINI_EMBED_TASK_TYPE = "SEMANTIC_SIMILARITY"

#: The model emits 3072 dimensions by default. Retrieval is a linear scan, so
#: width is a direct latency cost; 768 is the documented quality/size sweet spot
#: and truncated outputs are re-normalized below, as Google requires.
GEMINI_EMBED_DIMENSIONS = 768


class GeminiEmbedder:
    """Google ``gemini-embedding-001`` via the Generative Language REST API."""

    ENDPOINT = (
        "https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent"
    )

    def __init__(
        self,
        api_key: str,
        model: str = GEMINI_EMBED_MODEL,
        dimensions: int | None = GEMINI_EMBED_DIMENSIONS,
        task_type: str | None = GEMINI_EMBED_TASK_TYPE,
        timeout: float = 10.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions
        self.task_type = task_type
        self.timeout = timeout
        self.name = f"gemini-{model}" if not model.startswith("gemini") else model

    def embed(self, text: str) -> list[float]:
        import httpx

        payload: dict[str, object] = {
            "model": f"models/{self.model}",
            "content": {"parts": [{"text": text}]},
        }
        if self.dimensions is not None:
            payload["outputDimensionality"] = self.dimensions
        if self.task_type is not None:
            payload["taskType"] = self.task_type
        response = httpx.post(
            self.ENDPOINT.format(model=self.model),
            json=payload,
            headers={"x-goog-api-key": self.api_key},
            timeout=self.timeout,
        )
        response.raise_for_status()
        values = dig(response.json(), "embedding", "values")
        if not isinstance(values, list) or not values:
            raise ValueError("embedding response carried no values")
        return _normalize([float(v) for v in values])


class MemoizingEmbedder:
    """Bounded exact-string memo in front of a remote embedder.

    A remote embedding call is ~350-400 ms, and the cache gate pays it on every
    lookup -- so a repeated query string would otherwise cost a network round
    trip to discover it is already cached. This collapses that to a dict hit.
    It only helps byte-identical repeats; a novel paraphrase still pays the call.
    """

    def __init__(self, inner: Embedder, max_entries: int = 4096) -> None:
        self.inner = inner
        self.max_entries = max_entries
        self.name = inner.name
        self._memo: OrderedDict[str, list[float]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def embed(self, text: str) -> list[float]:
        cached = self._memo.get(text)
        if cached is not None:
            self._memo.move_to_end(text)
            self.hits += 1
            return cached
        self.misses += 1
        vector = self.inner.embed(text)
        self._memo[text] = vector
        if len(self._memo) > self.max_entries:
            self._memo.popitem(last=False)
        return vector


def build_embedder(prefer_live: bool = True) -> Embedder:
    """Gemini when ``GEMINI_API_KEY`` is set, otherwise the offline mock.

    The remote backend is memoized; the local one is already faster than a dict
    lookup would save, so it is returned bare.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if prefer_live and api_key:
        return MemoizingEmbedder(GeminiEmbedder(api_key))
    return LexicalEmbedder()


@dataclass(slots=True)
class _Record:
    key: str
    namespace: str
    entry: CacheEntry
    vector: Vector


class SemanticCache:
    """In-memory vector store with cosine retrieval, TTL and LRU eviction."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        threshold: float = DEFAULT_THRESHOLD,
        ttl_seconds: float | None = None,
        max_entries: int = 1024,
    ) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self.embedder = embedder if embedder is not None else build_embedder()
        self.threshold = threshold
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._store: OrderedDict[str, _Record] = OrderedDict()
        self.evictions = 0

    def __len__(self) -> int:
        return len(self._store)

    @staticmethod
    def make_key(namespace: str, query: str) -> str:
        digest = hashlib.sha1(f"{namespace}\x00{query}".encode("utf-8")).hexdigest()
        return f"{namespace}:{digest[:16]}"

    def embed(self, text: str) -> list[float]:
        return self.embedder.embed(text)

    def lookup(
        self,
        query: str,
        namespace: str = "default",
        embedding: list[float] | None = None,
        now: float | None = None,
    ) -> tuple[CacheEntry, float] | None:
        """Return the best match at or above threshold, or ``None``."""
        now = time.time() if now is None else now
        self._purge_expired(now)
        candidates = [r for r in self._store.values() if r.namespace == namespace]
        if not candidates:
            return None

        probe = np.asarray(
            embedding if embedding is not None else self.embed(query),
            dtype=np.float32,
        )
        norm = float(np.linalg.norm(probe))
        if norm == 0.0:
            return None
        probe = probe / norm

        matrix = np.vstack([r.vector for r in candidates])
        scores = matrix @ probe
        best_index = int(np.argmax(scores))
        best_score = float(np.clip(float(scores[best_index]), -1.0, 1.0))
        if best_score + SCORE_EPSILON < self.threshold:
            return None

        record = candidates[best_index]
        self._store.move_to_end(record.key)  # LRU: recency on read
        return record.entry, best_score

    def put(
        self,
        entry: CacheEntry,
        namespace: str = "default",
        embedding: list[float] | None = None,
    ) -> str:
        vector_values = embedding if embedding is not None else entry.embedding
        if not vector_values:
            vector_values = self.embed(entry.query)
        entry.embedding = _normalize(list(vector_values))
        key = self.make_key(namespace, entry.query)
        self._store[key] = _Record(
            key=key,
            namespace=namespace,
            entry=entry,
            vector=np.asarray(entry.embedding, dtype=np.float32),
        )
        self._store.move_to_end(key)
        self._enforce_capacity()
        return key

    def evict(self, key: str) -> bool:
        return self._store.pop(key, None) is not None

    def clear(self, namespace: str | None = None) -> int:
        """Drop one namespace, or everything when ``namespace`` is ``None``."""
        if namespace is None:
            removed = len(self._store)
            self._store.clear()
            return removed
        doomed = [k for k, r in self._store.items() if r.namespace == namespace]
        for key in doomed:
            del self._store[key]
        return len(doomed)

    def namespaces(self) -> set[str]:
        return {r.namespace for r in self._store.values()}

    def _enforce_capacity(self) -> None:
        while len(self._store) > self.max_entries:
            self._store.popitem(last=False)  # oldest touched
            self.evictions += 1

    def _purge_expired(self, now: float) -> None:
        if self.ttl_seconds is None:
            return
        stale = [
            k for k, r in self._store.items() if r.entry.is_expired(self.ttl_seconds, now)
        ]
        for key in stale:
            del self._store[key]
