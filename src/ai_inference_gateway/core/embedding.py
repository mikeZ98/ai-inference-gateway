"""Embedding clients producing fixed-dimension dense vectors."""

import hashlib
import math
import re
from collections.abc import Iterator
from itertools import pairwise
from typing import Annotated, Final, Protocol, TypedDict

import httpx
from pydantic import Field, FiniteFloat, SecretStr, TypeAdapter, ValidationError

from ai_inference_gateway.core.errors import EmbeddingError
from ai_inference_gateway.schemas import Embedding

DEFAULT_DIMENSION: Final = 384

_WORD_RE: Final = re.compile(r"\w+")
_SIGN_BIT: Final = 1 << 63


class EmbeddingClient(Protocol):
    """Turns text into a dense vector of exactly ``dimension`` floats."""

    @property
    def dimension(self) -> int:
        """Vector length produced by :meth:`embed`."""
        ...

    async def embed(self, text: str) -> Embedding:
        """Embed ``text``. Raises ``EmbeddingError``."""
        ...


class HashingEmbedder:
    """Deterministic, offline embedder using signed feature hashing.

    Features are word unigrams, word bigrams and character trigrams of the
    case/whitespace-normalized text, hashed with BLAKE2b into ``dimension``
    buckets and L2-normalized. Near-duplicate prompts get cosine similarity
    close to 1; no network, model download or randomness is involved.
    """

    def __init__(self, dimension: int = DEFAULT_DIMENSION) -> None:
        """Create an embedder producing ``dimension``-length vectors."""
        if dimension < 1:
            raise ValueError("dimension must be positive")
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        """Vector length produced by :meth:`embed`."""
        return self._dimension

    async def embed(self, text: str) -> Embedding:
        """Embed ``text`` locally."""
        normalized = " ".join(text.lower().split())
        if not normalized:
            raise EmbeddingError("cannot embed blank text")
        vector = [0.0] * self._dimension
        for feature in _features(normalized):
            digest = int.from_bytes(
                hashlib.blake2b(feature.encode(), digest_size=8).digest(), "big"
            )
            vector[digest % self._dimension] += 1.0 if digest & _SIGN_BIT else -1.0
        norm = math.sqrt(math.fsum(v * v for v in vector))
        if norm == 0.0:
            # Every feature cancelled out; fall back to a fixed unit vector.
            return (1.0,) + (0.0,) * (self._dimension - 1)
        return tuple(v / norm for v in vector)


def _features(normalized: str) -> Iterator[str]:
    words = _WORD_RE.findall(normalized)
    yield from (f"w:{word}" for word in words)
    yield from (f"b:{left} {right}" for left, right in pairwise(words))
    padded = f" {normalized} "
    yield from (f"c:{padded[i : i + 3]}" for i in range(len(padded) - 2))


class _EmbeddingItem(TypedDict):
    embedding: list[FiniteFloat]


class _EmbeddingResponse(TypedDict):
    data: Annotated[list[_EmbeddingItem], Field(min_length=1)]


_RESPONSE_ADAPTER: Final = TypeAdapter(_EmbeddingResponse)


class OpenAICompatibleEmbedder:
    """Production adapter for any OpenAI-compatible ``POST /embeddings`` endpoint.

    Works with OpenAI, Mistral (``mistral-embed``), Text Embeddings Inference and
    similar servers. The shared ``httpx.AsyncClient`` is injected so connection
    pooling is owned by the application and tests can mock transport offline.
    """

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        base_url: str,
        api_key: SecretStr | None,
        model: str,
        dimension: int,
        timeout_s: float = 10.0,
    ) -> None:
        """Configure the endpoint, credentials and expected vector size."""
        if dimension < 1:
            raise ValueError("dimension must be positive")
        self._client = client
        self._url = f"{base_url.rstrip('/')}/embeddings"
        self._api_key = api_key
        self._model = model
        self._dimension = dimension
        self._timeout_s = timeout_s

    @property
    def dimension(self) -> int:
        """Vector length produced by :meth:`embed`."""
        return self._dimension

    def _headers(self) -> dict[str, str]:
        if self._api_key is None:
            return {}
        return {"Authorization": f"Bearer {self._api_key.get_secret_value()}"}

    async def embed(self, text: str) -> Embedding:
        """Embed ``text`` via the remote endpoint."""
        # Transport errors are raised 'from None': httpx exceptions carry the
        # request (incl. auth headers), which must never reach tracebacks/trackers.
        try:
            response = await self._client.post(
                self._url,
                json={"model": self._model, "input": text},
                headers=self._headers(),
                timeout=self._timeout_s,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise EmbeddingError(
                f"embedding endpoint returned HTTP {exc.response.status_code}"
            ) from None
        except httpx.HTTPError as exc:
            raise EmbeddingError(f"embedding request failed: {type(exc).__name__}") from None

        try:
            payload = _RESPONSE_ADAPTER.validate_json(response.content, strict=True)
        except ValidationError as exc:
            raise EmbeddingError("embedding response has an unexpected shape") from exc

        vector = tuple(payload["data"][0]["embedding"])
        if len(vector) != self._dimension:
            raise EmbeddingError(
                f"embedding has dimension {len(vector)}, expected {self._dimension}"
            )
        return vector
