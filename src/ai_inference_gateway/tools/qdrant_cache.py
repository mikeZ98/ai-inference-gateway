"""Qdrant-backed implementation of the core ``SemanticCache`` port."""

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Final, Self

from pydantic import SecretStr, ValidationError
from qdrant_client import AsyncQdrantClient, models

from ai_inference_gateway.core.errors import CacheError
from ai_inference_gateway.schemas import CacheEntry, CacheQuery

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION: Final = "semantic_cache"
_POINT_NAMESPACE: Final = uuid.UUID("6f1c3c56-6b8e-4c38-9a2e-2f3e0c7d9b41")


@asynccontextmanager
async def _translate(operation: str) -> AsyncIterator[None]:
    """Turn any client/transport/database failure into ``CacheError``."""
    try:
        yield
    except CacheError:
        raise
    except Exception as exc:
        # Adapter boundary: qdrant-client surfaces httpx, gRPC, ValueError and
        # its own exception types; callers must only ever see CacheError.
        raise CacheError(f"qdrant {operation} failed: {type(exc).__name__}") from exc


class QdrantSemanticCache:
    """Cosine-similarity cache of prompt/response pairs.

    Point IDs are derived from the prompt text, so re-storing a prompt
    overwrites its previous entry. The collection is created (or validated)
    lazily on first use with Cosine distance and the configured dimension.
    """

    def __init__(
        self,
        client: AsyncQdrantClient,
        *,
        dimension: int,
        collection: str = DEFAULT_COLLECTION,
    ) -> None:
        """Wrap an existing client; use :meth:`in_memory` / :meth:`remote` to build one."""
        if dimension < 1:
            raise ValueError("dimension must be positive")
        self._client = client
        self._dimension = dimension
        self._collection = collection
        self._ready = False
        self._init_lock = asyncio.Lock()

    @classmethod
    def in_memory(cls, *, dimension: int, collection: str = DEFAULT_COLLECTION) -> Self:
        """Process-local cache for tests and development (no network)."""
        return cls(
            AsyncQdrantClient(location=":memory:"), dimension=dimension, collection=collection
        )

    @classmethod
    def remote(
        cls,
        *,
        url: str,
        dimension: int,
        api_key: SecretStr | None = None,
        collection: str = DEFAULT_COLLECTION,
        timeout_s: int = 5,
    ) -> Self:
        """Cache backed by a Qdrant server or Qdrant Cloud cluster."""
        client = AsyncQdrantClient(
            url=url,
            api_key=api_key.get_secret_value() if api_key else None,
            timeout=timeout_s,
        )
        return cls(client, dimension=dimension, collection=collection)

    @property
    def dimension(self) -> int:
        """Vector size of the collection."""
        return self._dimension

    async def initialize(self) -> None:
        """Create the collection, or verify an existing one matches. Idempotent."""
        async with self._init_lock:
            if self._ready:
                return
            async with _translate("collection init"):
                if not await self._client.collection_exists(self._collection):
                    await self._client.create_collection(
                        collection_name=self._collection,
                        vectors_config=models.VectorParams(
                            size=self._dimension, distance=models.Distance.COSINE
                        ),
                    )
                await self._verify_collection()
            self._ready = True

    async def _verify_collection(self) -> None:
        info = await self._client.get_collection(self._collection)
        params = info.config.params.vectors
        if not isinstance(params, models.VectorParams):
            raise CacheError(f"collection {self._collection!r} must use a single unnamed vector")
        if params.size != self._dimension or params.distance != models.Distance.COSINE:
            raise CacheError(
                f"collection {self._collection!r} is {params.size}-d/{params.distance}, "
                f"expected {self._dimension}-d/Cosine"
            )

    async def lookup(self, query: CacheQuery) -> CacheEntry | None:
        """Return the nearest entry whose similarity is strictly above the threshold."""
        self._check_dimension(len(query.embedding))
        await self.initialize()
        async with _translate("query"):
            result = await self._client.query_points(
                collection_name=self._collection,
                query=list(query.embedding),
                limit=query.limit,
                # Qdrant's threshold is inclusive; is_hit() below enforces '>'.
                score_threshold=query.similarity_threshold,
                with_payload=True,
                with_vectors=True,
            )
        for point in result.points:
            score = max(-1.0, min(1.0, point.score))
            if query.is_hit(score):
                return self._to_entry(point, score)
        return None

    async def store(self, entry: CacheEntry) -> None:
        """Upsert ``entry`` keyed by its prompt."""
        self._check_dimension(len(entry.embedding))
        await self.initialize()
        payload = entry.model_dump(mode="json", exclude={"embedding", "similarity_score"})
        point = models.PointStruct(
            id=str(uuid.uuid5(_POINT_NAMESPACE, entry.prompt)),
            vector=list(entry.embedding),
            payload=payload,
        )
        async with _translate("upsert"):
            await self._client.upsert(collection_name=self._collection, points=[point])

    async def close(self) -> None:
        """Release the underlying client."""
        async with _translate("close"):
            await self._client.close()

    def _check_dimension(self, size: int) -> None:
        if size != self._dimension:
            raise CacheError(f"vector has dimension {size}, collection expects {self._dimension}")

    def _to_entry(self, point: models.ScoredPoint, score: float) -> CacheEntry:
        if not isinstance(point.vector, list):
            raise CacheError(f"point {point.id} has no dense vector")
        document = {**(point.payload or {}), "embedding": point.vector, "similarity_score": score}
        try:
            # JSON-mode validation keeps strict semantics while accepting the
            # JSON-native encodings Qdrant stores (enum values, ISO datetimes, lists).
            return CacheEntry.model_validate_json(json.dumps(document))
        except (ValidationError, TypeError, ValueError) as exc:
            logger.warning("discarding corrupt cache entry", extra={"point_id": str(point.id)})
            raise CacheError(f"point {point.id} is not a valid cache entry") from exc
