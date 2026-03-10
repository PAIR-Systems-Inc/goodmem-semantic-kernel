"""Async HTTP client for the GoodMem REST API (internal use only)."""

import json
import logging
from typing import Any
from urllib.parse import quote

import httpx

logger = logging.getLogger(__name__)


class GoodMemAsyncClient:
    """Async HTTP wrapper around the GoodMem REST API.

    This is an internal class; consumers should use ``GoodMemCollection``
    or ``GoodMemStore`` instead.

    Args:
        base_url: GoodMem server base URL (no trailing slash, no ``/v1``).
        api_key: API key sent in the ``x-api-key`` header.
        http_client: Optional pre-configured ``httpx.AsyncClient`` to use.
            If not provided, one is created and owned by this instance.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        http_client: httpx.AsyncClient | None = None,
        verify_ssl: bool = True,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key.strip()
        self._headers = {"x-api-key": self._api_key}
        self._owned = http_client is None
        self._client = http_client or httpx.AsyncClient(
            base_url=self._base_url,
            headers=self._headers,
            timeout=30.0,
            verify=verify_ssl,
        )

    async def aclose(self) -> None:
        """Close the underlying HTTP client (only if owned by this instance)."""
        if self._owned:
            await self._client.aclose()

    async def __aenter__(self) -> "GoodMemAsyncClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.aclose()

    # ------------------------------------------------------------------
    # Spaces
    # ------------------------------------------------------------------

    async def list_spaces(self, name_filter: str | None = None) -> list[dict[str, Any]]:
        """List spaces, optionally filtering by name.

        Args:
            name_filter: Optional exact-name filter.

        Returns:
            List of space dicts from the API.
        """
        all_spaces: list[dict[str, Any]] = []
        next_token: str | None = None

        while True:
            params: dict[str, Any] = {"maxResults": 1000}
            if next_token:
                params["nextToken"] = next_token
            if name_filter:
                params["nameFilter"] = name_filter

            response = await self._client.get("/v1/spaces", params=params)
            response.raise_for_status()

            data = response.json()
            all_spaces.extend(data.get("spaces", []))

            next_token = data.get("nextToken")
            if not next_token:
                break

        return all_spaces

    async def get_space(self, space_id: str) -> dict[str, Any] | None:
        """Get a space by ID.

        Returns:
            The space dict, or ``None`` on 404.
        """
        encoded = quote(space_id, safe="")
        response = await self._client.get(f"/v1/spaces/{encoded}")
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()

    async def create_space(
        self,
        name: str,
        embedder_id: str,
        space_id: str | None = None,
    ) -> dict[str, Any]:
        """Create a new space.

        Args:
            name: Human-readable space name (used as the collection name).
            embedder_id: Embedder UUID to attach to the space.
            space_id: Optional client-supplied UUID. Server generates one if omitted.

        Returns:
            The created space dict containing ``spaceId``.
        """
        payload: dict[str, Any] = {
            "name": name,
            "spaceEmbedders": [
                {"embedderId": embedder_id, "defaultRetrievalWeight": 1.0}
            ],
            "defaultChunkingConfig": {
                "recursive": {
                    "chunkSize": 512,
                    "chunkOverlap": 64,
                    "keepStrategy": "KEEP_END",
                    "lengthMeasurement": "CHARACTER_COUNT",
                }
            },
        }
        if space_id is not None:
            payload["spaceId"] = space_id

        response = await self._client.post("/v1/spaces", json=payload)
        response.raise_for_status()
        return response.json()

    async def delete_space(self, space_id: str) -> None:
        """Delete a space by ID."""
        encoded = quote(space_id, safe="")
        response = await self._client.delete(f"/v1/spaces/{encoded}")
        response.raise_for_status()

    # ------------------------------------------------------------------
    # Embedders
    # ------------------------------------------------------------------

    async def list_embedders(self) -> list[dict[str, Any]]:
        """List all configured embedders."""
        response = await self._client.get("/v1/embedders")
        response.raise_for_status()
        return response.json().get("embedders", [])

    # ------------------------------------------------------------------
    # Memories
    # ------------------------------------------------------------------

    async def create_memory(
        self,
        space_id: str,
        content: str,
        content_type: str = "text/plain",
        metadata: dict[str, Any] | None = None,
        memory_id: str | None = None,
    ) -> dict[str, Any]:
        """Create a new memory (text).

        Args:
            space_id: Target space UUID.
            content: Raw text content to embed.
            content_type: MIME type (default ``text/plain``).
            metadata: Optional JSONB metadata dict.
            memory_id: Optional client-supplied UUID for the memory.

        Returns:
            API response dict containing ``memoryId``.
        """
        payload: dict[str, Any] = {
            "spaceId": space_id,
            "originalContent": content,
            "contentType": content_type,
        }
        if metadata:
            payload["metadata"] = metadata
        if memory_id:
            payload["memoryId"] = memory_id

        response = await self._client.post("/v1/memories", json=payload)
        response.raise_for_status()
        return response.json()

    async def get_memories_batch(self, memory_ids: list[str]) -> list[dict[str, Any]]:
        """Batch-fetch memories by ID.

        Args:
            memory_ids: List of memory UUIDs to retrieve.

        Returns:
            List of memory dicts. Missing IDs are silently omitted.
        """
        if not memory_ids:
            return []
        response = await self._client.post(
            "/v1/memories:batchGet",
            json={"memoryIds": list(memory_ids)},
        )
        response.raise_for_status()
        return response.json().get("memories", [])

    async def delete_memory(self, memory_id: str) -> None:
        """Delete a memory by ID. Silently ignores 404."""
        encoded = quote(memory_id, safe="")
        response = await self._client.delete(f"/v1/memories/{encoded}")
        if response.status_code == 404:
            return
        response.raise_for_status()

    async def retrieve_memories(
        self,
        query: str,
        space_ids: list[str],
        top: int = 5,
        filter_expr: str | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic search over one or more spaces.

        Posts to ``/v1/memories:retrieve`` and parses the NDJSON response.
        Each line in the response is either a ``retrievedItem`` event
        (containing a chunk and score) or a ``memoryDefinition`` event
        (containing full memory metadata). This method correlates them and
        returns a unified list.

        Args:
            query: Natural-language search query (server embeds it).
            space_ids: List of space UUIDs to search.
            top: Maximum number of results to return.
            filter_expr: Optional filter expression (reserved for future use).

        Returns:
            List of dicts with keys ``chunk``, ``memory``, and ``score``:

            .. code-block:: python

                [
                    {
                        "chunk": {...},   # retrievedItem payload
                        "memory": {...},  # memoryDefinition payload (may be {})
                        "score": 0.87,
                    },
                    ...
                ]
        """
        payload: dict[str, Any] = {
            "message": query,
            "spaceKeys": [{"spaceId": sid} for sid in space_ids],
            "requestedSize": top,
        }
        if filter_expr:
            payload["filterExpression"] = filter_expr

        headers = {**self._headers, "Accept": "application/x-ndjson"}
        response = await self._client.post(
            "/v1/memories:retrieve",
            json=payload,
            headers=headers,
        )
        response.raise_for_status()

        # Parse NDJSON events.  Each line is one of:
        #   {"memoryDefinition": Memory}              — client-side memory cache
        #                                               (indexed 0, 1, 2, … by arrival order)
        #   {"retrievedItem": {"chunk": ChunkRef}}    — a result chunk with score
        #   {"resultSetBoundary": ...}                — stream markers (ignored)
        #   {"status": ...}                           — warnings (ignored)
        #
        # NOTE: retrievedItem.memory is NOT used by the server ("The server does not use
        # this field in the current implementation." — memory.proto L230).  Memory metadata
        # arrives as top-level `memoryDefinition` events.
        #
        # ChunkReference = {"chunk": MemoryChunk, "memoryIndex": int, "relevanceScore": float}
        # MemoryChunk    = {"chunkId": str, "memoryId": str, "chunkText": str, ...}
        # memoryIndex    = 0-based position of the parent Memory in memory_list
        memory_list: list[dict[str, Any]] = []   # Memory dicts, indexed by arrival order
        chunk_refs: list[dict[str, Any]] = []    # ChunkReference dicts

        for line in response.text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Failed to parse NDJSON line: %s", line)
                continue

            # Top-level memory definition (client-side cache, correlated by position)
            if "memoryDefinition" in event:
                memory_list.append(event["memoryDefinition"])
                continue

            item = event.get("retrievedItem")
            if item is None:
                continue
            if "chunk" in item:
                chunk_refs.append(item["chunk"])  # ChunkReference

        # Build correlated result list.
        # memoryIndex is the 0-based index of the parent memory in memory_list.
        # relevanceScore is a raw pgvector value (negative inner product: lower = more
        # similar).  We negate it so callers receive higher-is-better similarity scores.
        results: list[dict[str, Any]] = []
        for chunk_ref in chunk_refs:
            memory_chunk = chunk_ref.get("chunk", {})   # nested MemoryChunk
            memory_index = chunk_ref.get("memoryIndex")
            mem = (
                memory_list[memory_index]
                if (memory_index is not None and 0 <= memory_index < len(memory_list))
                else {}
            )
            raw_score = float(chunk_ref.get("relevanceScore", 0.0))
            results.append(
                {
                    "chunk": memory_chunk,
                    "memory": mem,
                    "score": -raw_score,  # negate: GoodMem returns lower-is-better
                }
            )

        return results
