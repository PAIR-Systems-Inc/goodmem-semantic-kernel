"""GoodMem VectorStoreCollection for Semantic Kernel."""

from __future__ import annotations

import ast
import asyncio
import logging
import sys
import time
import warnings
from collections.abc import Sequence
from typing import Any, ClassVar, Generic

from goodmem import AsyncGoodmem
from goodmem.errors import GoodMemError, NotFoundError
from semantic_kernel.data.vector import (
    GetFilteredRecordOptions,
    KernelSearchResults,
    SearchType,
    TModel,
    VectorSearch,
    VectorSearchOptions,
    VectorSearchResult,
    VectorStoreCollection,
)
from semantic_kernel.exceptions.vector_store_exceptions import (
    VectorStoreInitializationException,
    VectorStoreOperationException,
    VectorStoreOperationNotSupportedException,
)
from semantic_kernel.functions import KernelParameterMetadata, KernelPlugin

if sys.version_info >= (3, 12):
    from typing import override  # pragma: no cover
else:
    from typing_extensions import override  # pragma: no cover

from goodmem_semantic_kernel import filters as gm_filters
from goodmem_semantic_kernel._connection import GoodMemConnection
from goodmem_semantic_kernel._ids import require_uuid
from goodmem_semantic_kernel._results import (
    classify,
    hits_from_events,
    score_for_host,
)
from goodmem_semantic_kernel._typing import AsyncGoodmemClient
from goodmem_semantic_kernel.settings import GoodMemSettings

logger = logging.getLogger(__name__)

TKey = str


class GoodMemUpsertError(VectorStoreOperationException):
    """An upsert failed. Carries the keys that were written before it did.

    GoodMem has no update endpoint, so replacing a record means deleting the
    old memory and creating a new one. When the create fails, the connector
    restores the previous content; ``restored`` says whether that succeeded,
    and ``lost_key`` names the record if it did not.
    """

    def __init__(
        self,
        message: str,
        *,
        written_keys: list[str] | None = None,
        restored: bool | None = None,
        lost_key: str | None = None,
    ) -> None:
        super().__init__(message)
        self.written_keys = written_keys or []
        self.restored = restored
        self.lost_key = lost_key


class GoodMemCollection(
    VectorStoreCollection[TKey, TModel],
    VectorSearch[TKey, TModel],
    Generic[TModel],
):
    """Semantic Kernel VectorStoreCollection backed by GoodMem.

    Maps SK collections to GoodMem Spaces (1:1); the collection name is the
    space name. GoodMem embeds server-side, so no local embedding generator is
    required or used.

    Example::

        @vectorstoremodel
        @dataclass
        class Note:
            id: Annotated[str | None, VectorStoreField("key")] = None
            content: Annotated[str, VectorStoreField("data")] = ""
            tag: Annotated[str | None, VectorStoreField("data")] = None

        async with GoodMemCollection(
            record_type=Note,
            collection_name="my-notes",
            settings=GoodMemSettings(),
        ) as coll:
            await coll.ensure_collection_exists()
            keys = await coll.upsert(Note(content="Hello, world!"))
            results = await coll.search("hello", filter=lambda n: n.tag == "greeting")

    Args:
        record_type: The data model class decorated with ``@vectorstoremodel``.
        collection_name: GoodMem space name.
        settings: :class:`GoodMemSettings` (reads ``GOODMEM_*`` env vars).
        client: Optional ``goodmem.AsyncGoodmem`` to inject. The caller keeps
            ownership: its server, credentials and TLS settings are used as-is
            and it is never closed here.
        **kwargs: Forwarded to ``VectorStoreCollection``.
    """

    supported_key_types: ClassVar[set[str] | None] = {"str"}
    supported_search_types: ClassVar[set[SearchType]] = {SearchType.VECTOR}

    settings: GoodMemSettings

    def __init__(
        self,
        record_type: type[TModel],
        *,
        collection_name: str | None = None,
        settings: GoodMemSettings | None = None,
        client: AsyncGoodmem | None = None,
        connection: GoodMemConnection | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialise the GoodMem collection."""
        resolved_settings = settings or GoodMemSettings()  # type: ignore[call-arg]
        # A connection handed in by a store is shared: several collections use
        # it, so this one must not close it even though the store created it.
        shared = connection is not None
        conn = connection or GoodMemConnection(resolved_settings, client)

        super().__init__(  # type: ignore[call-arg]
            record_type=record_type,
            collection_name=collection_name or "",
            settings=resolved_settings,
            managed_client=conn.owns_client and not shared,
            **kwargs,
        )
        # VectorStoreCollection is a pydantic model; these are plain attributes.
        object.__setattr__(self, "_conn", conn)
        object.__setattr__(self, "_space_id_cache", {})

    @property
    def _client(self) -> AsyncGoodmemClient:
        connection: GoodMemConnection = self._conn  # type: ignore[attr-defined]
        return connection.client

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    @override
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Close the SDK client if we own it."""
        if self.managed_client:
            connection: GoodMemConnection = self._conn  # type: ignore[attr-defined]
            await connection.aclose()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_content_storage_name(self) -> str | None:
        """Return the storage name of the DATA field used as ``originalContent``.

        Resolution order: a field named ``content``; else the first ``str``
        field; else the first data field.
        """
        data_fields = self.definition.data_fields
        for field in data_fields:
            if field.name == "content":
                return field.storage_name or field.name
        for field in data_fields:
            if field.type_ == "str":
                return field.storage_name or field.name
        if data_fields:
            return data_fields[0].storage_name or data_fields[0].name
        return None

    async def _resolve_space_id(self) -> str:
        """Return the GoodMem space UUID for this collection, creating it if needed.

        A space is only reused when its embedder matches the configured one.
        The embedder decides how everything in the space is indexed and cannot
        be changed afterwards, so silently writing into a space built on a
        different embedder produces results that look fine and are not.
        """
        name = self.collection_name
        cache: dict[str, str] = self._space_id_cache  # type: ignore[attr-defined]
        if name in cache:
            return cache[name]

        configured = self.settings.embedder_id
        if configured:
            try:
                configured = require_uuid(configured, "embedder_id (GOODMEM_EMBEDDER_ID)")
            except ValueError as exc:
                raise VectorStoreInitializationException(str(exc)) from exc

        matches = [
            space
            async for space in await self._client.spaces.list(name_filter=name, max_items=1000)
            if space.name == name
        ]
        if len(matches) > 1:
            raise VectorStoreInitializationException(
                f"{len(matches)} GoodMem spaces are named {name!r}. Delete the "
                "duplicates or address the space by its UUID."
            )
        if matches:
            space = matches[0]
            if configured:
                existing = {e.embedder_id for e in (space.space_embedders or []) if e.embedder_id}
                if existing and configured not in existing:
                    raise VectorStoreInitializationException(
                        f"Space {name!r} is indexed by embedder(s) "
                        f"{', '.join(sorted(existing))}, but GOODMEM_EMBEDDER_ID is "
                        f"{configured}. An embedder cannot be changed after the "
                        "space is created; use the space's own embedder or a "
                        "different collection name."
                    )
            cache[name] = space.space_id
            return space.space_id

        embedder_id = configured
        if not embedder_id:
            raise VectorStoreInitializationException(
                f"Cannot create the GoodMem space {name!r}: no embedder is "
                "configured. Set GOODMEM_EMBEDDER_ID (or settings.embedder_id) "
                "to the embedder this collection should be indexed with. The "
                "connector will not pick one for you, because the choice is "
                "permanent for the space."
            )
        created = await self._client.spaces.create(
            name=name,
            space_embedders=[{"embedderId": embedder_id}],
        )
        cache[name] = created.space_id
        return created.space_id

    async def _wait_for_indexing(self, memory_id: str) -> None:
        """Block until a memory finishes indexing, or the timeout elapses."""
        memory_id = require_uuid(memory_id, "memory_id")
        deadline = time.monotonic() + self.settings.indexing_timeout
        while True:
            memory = await self._client.memories.get(id=memory_id)
            status = memory.processing_status
            if status == "COMPLETED":
                return
            if status == "FAILED":
                raise VectorStoreOperationException(f"GoodMem failed to index memory {memory_id}.")
            if time.monotonic() >= deadline:
                warnings.warn(
                    f"Memory {memory_id} was still {status} after "
                    f"{self.settings.indexing_timeout}s. It was written; it may "
                    "not be searchable yet.",
                    stacklevel=2,
                )
                return
            await asyncio.sleep(0.5)

    # ------------------------------------------------------------------
    # Serialization / Deserialization
    # ------------------------------------------------------------------

    @override
    def _serialize_dicts_to_store_models(
        self,
        records: Sequence[dict[str, Any]],
        **kwargs: Any,
    ) -> Sequence[Any]:
        """Convert SK record dicts → GoodMem create-memory payloads."""
        key_sname = self.definition.key_field_storage_name
        content_sname = self._get_content_storage_name()

        result = []
        for record in records:
            key_val = record.get(key_sname)

            metadata: dict[str, Any] = {}
            for field in self.definition.data_fields:
                sname = field.storage_name or field.name
                if sname == content_sname:
                    continue
                val = record.get(sname)
                if val is not None:
                    metadata[sname] = val

            store_model: dict[str, Any] = {
                "originalContent": (record.get(content_sname) if content_sname else "") or "",
                "contentType": "text/plain",
            }
            # None lets the server assign an id. Anything else, "" included,
            # is a key and must pass the UUID check in _inner_upsert.
            if key_val is not None:
                store_model["memoryId"] = str(key_val)
            if metadata:
                store_model["metadata"] = metadata

            result.append(store_model)
        return result

    @override
    def _deserialize_store_models_to_dicts(
        self,
        records: Sequence[Any],
        **kwargs: Any,
    ) -> Sequence[dict[str, Any]]:
        """Convert GoodMem memory dicts → SK record dicts."""
        key_sname = self.definition.key_field_storage_name
        content_sname = self._get_content_storage_name()
        vec_snames = {f.storage_name or f.name for f in self.definition.vector_fields}

        result = []
        for mem in records:
            d: dict[str, Any] = {key_sname: mem.get("memoryId")}

            if content_sname:
                d[content_sname] = mem.get("originalContent", "")

            gm_metadata: dict[str, Any] = mem.get("metadata") or {}
            for field in self.definition.data_fields:
                sname = field.storage_name or field.name
                if sname == content_sname:
                    continue
                d[sname] = gm_metadata.get(sname)

            # GoodMem embeds server-side and does not return vectors.
            for sname in vec_snames:
                d[sname] = None

            result.append(d)
        return result

    # ------------------------------------------------------------------
    # Collection lifecycle
    # ------------------------------------------------------------------

    @override
    async def collection_exists(self, **kwargs: Any) -> bool:
        """Return True if a GoodMem space with this collection's name exists."""
        async for space in await self._client.spaces.list(
            name_filter=self.collection_name, max_items=1000
        ):
            if space.name == self.collection_name:
                return True
        return False

    @override
    async def ensure_collection_exists(self, **kwargs: Any) -> None:
        """Create the GoodMem space if it does not already exist."""
        await self._resolve_space_id()

    @override
    async def ensure_collection_deleted(self, **kwargs: Any) -> None:
        """Delete the GoodMem space for this collection, if it exists."""
        async for space in await self._client.spaces.list(
            name_filter=self.collection_name, max_items=1000
        ):
            if space.name == self.collection_name:
                await self._client.spaces.delete(id=require_uuid(space.space_id, "space_id"))
                self._space_id_cache.pop(self.collection_name, None)  # type: ignore[attr-defined]
                return

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------

    async def _replace_existing(
        self,
        space_id: str,
        memory_id: str,
        store_model: dict[str, Any],
        written: list[str],
    ) -> str:
        """Replace a memory that already exists, restoring it if the write fails.

        GoodMem has no update endpoint (``PUT``/``PATCH`` return 404) and
        rejects a create that reuses an existing id with 409, so the old
        memory must go before the new one can be written. To keep a failed
        update from destroying data, the current content and metadata are read
        first and written back if the create fails.
        """
        try:
            previous_content = await self._client.memories.content(id=memory_id)
            previous = await self._client.memories.get(id=memory_id)
            previous_metadata = dict(previous.metadata or {})
            previous_type = previous.content_type or "text/plain"
        except NotFoundError:
            # Nothing to replace after all; fall through to a plain create.
            return await self._create(space_id, store_model, memory_id)

        await self._client.memories.delete(id=memory_id)
        try:
            return await self._create(space_id, store_model, memory_id)
        except Exception as exc:
            restored = True
            try:
                await self._client.memories.create(
                    space_id=space_id,
                    memory_id=memory_id,
                    original_content=previous_content.decode("utf-8", errors="replace"),
                    content_type=previous_type,
                    metadata=previous_metadata or None,
                )
            except Exception:  # pragma: no cover - restore of a restore
                restored = False
            raise GoodMemUpsertError(
                f"Updating record {memory_id} failed: {exc}. "
                + (
                    "The previous version was restored."
                    if restored
                    else "The previous version could NOT be restored and is lost."
                ),
                written_keys=written,
                restored=restored,
                lost_key=None if restored else memory_id,
            ) from exc

    async def _create(
        self,
        space_id: str,
        store_model: dict[str, Any],
        memory_id: str | None,
    ) -> str:
        memory = await self._client.memories.create(
            space_id=space_id,
            original_content=store_model.get("originalContent", ""),
            content_type=store_model.get("contentType", "text/plain"),
            metadata=store_model.get("metadata"),
            memory_id=memory_id,
        )
        return memory.memory_id

    @override
    async def _inner_upsert(
        self,
        records: Sequence[Any],
        **kwargs: Any,
    ) -> Sequence[TKey]:
        """Write records to GoodMem, keeping already-written keys on failure.

        Every key is checked before anything is sent, so one bad key refuses
        the whole batch rather than leaving it half written.
        """
        memory_ids = [
            None if model.get("memoryId") is None else require_uuid(model["memoryId"], "key")
            for model in records
        ]
        space_id = await self._resolve_space_id()
        keys: list[str] = []

        for store_model, memory_id in zip(records, memory_ids, strict=True):
            try:
                if memory_id and await self._exists(memory_id):
                    returned = await self._replace_existing(space_id, memory_id, store_model, keys)
                else:
                    returned = await self._create(space_id, store_model, memory_id)
            except GoodMemUpsertError:
                raise
            except GoodMemError as exc:
                raise GoodMemUpsertError(
                    f"Writing to GoodMem failed: {exc}", written_keys=keys
                ) from exc
            keys.append(returned)

        if self.settings.wait_for_indexing:
            for key in keys:
                await self._wait_for_indexing(key)

        return keys

    async def _exists(self, memory_id: str) -> bool:
        try:
            await self._client.memories.get(id=memory_id)
        except NotFoundError:
            return False
        return True

    @override
    async def _inner_get(
        self,
        keys: Sequence[TKey] | None = None,
        options: GetFilteredRecordOptions | None = None,
        **kwargs: Any,
    ) -> Sequence[Any] | None:
        """Fetch memories by key, in the order asked for, with their content.

        The batch endpoint answers with ``{"results": [{"success", "memory"}]}``
        and omits the content unless it is asked for. Reading ``memories`` from
        that payload, as this connector used to, always yielded nothing.
        """
        if not keys:
            return None

        # Ids travel in the body here, not the path; checked for consistency.
        keys = [require_uuid(key, "key") for key in keys]
        response = await self._client.memories.batch_get(memory_ids=keys, include_content=True)
        by_id: dict[str, dict[str, Any]] = {}
        for result in response.results or []:
            if not result.success or result.memory is None:
                continue
            memory = result.memory
            content = memory.original_content
            if isinstance(content, bytes):
                content = content.decode("utf-8", errors="replace")
            by_id[memory.memory_id] = {
                "memoryId": memory.memory_id,
                "originalContent": content or "",
                "metadata": dict(memory.metadata or {}),
            }

        found = [by_id[key] for key in keys if key in by_id]
        return found or None

    @override
    async def _inner_delete(self, keys: Sequence[TKey], **kwargs: Any) -> None:
        """Delete memories by key, tolerating keys that are already gone.

        Every key is checked before the first delete: a key is a URL path
        segment, and ``"../spaces/<id>"`` would otherwise delete a whole space.
        """
        memory_ids = [require_uuid(key, "key") for key in keys]
        for memory_id in memory_ids:
            try:
                await self._client.memories.delete(id=memory_id)
            except NotFoundError:
                continue

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    @override
    async def _inner_search(
        self,
        search_type: SearchType,
        options: VectorSearchOptions,
        values: Any | None = None,
        vector: Sequence[float | int] | None = None,
        **kwargs: Any,
    ) -> KernelSearchResults[VectorSearchResult[TModel]]:
        """Search GoodMem, reporting any problem the server reported.

        When the server reports a problem, the results still come back and
        ``KernelSearchResults.metadata`` carries ``goodmem_partial`` and
        ``goodmem_statuses``. A search that reported a problem and found
        nothing returns empty with a warning; it never raises. This is the
        retrieval status contract every GoodMem integration follows.
        """
        if vector is not None:
            raise VectorStoreOperationNotSupportedException(
                "GoodMem does not support pre-computed vector search via the "
                "REST API. Pass text via the 'values' argument — GoodMem will "
                "embed it server-side."
            )

        reranker_id = self.settings.reranker_id
        if reranker_id:
            reranker_id = require_uuid(reranker_id, "reranker_id")

        space_id = await self._resolve_space_id()
        space_key: dict[str, Any] = {"spaceId": space_id}
        if filter_expression := self._build_filter(options.filter):
            space_key["filter"] = (
                filter_expression
                if isinstance(filter_expression, str)
                else gm_filters.all_of(*filter_expression)
            )

        request: dict[str, Any] = {
            "message": str(values) if values is not None else "",
            "space_keys": [space_key],
            "requested_size": options.top,
        }
        if reranker_id:
            request["reranker_id"] = reranker_id

        events, truncated = await self._collect(request)

        statuses, degraded = classify(events)
        if truncated is not None:
            # A stream that stops mid-line must not discard the events that
            # arrived before it; report it the way a server status is reported.
            statuses.append({"code": "MALFORMED_STREAM", "message": str(truncated)})
            degraded = True
        hits = hits_from_events(events, reranked=bool(reranker_id))[: options.top]

        if degraded and not hits:
            # Contract Q4b: empty plus a flag, never an exception.
            warnings.warn(
                f"GoodMem reported a problem and returned no results: {statuses}",
                stacklevel=2,
            )
            logger.warning("GoodMem search returned no results: %s", statuses)

        metadata: dict[str, Any] = {"goodmem_partial": degraded}
        if statuses:
            metadata["goodmem_statuses"] = statuses

        return KernelSearchResults(
            results=self._get_vector_search_results_from_results(hits, options),
            total_count=len(hits),
            metadata=metadata,
        )

    async def _collect(self, request: dict[str, Any]) -> tuple[list[Any], Exception | None]:
        """Read the retrieval stream, keeping whatever arrived before a break.

        The SDK raises on a malformed NDJSON line. Collecting the stream in one
        call would therefore throw away every event that had already been
        parsed, turning a truncated response into a total failure.
        """
        events: list[Any] = []
        try:
            stream = await self._client.memories.retrieve(**request)
            async with stream as open_stream:
                async for event in open_stream:
                    events.append(event)
        except GoodMemError as exc:
            if not events:
                raise VectorStoreOperationException(f"GoodMem search failed: {exc}") from exc
            return events, exc
        return events, None

    @override
    def _get_record_from_result(self, result: Any) -> Any:
        """Extract the record payload from a joined retrieval hit.

        Content comes from the chunk: GoodMem's ``originalContent`` is
        write-only and is null in retrieve responses.
        """
        return {
            "memoryId": result.get("memory_id"),
            "originalContent": result.get("chunk_text", ""),
            "metadata": result.get("metadata") or {},
        }

    @override
    def _get_score_from_result(self, result: Any) -> float | None:
        """Return the score under Semantic Kernel's higher-is-better convention."""
        return score_for_host(result)

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------

    def _field_name(self, node: ast.AST) -> str:
        """Return the metadata field a lambda attribute refers to."""
        if isinstance(node, ast.Attribute):
            name = node.attr
        elif isinstance(node, ast.Name):
            name = node.id
        else:
            raise VectorStoreOperationNotSupportedException(
                f"Unsupported filter target: {ast.dump(node)}"
            )
        content_sname = self._get_content_storage_name()
        if name == content_sname:
            raise VectorStoreOperationNotSupportedException(
                f"{name!r} is stored as the memory's content, not as metadata, "
                "so it cannot be filtered on. Filter on a metadata field."
            )
        for field in self.definition.data_fields:
            if name in (field.name, field.storage_name):
                return field.storage_name or field.name
        raise VectorStoreOperationNotSupportedException(
            f"{name!r} is not a field on this collection's record type."
        )

    @override
    def _lambda_parser(self, node: ast.AST) -> Any:
        """Translate a Semantic Kernel filter lambda into a GoodMem expression.

        Supports ``==``, ``!=``, ``<``, ``<=``, ``>``, ``>=``, ``in``,
        ``not in``, ``and``, ``or`` and ``not``. Values are quoted and cast by
        :mod:`goodmem_semantic_kernel.filters` using the escaping and casts the
        server actually accepts, so a value containing an apostrophe is a
        value, never syntax.
        """
        match node:
            case ast.Compare():
                if len(node.ops) > 1:
                    parts = []
                    for index in range(len(node.ops)):
                        left = node.left if index == 0 else node.comparators[index - 1]
                        parts.append(
                            self._lambda_parser(
                                ast.Compare(
                                    left=left,
                                    ops=[node.ops[index]],
                                    comparators=[node.comparators[index]],
                                )
                            )
                        )
                    return gm_filters.all_of(*parts)

                field = self._field_name(node.left)
                value = self._lambda_parser(node.comparators[0])
                match node.ops[0]:
                    case ast.Eq():
                        return gm_filters.compare(field, "=", value)
                    case ast.NotEq():
                        return gm_filters.compare(field, "!=", value)
                    case ast.Gt():
                        return gm_filters.compare(field, ">", value)
                    case ast.GtE():
                        return gm_filters.compare(field, ">=", value)
                    case ast.Lt():
                        return gm_filters.compare(field, "<", value)
                    case ast.LtE():
                        return gm_filters.compare(field, "<=", value)
                    case ast.In():
                        return gm_filters.is_in(field, value)
                    case ast.NotIn():
                        return gm_filters.negate(gm_filters.is_in(field, value))
                raise VectorStoreOperationNotSupportedException(
                    f"Unsupported filter operator: {type(node.ops[0]).__name__}"
                )
            case ast.BoolOp():
                parts = [self._lambda_parser(value) for value in node.values]
                if isinstance(node.op, ast.And):
                    return gm_filters.all_of(*parts)
                return gm_filters.any_of(*parts)
            case ast.UnaryOp():
                if isinstance(node.op, ast.Not):
                    return gm_filters.negate(self._lambda_parser(node.operand))
                raise VectorStoreOperationNotSupportedException(
                    f"Unsupported unary operator: {type(node.op).__name__}"
                )
            case ast.Constant():
                return node.value
            case ast.List() | ast.Tuple() | ast.Set():
                return [self._lambda_parser(element) for element in node.elts]
            case ast.Attribute():
                return self._field_name(node)
            case ast.Name():
                # Semantic Kernel parses the lambda's *source*, so a variable
                # in it cannot be resolved to a value here.
                raise VectorStoreOperationNotSupportedException(
                    f"Filter values must be literals; {node.id!r} is a variable. "
                    "Inline the value, or pass a GoodMem filter expression as a "
                    "string."
                )
            case ast.Lambda():
                return self._lambda_parser(node.body)
            case _:
                raise VectorStoreOperationNotSupportedException(
                    f"Unsupported filter expression: {type(node).__name__}"
                )

    # ------------------------------------------------------------------
    # Plugin convenience
    # ------------------------------------------------------------------

    def as_plugin(
        self,
        name: str = "memory",
        description: str = "Search long-term memory for relevant facts and context.",
        *,
        function_name: str = "recall",
        top: int = 3,
    ) -> KernelPlugin:
        """Wrap this collection as a :class:`KernelPlugin` for an agent.

        The model supplies a query and a result count; the space, filters and
        credentials stay with the developer.
        """
        return KernelPlugin(
            name=name,
            description=description,
            functions=[
                self.create_search_function(
                    function_name=function_name,
                    description=description,
                    parameters=[
                        KernelParameterMetadata(
                            name="query",
                            description="What to search for in memory.",
                            type="str",
                            is_required=True,
                            type_object=str,
                        ),
                        KernelParameterMetadata(
                            name="top",
                            description=f"Number of memories to retrieve (default {top}).",
                            type="int",
                            default_value=top,
                            type_object=int,
                        ),
                    ],
                    string_mapper=lambda r: str(getattr(r.record, "content", "")),
                ),
            ],
        )
