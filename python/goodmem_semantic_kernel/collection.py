"""GoodMem VectorStoreCollection for Semantic Kernel."""

import logging
import sys
from ast import AST
from collections.abc import Sequence
from typing import Any, ClassVar, Generic

from semantic_kernel.data.vector import (
    FieldTypes,
    GetFilteredRecordOptions,
    KernelSearchResults,
    SearchType,
    TModel,
    VectorSearch,
    VectorSearchOptions,
    VectorSearchResult,
    VectorStoreCollection,
)
from semantic_kernel.functions import KernelParameterMetadata, KernelPlugin
from semantic_kernel.exceptions.vector_store_exceptions import (
    VectorStoreInitializationException,
    VectorStoreOperationException,
    VectorStoreOperationNotSupportedException,
)

if sys.version_info >= (3, 12):
    from typing import override  # pragma: no cover
else:
    from typing_extensions import override  # pragma: no cover

from goodmem_semantic_kernel._client import GoodMemAsyncClient
from goodmem_semantic_kernel.settings import GoodMemSettings

logger = logging.getLogger(__name__)

TKey = str


class GoodMemCollection(
    VectorStoreCollection[TKey, TModel],
    VectorSearch[TKey, TModel],
    Generic[TModel],
):
    """Semantic Kernel VectorStoreCollection backed by GoodMem.

    Maps SK collections to GoodMem Spaces (1:1). The collection name is the
    human-readable space name.  Server-side embedding means no local embedding
    generator is required (or used even if provided).

    Example::

        @vectorstoremodel
        @dataclass
        class Note:
            id: Annotated[str | None, VectorStoreField("key")] = None
            content: Annotated[str, VectorStoreField("data")]
            tag: Annotated[str | None, VectorStoreField("data")] = None

        async with GoodMemCollection(
            record_type=Note,
            collection_name="my-notes",
            settings=GoodMemSettings(),
        ) as coll:
            await coll.ensure_collection_exists()
            keys = await coll.upsert(Note(content="Hello, world!"))
            results = await coll.search("hello")

    Args:
        record_type: The data model class decorated with ``@vectorstoremodel``.
        collection_name: GoodMem space name.  If omitted, derived from
            the data model's ``__kernel_vectorstoremodel_definition__``.
        settings: :class:`GoodMemSettings` (reads ``GOODMEM_*`` env vars by
            default).
        client: Optional pre-built :class:`GoodMemAsyncClient` to inject.
            When provided, ``managed_client`` is set to ``False`` and the
            caller owns the client lifetime.
        **kwargs: Forwarded to :class:`~semantic_kernel.data.vector.VectorStoreCollection`.
    """

    supported_key_types: ClassVar[set[str] | None] = {"str"}
    supported_search_types: ClassVar[set[SearchType]] = {SearchType.VECTOR}

    settings: GoodMemSettings
    # Internal HTTP client.  Set in model_post_init; public for injection.
    _http: GoodMemAsyncClient
    # collection_name → resolved GoodMem space UUID
    _space_id_cache: dict[str, str]

    def __init__(
        self,
        record_type: type[object],
        *,
        collection_name: str | None = None,
        settings: GoodMemSettings | None = None,
        client: GoodMemAsyncClient | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialise the GoodMem collection."""
        resolved_settings = settings or GoodMemSettings()
        managed = client is None

        super().__init__(
            record_type=record_type,
            collection_name=collection_name or "",
            settings=resolved_settings,
            managed_client=managed,
            **kwargs,
        )
        # Bypass pydantic private-attr restrictions by using object.__setattr__
        object.__setattr__(
            self,
            "_http",
            client
            or GoodMemAsyncClient(
                base_url=resolved_settings.base_url,
                api_key=resolved_settings.api_key.get_secret_value(),
                verify_ssl=resolved_settings.verify_ssl,
            ),
        )
        object.__setattr__(self, "_space_id_cache", {})

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    @override
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Close the HTTP client if we own it."""
        if self.managed_client:
            await self._http.aclose()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_content_storage_name(self) -> str | None:
        """Return the storage name of the DATA field used as ``originalContent``.

        Resolution order:
        1. DATA field whose ``name`` is ``"content"``.
        2. First DATA field whose ``type_`` is ``"str"``.
        3. First DATA field unconditionally.
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
        """Return the GoodMem space UUID for the current collection_name.

        Creates the space (and resolves the embedder) if it does not yet exist.
        Results are cached on the instance.
        """
        name = self.collection_name
        if name in self._space_id_cache:
            return self._space_id_cache[name]

        spaces = await self._http.list_spaces(name_filter=name)
        for space in spaces:
            if space.get("name") == name:
                sid = space["spaceId"]
                self._space_id_cache[name] = sid
                return sid

        # Space doesn't exist — create it.
        embedder_id = await self._resolve_embedder_id()
        space = await self._http.create_space(name=name, embedder_id=embedder_id)
        sid = space["spaceId"]
        self._space_id_cache[name] = sid
        return sid

    async def _resolve_embedder_id(self) -> str:
        """Return a valid embedder UUID.

        Uses ``settings.embedder_id`` when set; otherwise picks the first
        available embedder.  Raises if none are configured.
        """
        if self.settings.embedder_id:
            return self.settings.embedder_id

        embedders = await self._http.list_embedders()
        if embedders:
            eid = embedders[0].get("embedderId")
            if eid:
                return eid

        raise VectorStoreInitializationException(
            "No embedders are configured in GoodMem.  "
            "Create at least one embedder via the GoodMem API or set "
            "GOODMEM_EMBEDDER_ID to the UUID of an existing embedder."
        )

    # ------------------------------------------------------------------
    # Serialization / Deserialization
    # ------------------------------------------------------------------

    @override
    def _serialize_dicts_to_store_models(
        self,
        records: Sequence[dict[str, Any]],
        **kwargs: Any,
    ) -> Sequence[Any]:
        """Convert SK record dicts → GoodMem create-memory payloads.

        The space ID is not included here; it is injected in
        :meth:`_inner_upsert` at write time.
        """
        key_sname = self.definition.key_field_storage_name
        content_sname = self._get_content_storage_name()
        vec_snames = {f.storage_name or f.name for f in self.definition.vector_fields}

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
                "originalContent": record.get(content_sname) or "",
                "contentType": "text/plain",
            }
            if key_val:
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
            d: dict[str, Any] = {}

            # Key
            d[key_sname] = mem.get("memoryId")

            # Content
            if content_sname:
                d[content_sname] = mem.get("originalContent", "")

            # Metadata fields → individual SK data fields
            gm_metadata: dict[str, Any] = mem.get("metadata") or {}
            for field in self.definition.data_fields:
                sname = field.storage_name or field.name
                if sname == content_sname:
                    continue
                d[sname] = gm_metadata.get(sname)

            # Vector fields are not returned by GoodMem
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
        spaces = await self._http.list_spaces(name_filter=self.collection_name)
        return any(s.get("name") == self.collection_name for s in spaces)

    @override
    async def ensure_collection_exists(self, **kwargs: Any) -> None:
        """Create the GoodMem space if it does not already exist.

        Raises:
            VectorStoreInitializationException: If no embedder is available and
                ``GOODMEM_EMBEDDER_ID`` is not set.
        """
        await self._resolve_space_id()

    @override
    async def ensure_collection_deleted(self, **kwargs: Any) -> None:
        """Delete the GoodMem space for this collection, if it exists."""
        spaces = await self._http.list_spaces(name_filter=self.collection_name)
        for space in spaces:
            if space.get("name") == self.collection_name:
                await self._http.delete_space(space["spaceId"])
                self._space_id_cache.pop(self.collection_name, None)
                return

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------

    @override
    async def _inner_upsert(
        self,
        records: Sequence[Any],
        **kwargs: Any,
    ) -> Sequence[TKey]:
        """Write records to GoodMem using delete-then-create upsert semantics.

        GoodMem memories are immutable (no PUT endpoint), so upsert is
        implemented as: delete existing memory (if key provided, ignore 404),
        then POST a new memory.  The server-generated ``memoryId`` is returned
        for records without a client-supplied key.
        """
        space_id = await self._resolve_space_id()
        keys: list[str] = []

        for store_model in records:
            memory_id: str | None = store_model.get("memoryId")

            if memory_id:
                # Delete first — ignore 404 (may not exist yet)
                await self._http.delete_memory(memory_id)

            response = await self._http.create_memory(
                space_id=space_id,
                content=store_model.get("originalContent", ""),
                content_type=store_model.get("contentType", "text/plain"),
                metadata=store_model.get("metadata"),
                memory_id=memory_id,
            )
            returned_id = response.get("memoryId") or memory_id or ""
            keys.append(returned_id)

        return keys

    @override
    async def _inner_get(
        self,
        keys: Sequence[TKey] | None = None,
        options: GetFilteredRecordOptions | None = None,
        **kwargs: Any,
    ) -> Sequence[Any] | None:
        """Fetch memories by ID using batchGet."""
        if not keys:
            return None
        memories = await self._http.get_memories_batch(list(keys))
        if not memories:
            return None
        return memories

    @override
    async def _inner_delete(self, keys: Sequence[TKey], **kwargs: Any) -> None:
        """Delete memories by key (ignores 404 for each)."""
        for key in keys:
            await self._http.delete_memory(key)

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
        """Perform semantic search against GoodMem.

        Only text search (``values=``) is supported.  GoodMem embeds the
        query server-side, so no local embedding generator is required.

        Raises:
            VectorStoreOperationNotSupportedException: If a pre-computed
                ``vector=`` is provided (GoodMem's REST API does not accept
                raw query vectors).
            VectorStoreOperationNotSupportedException: If ``options.filter``
                is set (filter support is deferred to a future version).
        """
        if vector is not None:
            raise VectorStoreOperationNotSupportedException(
                "GoodMem does not support pre-computed vector search via the REST API.  "
                "Pass text via the 'values' argument — GoodMem will embed it server-side."
            )

        if options.filter:
            raise VectorStoreOperationNotSupportedException(
                "Filter expressions are not yet supported by the GoodMem SK connector.  "
                "Remove the 'filter' parameter or post-filter the results in application code."
            )

        space_id = await self._resolve_space_id()

        results = await self._http.retrieve_memories(
            query=str(values) if values is not None else "",
            space_ids=[space_id],
            top=options.top,
        )

        return KernelSearchResults(
            results=self._get_vector_search_results_from_results(results, options),
            total_count=len(results),
        )

    @override
    def _get_record_from_result(self, result: Any) -> Any:
        """Extract the GoodMem memory dict from a retrieve result.

        The retrieve results from :meth:`~._client.GoodMemAsyncClient.retrieve_memories`
        are ``{"chunk": {chunkData}, "memory": {memoryDef}, "score": float}`` dicts.

        Content MUST come from ``chunk.chunkText`` — GoodMem's ``originalContent``
        field is write-only and is always ``null`` in retrieve/get responses.
        Metadata comes from the correlated ``memoryDefinition`` when available.
        """
        chunk = result.get("chunk") or {}
        mem = result.get("memory") or {}
        return {
            "memoryId": chunk.get("memoryId") or mem.get("memoryId"),
            "originalContent": chunk.get("chunkText", ""),
            "metadata": mem.get("metadata") or {},
        }

    @override
    def _get_score_from_result(self, result: Any) -> float | None:
        """Extract the similarity score from a retrieve result."""
        return result.get("score")

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
        """Wrap this collection as a :class:`KernelPlugin` ready to pass to an agent.

        This is a convenience wrapper around :meth:`create_search_function` that
        provides sensible defaults so you don't need to know the SK plugin internals
        to wire GoodMem into an agent.

        Example::

            plugin = collection.as_plugin(name="memory")

            agent = ChatCompletionAgent(
                service=OpenAIChatCompletion(),
                function_choice_behavior=FunctionChoiceBehavior.Auto(),
                plugins=[plugin],
            )

        Args:
            name: Plugin name shown to the LLM (e.g. ``"memory"``).
            description: Plugin-level description surfaced to the LLM.
            function_name: Name of the search function inside the plugin
                (default ``"recall"``).
            top: Default number of memories to retrieve per search call.

        Returns:
            A :class:`KernelPlugin` containing a single ``recall`` search function.
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
                    string_mapper=lambda r: r.record.content,
                ),
            ],
        )

    # ------------------------------------------------------------------
    # Filter support (not implemented in v1)
    # ------------------------------------------------------------------

    @override
    def _lambda_parser(self, node: AST) -> Any:
        raise VectorStoreOperationNotSupportedException(
            "Filter expressions are not supported by the GoodMem SK connector in v1.  "
            "Remove the 'filter' parameter to perform unfiltered search."
        )
