"""GoodMem VectorStore for Semantic Kernel."""

import sys
from collections.abc import Sequence
from typing import Any

from semantic_kernel.connectors.ai.embedding_generator_base import EmbeddingGeneratorBase
from semantic_kernel.data.vector import (
    TModel,
    VectorStore,
    VectorStoreCollectionDefinition,
    VectorStoreField,
)

if sys.version_info >= (3, 12):
    from typing import override  # pragma: no cover
else:
    from typing_extensions import override  # pragma: no cover

from goodmem import AsyncGoodmem

from goodmem_semantic_kernel._connection import GoodMemConnection
from goodmem_semantic_kernel.collection import GoodMemCollection
from goodmem_semantic_kernel.settings import GoodMemSettings


class GoodMemStore(VectorStore):
    """Semantic Kernel VectorStore backed by GoodMem.

    Acts as a factory for :class:`GoodMemCollection` instances and provides
    an enumeration of available spaces (collections).  A single underlying
    :class:`~._client.GoodMemAsyncClient` is shared across all collections
    created by this store instance.

    Usage (context-manager pattern)::

        async with GoodMemStore(settings=GoodMemSettings()) as store:
            collection = store.get_collection(
                record_type=MyModel,
                collection_name="my-space",
            )
            await collection.ensure_collection_exists()
            await collection.upsert(MyModel(content="Hello"))

    Usage (manual lifecycle)::

        store = GoodMemStore(settings=GoodMemSettings())
        try:
            collection = store.get_collection(MyModel, collection_name="my-space")
            await collection.ensure_collection_exists()
        finally:
            await store.__aexit__(None, None, None)

    Args:
        settings: :class:`GoodMemSettings` (reads ``GOODMEM_*`` env vars by
            default).
        client: Optional pre-built :class:`GoodMemAsyncClient` to inject.
            When provided, the caller owns the client lifetime.
        **kwargs: Forwarded to :class:`~semantic_kernel.data.vector.VectorStore`.
    """

    settings: GoodMemSettings

    def __init__(
        self,
        settings: GoodMemSettings | None = None,
        client: AsyncGoodmem | None = None,
        **kwargs: Any,
    ) -> None:
        resolved_settings = settings or GoodMemSettings()  # type: ignore[call-arg]
        conn = GoodMemConnection(resolved_settings, client)

        # `settings` is a field on this subclass, not on VectorStore's __init__.
        super().__init__(  # type: ignore[call-arg]
            managed_client=conn.owns_client, settings=resolved_settings, **kwargs
        )
        object.__setattr__(self, "_conn", conn)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    @override
    async def __aenter__(self) -> "GoodMemStore":
        return self

    @override
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Close the shared SDK client if we own it."""
        if self.managed_client:
            connection: GoodMemConnection = self._conn  # type: ignore[attr-defined]
            await connection.aclose()

    # ------------------------------------------------------------------
    # VectorStore interface
    # ------------------------------------------------------------------

    @override
    def get_collection(
        self,
        record_type: type[TModel],
        *,
        definition: VectorStoreCollectionDefinition | None = None,
        collection_name: str | None = None,
        embedding_generator: EmbeddingGeneratorBase | None = None,
        **kwargs: Any,
    ) -> GoodMemCollection:
        """Return a :class:`GoodMemCollection` connected to this store's client.

        The returned collection shares the store's HTTP client and therefore
        shares its lifecycle — do **not** close the collection independently
        when it was obtained from a store.

        Args:
            record_type: The data model class.
            definition: Optional explicit collection definition.
            collection_name: GoodMem space name.
            embedding_generator: Accepted for interface compatibility; unused
                (GoodMem embeds server-side).
            **kwargs: Forwarded to :class:`GoodMemCollection`.

        Returns:
            A :class:`GoodMemCollection` instance with ``managed_client=False``.
        """
        return GoodMemCollection(
            record_type=record_type,
            definition=definition,
            collection_name=collection_name,
            settings=self.settings,
            connection=self._conn,  # type: ignore[attr-defined]  # shared client
            **kwargs,
        )

    @property
    def _connection(self) -> GoodMemConnection:
        return self._conn  # type: ignore[attr-defined]

    @override
    async def list_collection_names(self, **kwargs: Any) -> Sequence[str]:
        """Return the names of all GoodMem spaces visible to this API key.

        Follows SDK pagination rather than stopping at the first page.
        """
        return [
            space.name
            async for space in await self._connection.client.spaces.list(max_items=1000)
            if space.name
        ]

    @override
    async def ensure_collection_deleted(self, collection_name: str) -> None:
        """Delete the GoodMem space named ``collection_name``, if it exists.

        Semantic Kernel's default swallows ``VectorStoreOperationException``.
        That is what the collection raises when it refuses to delete a space
        the server listed under an id that is not a UUID, so the default would
        report a delete that never happened as done.
        """
        definition = VectorStoreCollectionDefinition(fields=[VectorStoreField("key", name="id")])
        collection = self.get_collection(
            record_type=dict, definition=definition, collection_name=collection_name
        )
        await collection.ensure_collection_deleted()
