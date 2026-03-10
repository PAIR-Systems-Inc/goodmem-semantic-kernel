"""GoodMem VectorStore for Semantic Kernel."""

import sys
from collections.abc import Sequence
from typing import Any

from semantic_kernel.connectors.ai.embedding_generator_base import EmbeddingGeneratorBase
from semantic_kernel.data.vector import (
    TModel,
    VectorStore,
    VectorStoreCollection,
    VectorStoreCollectionDefinition,
)

if sys.version_info >= (3, 12):
    from typing import override  # pragma: no cover
else:
    from typing_extensions import override  # pragma: no cover

from goodmem_semantic_kernel._client import GoodMemAsyncClient
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
    _http: GoodMemAsyncClient

    def __init__(
        self,
        settings: GoodMemSettings | None = None,
        client: GoodMemAsyncClient | None = None,
        **kwargs: Any,
    ) -> None:
        resolved_settings = settings or GoodMemSettings()
        managed = client is None

        super().__init__(managed_client=managed, settings=resolved_settings, **kwargs)

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

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    @override
    async def __aenter__(self) -> "GoodMemStore":
        return self

    @override
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Close the shared HTTP client if we own it."""
        if self.managed_client:
            await self._http.aclose()

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
            client=self._http,  # shared client — managed_client=False
            **kwargs,
        )

    @override
    async def list_collection_names(self, **kwargs: Any) -> Sequence[str]:
        """Return the names of all GoodMem spaces visible to this API key."""
        spaces = await self._http.list_spaces()
        return [s["name"] for s in spaces if "name" in s]
