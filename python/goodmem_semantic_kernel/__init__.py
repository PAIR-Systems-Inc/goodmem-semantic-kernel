"""GoodMem Semantic Kernel integration.

Provides a drop-in :class:`~goodmem_semantic_kernel.collection.GoodMemCollection` and
:class:`~goodmem_semantic_kernel.store.GoodMemStore` that back Semantic Kernel's vector
store abstractions with the GoodMem memory API.

Quick start::

    from dataclasses import dataclass
    from goodmem_semantic_kernel import GoodMemCollection, GoodMemStore, GoodMemSettings
    from semantic_kernel.data.vector import vectorstoremodel, VectorStoreField
    from typing import Annotated

    @vectorstoremodel
    @dataclass
    class Note:
        id: Annotated[str | None, VectorStoreField("key")] = None
        content: Annotated[str, VectorStoreField("data", type="str")] = ""

    async with GoodMemStore() as store:
        coll = store.get_collection(Note, collection_name="notes")
        await coll.ensure_collection_exists()
        keys = await coll.upsert(Note(content="Remember to buy milk"))
        results = await coll.search("grocery list")
        async for r in results.results:
            print(r.record.content, r.score)
"""

from goodmem_semantic_kernel._client import GoodMemAsyncClient
from goodmem_semantic_kernel.collection import GoodMemCollection
from goodmem_semantic_kernel.settings import GoodMemSettings
from goodmem_semantic_kernel.store import GoodMemStore

__all__ = [
    "GoodMemAsyncClient",
    "GoodMemCollection",
    "GoodMemSettings",
    "GoodMemStore",
]
