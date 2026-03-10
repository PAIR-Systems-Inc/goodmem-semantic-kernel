#!/usr/bin/env python3
"""Option C example — GoodMemStore managing multiple collections.

A single store owns one HTTP connection shared across all collections.

Usage:
    GOODMEM_API_KEY=your-key \
    GOODMEM_BASE_URL=https://localhost:8080 \
    GOODMEM_VERIFY_SSL=false \
    python example_store.py
"""

import asyncio
import os
from dataclasses import dataclass
from typing import Annotated

from semantic_kernel.data.vector import VectorStoreField, vectorstoremodel

from goodmem_semantic_kernel import GoodMemStore


@vectorstoremodel
@dataclass
class Note:
    id: Annotated[str | None, VectorStoreField("key")] = None
    content: Annotated[str, VectorStoreField("data", type="str")] = ""
    source: Annotated[str | None, VectorStoreField("data")] = None


async def main() -> None:
    for var in ("GOODMEM_API_KEY", "GOODMEM_BASE_URL", "GOODMEM_VERIFY_SSL"):
        if not os.getenv(var):
            raise SystemExit(f"Set {var} before running this script.")

    async with GoodMemStore() as store:
        # 1. List all spaces currently visible to this API key
        names = await store.list_collection_names()
        print(f"Existing spaces: {names or '(none)'}")

        # 2. Get two collections from the same store (shared HTTP connection)
        notes = store.get_collection(Note, collection_name="store-notes")
        todos = store.get_collection(Note, collection_name="store-todos")

        # 3. Fresh slate for both
        for collection in (notes, todos):
            await collection.ensure_collection_deleted()
            await collection.ensure_collection_exists()
        print("Both collections ready (fresh).")

        # 4. Write into each
        await notes.upsert([
            Note(content="The Eiffel Tower is in Paris", source="facts"),
            Note(content="Mount Fuji is in Japan", source="facts"),
        ])
        await todos.upsert([
            Note(content="Buy groceries", source="chat"),
            Note(content="Call the dentist", source="chat"),
        ])
        print("Upserted into both collections.")

        # 5. Wait for embeddings
        print("Waiting for embeddings...")
        await asyncio.sleep(3)

        # 6. Search each independently
        print("\n--- notes search: 'famous landmarks in europe' ---")
        results = await notes.search("famous landmarks in europe", top=3)
        async for r in results.results:
            print(f"  [{r.score:.3f}] {r.record.content!r}")

        print("\n--- todos search: 'health appointments' ---")
        results = await todos.search("health appointments", top=3)
        async for r in results.results:
            print(f"  [{r.score:.3f}] {r.record.content!r}")

        # 7. Uncomment to clean up:
        # for collection in (notes, todos):
        #     await collection.ensure_collection_deleted()
        # print("\nCollections deleted.")


if __name__ == "__main__":
    asyncio.run(main())
