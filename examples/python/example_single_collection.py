#!/usr/bin/env python3
"""Option B example — single GoodMemCollection, no store boilerplate.

Usage:
    GOODMEM_API_KEY=your-key \
    GOODMEM_BASE_URL=https://localhost:8080 \
    GOODMEM_VERIFY_SSL=false \
    python example_single_collection.py
"""

import asyncio
import os
from dataclasses import dataclass
from typing import Annotated

from semantic_kernel.data.vector import VectorStoreField, vectorstoremodel

from goodmem_semantic_kernel import GoodMemCollection


@vectorstoremodel
@dataclass
class Note:
    id: Annotated[str | None, VectorStoreField("key")] = None
    content: Annotated[str, VectorStoreField("data", type="str")] = ""
    source: Annotated[str | None, VectorStoreField("data")] = None


async def main() -> None:
    for var in ("GOODMEM_API_KEY", "GOODMEM_BASE_URL", "GOODMEM_VERIFY_SSL"):
        if not os.environ.get(var):
            raise SystemExit(f"Set {var} before running this script.")

    async with GoodMemCollection(record_type=Note, collection_name="my-notes") as coll:
        # 1. Delete then recreate — ensures a clean slate each run
        await coll.ensure_collection_deleted()
        await coll.ensure_collection_exists()
        print("Collection ready (fresh).")

        # 2. Write a couple of notes
        while True:
            try:
                num_facts = int(input("How many notes do you want to add? (1-5): "))
                if 1 <= num_facts <= 5:
                    break
                print("Please enter a number between 1 and 5.")
            except ValueError:
                print("Please enter a valid number.")

        notes = []
        for i in range(1, num_facts + 1):
            content = input(f"note {i} content: ")
            source = input(f"note {i} source: ")
            notes.append(Note(content=content, source=source))

        keys = await coll.upsert(notes)
        print(f"Upserted {len(keys)} notes: {keys}")

        # 3. Wait for server-side embedding pipeline
        print("Waiting for embeddings...")
        await asyncio.sleep(3)

        # 4. Semantic search
        query = input("Enter a search query: ")
        print(f"\nSearching for: '{query}'")
        results = await coll.search(query, top=5)
        async for r in results.results:
            print(f"[{r.score:.3f}] {r.record.content!r} (source={r.record.source})")

        # 5. Uncomment to clean up after inspecting:
        # await coll.ensure_collection_deleted()
        # print("\nCollection deleted.")


if __name__ == "__main__":
    asyncio.run(main())
