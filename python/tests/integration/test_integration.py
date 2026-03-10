"""Integration tests for GoodMem SK connector.

These tests require a running GoodMem server.  They are skipped by default
and only run when:

1. The ``GOODMEM_API_KEY`` environment variable is set.
2. The ``--run-integration`` pytest flag is passed  (or ``-m integration``).

To run:

    GOODMEM_API_KEY=your-key pytest -m integration tests/integration/

The server URL defaults to ``http://localhost:8080`` but can be overridden via
``GOODMEM_BASE_URL``.
"""

import asyncio
import os
import uuid
from dataclasses import dataclass
from typing import Annotated

import pytest

from goodmem_semantic_kernel.collection import GoodMemCollection
from goodmem_semantic_kernel.settings import GoodMemSettings
from goodmem_semantic_kernel.store import GoodMemStore
from semantic_kernel.data.vector import VectorStoreField, vectorstoremodel

# ---------------------------------------------------------------------------
# Skip marker — only run when API key is available
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.integration

_SKIP_REASON = (
    "Integration tests require GOODMEM_API_KEY to be set.  "
    "Run with: GOODMEM_API_KEY=your-key pytest -m integration"
)


def _integration_settings() -> GoodMemSettings | None:
    api_key = os.getenv("GOODMEM_API_KEY")
    if not api_key:
        return None
    return GoodMemSettings(
        base_url=os.getenv("GOODMEM_BASE_URL", "http://localhost:8080"),
        api_key=api_key,
    )


# ---------------------------------------------------------------------------
# Test data model
# ---------------------------------------------------------------------------


@vectorstoremodel
@dataclass
class Fact:
    """Simple record for integration tests."""

    id: Annotated[str | None, VectorStoreField("key")] = None
    content: Annotated[str, VectorStoreField("data", type="str")] = ""
    category: Annotated[str | None, VectorStoreField("data")] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def unique_space_name() -> str:
    return f"sk-integration-test-{uuid.uuid4().hex[:8]}"


async def _wait_for_results(
    coll: GoodMemCollection,
    query: str,
    *,
    top: int = 5,
    timeout: float = 30.0,
    poll_interval: float = 1.5,
) -> list:
    """Poll search until at least one result appears or timeout is reached.

    GoodMem's embedding pipeline is async — content is searchable only after
    the server-side embedding job completes, which depends on the embedding
    provider's latency.  A fixed sleep is unreliable; polling is not.
    """
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        results = await coll.search(query, top=top)
        items = [r async for r in results.results]
        if items:
            return items
        await asyncio.sleep(poll_interval)
    return []


# ---------------------------------------------------------------------------
# Full round-trip test
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("GOODMEM_API_KEY"),
    reason=_SKIP_REASON,
)
async def test_full_round_trip():
    """Create collection → upsert → search → delete → ensure_collection_deleted."""
    settings = _integration_settings()
    assert settings is not None

    space_name = unique_space_name()

    async with GoodMemStore(settings=settings) as store:
        coll = store.get_collection(Fact, collection_name=space_name)

        # 1. Create collection
        await coll.ensure_collection_exists()
        assert await coll.collection_exists()

        # 2. Upsert two records
        keys = await coll.upsert(
            [
                Fact(content="The sky is blue", category="nature"),
                Fact(content="Bananas are yellow", category="food"),
            ]
        )
        assert len(keys) == 2
        assert all(k for k in keys)

        # 3. Search — poll until the embedding pipeline finishes
        items = await _wait_for_results(coll, "sky colour", top=5)
        assert len(items) > 0, "Search returned no results after waiting for embeddings"
        contents = [item.record.content for item in items]
        assert any("sky" in c.lower() for c in contents), (
            f"Expected sky-related result in: {contents}"
        )

        # 4. Delete one record
        key_to_delete = keys[0]
        await coll.delete(keys=[key_to_delete])

        # Verify it's gone from batchGet
        await asyncio.sleep(1)
        fetched = await coll.get(keys=keys)
        if fetched is not None:
            fetched_ids = [f.id for f in fetched] if isinstance(fetched, list) else [fetched.id]
            assert key_to_delete not in fetched_ids

        # 5. Delete the collection
        await coll.ensure_collection_deleted()
        assert not await coll.collection_exists()


@pytest.mark.skipif(
    not os.environ.get("GOODMEM_API_KEY"),
    reason=_SKIP_REASON,
)
async def test_upsert_then_search_returns_inserted_content():
    """Verify that recently upserted content is discoverable via search."""
    settings = _integration_settings()
    assert settings is not None

    space_name = unique_space_name()
    unique_phrase = f"xyzzy-{uuid.uuid4().hex[:6]}"

    async with GoodMemStore(settings=settings) as store:
        coll = store.get_collection(Fact, collection_name=space_name)
        try:
            await coll.ensure_collection_exists()
            await coll.upsert(Fact(content=f"Unique phrase: {unique_phrase}", category="test"))

            # Poll until the embedding pipeline finishes
            items = await _wait_for_results(coll, unique_phrase, top=3)
            assert len(items) > 0, "Search returned no results after waiting for embeddings"
            assert any(unique_phrase in (item.record.content or "") for item in items), (
                f"Did not find '{unique_phrase}' in results: "
                f"{[i.record.content for i in items]}"
            )
        finally:
            await coll.ensure_collection_deleted()


@pytest.mark.skipif(
    not os.environ.get("GOODMEM_API_KEY"),
    reason=_SKIP_REASON,
)
async def test_delete_removes_record_from_search_results():
    """Verify that deleting a record removes it from subsequent searches."""
    settings = _integration_settings()
    assert settings is not None

    space_name = unique_space_name()
    unique_phrase = f"deletable-{uuid.uuid4().hex[:6]}"

    async with GoodMemStore(settings=settings) as store:
        coll = store.get_collection(Fact, collection_name=space_name)
        try:
            await coll.ensure_collection_exists()
            keys = await coll.upsert(
                Fact(content=f"This will be deleted: {unique_phrase}")
            )
            target_key = keys if isinstance(keys, str) else keys[0]

            # Confirm it appears in search — poll until embedded
            pre_items = await _wait_for_results(coll, unique_phrase, top=5)
            assert any(unique_phrase in (i.record.content or "") for i in pre_items), (
                "Record did not appear in search after waiting for embeddings"
            )

            # Delete it
            await coll.delete(keys=[target_key])
            await asyncio.sleep(2)

            # Confirm it no longer appears
            post_delete = await coll.search(unique_phrase, top=5)
            post_items = [r async for r in post_delete.results]
            assert not any(
                unique_phrase in (i.record.content or "") for i in post_items
            ), f"Deleted record still appears in search: {[i.record.content for i in post_items]}"
        finally:
            await coll.ensure_collection_deleted()
