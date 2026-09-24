"""Live tests against a running GoodMem server.

Skipped unless GOODMEM_BASE_URL, GOODMEM_API_KEY and GOODMEM_EMBEDDER_ID are
set. Every space these create is deleted again, and the teardown is verified.
"""

import os
import uuid

import pytest
from semantic_kernel.exceptions.vector_store_exceptions import (
    VectorStoreInitializationException,
    VectorStoreOperationException,
)
from support import Note

from goodmem_semantic_kernel import GoodMemCollection, GoodMemSettings, GoodMemStore

LIVE = all(
    os.environ.get(name) for name in ("GOODMEM_BASE_URL", "GOODMEM_API_KEY", "GOODMEM_EMBEDDER_ID")
)
pytestmark = pytest.mark.skipif(
    not LIVE, reason="set GOODMEM_BASE_URL, GOODMEM_API_KEY and GOODMEM_EMBEDDER_ID"
)


def live_settings(**overrides):
    values = {
        "base_url": os.environ["GOODMEM_BASE_URL"],
        "api_key": os.environ["GOODMEM_API_KEY"],
        "embedder_id": os.environ["GOODMEM_EMBEDDER_ID"],
        "verify_ssl": os.environ.get("GOODMEM_VERIFY_SSL", "true").lower() != "false",
    }
    values.update(overrides)
    return GoodMemSettings(**values)


@pytest.fixture
async def collection():
    """A collection on its own space, deleted afterwards."""
    name = f"sk-live-{uuid.uuid4().hex[:8]}"
    coll = GoodMemCollection(record_type=Note, collection_name=name, settings=live_settings())
    await coll.ensure_collection_exists()
    try:
        yield coll
    finally:
        await coll.ensure_collection_deleted()
        assert not await coll.collection_exists(), "teardown left the space behind"
        await coll.__aexit__(None, None, None)


async def test_round_trip_write_read_search_delete(collection):
    key = str(uuid.uuid4())
    await collection.upsert(
        Note(id=key, content="Ada Lovelace wrote the first algorithm.", tag="history", year=1843)
    )

    records = await collection.get(keys=[key])
    assert records is not None, "a written record must be readable back"
    assert records[0].content.strip() == "Ada Lovelace wrote the first algorithm."
    assert records[0].tag == "history"

    results = await collection.search("who wrote the first algorithm", top=3)
    hits = [r async for r in results.results]
    assert hits, "the record must be searchable"
    assert "Ada Lovelace" in hits[0].record.content
    assert results.metadata["goodmem_partial"] is False

    await collection.delete([key])
    assert await collection.get(keys=[key]) is None


async def test_scores_are_higher_is_better(collection):
    await collection.upsert(
        [
            Note(content="Ada Lovelace wrote the first algorithm.", tag="a"),
            Note(content="A recipe for sourdough bread.", tag="b"),
        ]
    )
    results = await collection.search("who wrote the first algorithm", top=2)
    hits = [r async for r in results.results]

    assert len(hits) == 2
    assert hits[0].score > hits[1].score, "the best match must score highest"
    assert "Ada" in hits[0].record.content


async def test_a_failed_update_leaves_the_previous_version_in_place(collection):
    """The server rejects an empty originalContent, so this create fails after
    the delete. The record must survive."""
    key = str(uuid.uuid4())
    await collection.upsert(Note(id=key, content="Payroll runs on the 25th.", tag="hr"))

    with pytest.raises(VectorStoreOperationException):
        await collection.upsert(Note(id=key, content="", tag="hr"))

    records = await collection.get(keys=[key])
    assert records is not None, "a failed update must not destroy the record"
    assert records[0].content.strip() == "Payroll runs on the 25th."


async def test_a_successful_update_replaces_the_content(collection):
    key = str(uuid.uuid4())
    await collection.upsert(Note(id=key, content="First version.", tag="v1"))
    await collection.upsert(Note(id=key, content="Second version.", tag="v2"))

    records = await collection.get(keys=[key])
    assert records[0].content.strip() == "Second version."
    assert records[0].tag == "v2"


async def test_a_filtered_search_runs_server_side(collection):
    await collection.upsert(
        [
            Note(content="The quarterly report is ready.", tag="finance", year=2026),
            Note(content="The quarterly review is ready.", tag="hr", year=2026),
        ]
    )

    results = await collection.search("quarterly", filter=lambda n: n.tag == "finance", top=5)
    hits = [r async for r in results.results]

    assert hits, "the filter must not remove everything"
    assert all(h.record.tag == "finance" for h in hits)


async def test_a_numeric_filter_uses_the_numeric_cast(collection):
    await collection.upsert(
        [
            Note(content="Old minutes.", tag="notes", year=1999),
            Note(content="New minutes.", tag="notes", year=2026),
        ]
    )

    results = await collection.search("minutes", filter=lambda n: n.year > 2000, top=5)
    hits = [r async for r in results.results]

    assert [h.record.year for h in hits] == [2026]


async def test_a_boolean_filter_uses_the_boolean_cast(collection):
    """Comparing a boolean as text is accepted by the server and matches
    nothing, so this is the test that catches a stringified bool."""
    await collection.upsert(
        [
            Note(content="An active project.", tag="p", active=True),
            Note(content="A finished project.", tag="p", active=False),
        ]
    )

    results = await collection.search("project", filter=lambda n: n.active == True, top=5)  # noqa: E712
    hits = [r async for r in results.results]

    assert [h.record.content.strip() for h in hits] == ["An active project."]


async def test_an_apostrophe_in_a_filter_value_is_a_value(collection):
    await collection.upsert(
        [
            Note(content="Written by O'Brien.", tag="o'brien"),
            Note(content="Written by Smith.", tag="smith"),
        ]
    )

    results = await collection.search("written by", filter=lambda n: n.tag == "o'brien", top=5)
    hits = [r async for r in results.results]

    assert [h.record.tag for h in hits] == ["o'brien"]


async def test_an_injection_payload_matches_only_its_own_row(collection):
    # Written as a literal: Semantic Kernel parses the lambda's source, so a
    # variable inside it cannot be resolved to a value.
    await collection.upsert(
        [
            Note(content="The injected row.", tag="x' OR '1'='1"),
            Note(content="An innocent row.", tag="normal"),
            Note(content="Another innocent row.", tag="also-normal"),
        ]
    )

    # The call stays on one line: Semantic Kernel re-parses the lambda's own
    # source line, and a fragment of a wrapped call is not valid Python.
    results = await collection.search("row", top=10, filter=lambda n: n.tag == "x' OR '1'='1")
    hits = [r async for r in results.results]

    assert len(hits) == 1, "an injection payload must not widen the match"
    assert hits[0].record.tag == "x' OR '1'='1"


async def test_reusing_a_space_with_a_different_embedder_is_refused(collection):
    """The embedder is permanent for a space; writing into a mismatched one
    would index the data differently than the caller asked for."""
    other = GoodMemCollection(
        record_type=Note,
        collection_name=collection.collection_name,
        settings=live_settings(embedder_id="019cfd94-2844-7117-85ca-1b9919758a26"),
    )
    try:
        with pytest.raises(VectorStoreInitializationException, match="cannot be changed"):
            await other.ensure_collection_exists()
    finally:
        await other.__aexit__(None, None, None)


async def test_upsert_waits_so_a_search_can_find_what_was_just_written(collection):
    """GoodMem indexes asynchronously; without the wait this race is real."""
    marker = uuid.uuid4().hex
    await collection.upsert(Note(content=f"The access code is {marker}.", tag="secret"))

    results = await collection.search(f"access code {marker}", top=5)
    hits = [r async for r in results.results]

    assert any(marker in h.record.content for h in hits)


async def test_the_store_lists_and_shares_one_client(collection):
    async with GoodMemStore(settings=live_settings()) as store:
        names = await store.list_collection_names()
        assert collection.collection_name in names

        shared = store.get_collection(Note, collection_name=collection.collection_name)
        assert shared.managed_client is False
        await shared.upsert(Note(content="Written through the store.", tag="store"))
        results = await shared.search("written through", top=3)
        assert [r async for r in results.results]


async def test_a_rejected_write_reports_the_servers_own_message(collection):
    with pytest.raises(VectorStoreOperationException) as caught:
        await collection.upsert(Note(content=""))

    assert "originalContent" in str(caught.value)
