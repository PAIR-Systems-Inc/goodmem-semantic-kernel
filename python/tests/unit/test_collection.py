"""Unit tests for GoodMemCollection (mock httpx)."""

from dataclasses import dataclass
from typing import Annotated
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from goodmem_semantic_kernel._client import GoodMemAsyncClient
from goodmem_semantic_kernel.collection import GoodMemCollection
from goodmem_semantic_kernel.settings import GoodMemSettings
from semantic_kernel.data.vector import VectorStoreField, vectorstoremodel
from semantic_kernel.exceptions.vector_store_exceptions import (
    VectorStoreOperationNotSupportedException,
)


# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------


@vectorstoremodel
@dataclass
class Note:
    id: Annotated[str | None, VectorStoreField("key")] = None
    content: Annotated[str, VectorStoreField("data", type="str")] = ""
    tag: Annotated[str | None, VectorStoreField("data")] = None


def make_settings(**overrides) -> GoodMemSettings:
    defaults = {
        "base_url": "http://localhost:8080",
        "api_key": "test-key",
        "embedder_id": "embedder-001",
    }
    defaults.update(overrides)
    return GoodMemSettings(**defaults)


def make_mock_client() -> GoodMemAsyncClient:
    """Return an AsyncMock that satisfies the GoodMemAsyncClient interface."""
    client = MagicMock(spec=GoodMemAsyncClient)
    client.list_spaces = AsyncMock(return_value=[])
    client.create_space = AsyncMock(return_value={"spaceId": "space-001"})
    client.delete_space = AsyncMock(return_value=None)
    client.list_embedders = AsyncMock(return_value=[{"embedderId": "embedder-001"}])
    client.create_memory = AsyncMock(return_value={"memoryId": "mem-001"})
    client.get_memories_batch = AsyncMock(return_value=[])
    client.delete_memory = AsyncMock(return_value=None)
    client.retrieve_memories = AsyncMock(return_value=[])
    client.aclose = AsyncMock(return_value=None)
    return client


def make_collection(
    client: GoodMemAsyncClient | None = None,
    collection_name: str = "test-space",
    **settings_overrides,
) -> GoodMemCollection:
    c = client or make_mock_client()
    return GoodMemCollection(
        record_type=Note,
        collection_name=collection_name,
        settings=make_settings(**settings_overrides),
        client=c,
    )


# ---------------------------------------------------------------------------
# collection_exists
# ---------------------------------------------------------------------------


async def test_collection_exists_returns_true_when_space_found():
    client = make_mock_client()
    client.list_spaces = AsyncMock(return_value=[{"name": "test-space", "spaceId": "s1"}])

    coll = make_collection(client)
    assert await coll.collection_exists() is True


async def test_collection_exists_returns_false_when_not_found():
    client = make_mock_client()
    client.list_spaces = AsyncMock(return_value=[{"name": "other-space", "spaceId": "s2"}])

    coll = make_collection(client)
    assert await coll.collection_exists() is False


# ---------------------------------------------------------------------------
# ensure_collection_exists
# ---------------------------------------------------------------------------


async def test_ensure_collection_exists_creates_space_when_missing():
    client = make_mock_client()
    client.list_spaces = AsyncMock(return_value=[])  # space not found
    client.create_space = AsyncMock(return_value={"spaceId": "new-space-id"})

    coll = make_collection(client)
    await coll.ensure_collection_exists()

    client.create_space.assert_awaited_once_with(
        name="test-space",
        embedder_id="embedder-001",
    )


async def test_ensure_collection_exists_skips_when_present():
    client = make_mock_client()
    client.list_spaces = AsyncMock(return_value=[{"name": "test-space", "spaceId": "s1"}])

    coll = make_collection(client)
    await coll.ensure_collection_exists()

    client.create_space.assert_not_awaited()


# ---------------------------------------------------------------------------
# upsert
# ---------------------------------------------------------------------------


async def test_upsert_single_record_without_key_returns_server_generated_id():
    client = make_mock_client()
    client.list_spaces = AsyncMock(return_value=[{"name": "test-space", "spaceId": "s1"}])
    client.create_memory = AsyncMock(return_value={"memoryId": "server-generated-id"})

    coll = make_collection(client)
    key = await coll.upsert(Note(content="Hello, world!"))

    assert key == "server-generated-id"
    client.create_memory.assert_awaited_once()
    call_kwargs = client.create_memory.call_args
    assert call_kwargs.kwargs["content"] == "Hello, world!"
    assert call_kwargs.kwargs["space_id"] == "s1"


async def test_upsert_single_record_with_key_does_delete_then_create():
    client = make_mock_client()
    client.list_spaces = AsyncMock(return_value=[{"name": "test-space", "spaceId": "s1"}])
    client.create_memory = AsyncMock(return_value={"memoryId": "my-id"})

    coll = make_collection(client)
    key = await coll.upsert(Note(id="my-id", content="Updated content"))

    assert key == "my-id"
    client.delete_memory.assert_awaited_once_with("my-id")
    client.create_memory.assert_awaited_once()


async def test_upsert_batch_returns_list_of_memory_ids():
    client = make_mock_client()
    client.list_spaces = AsyncMock(return_value=[{"name": "test-space", "spaceId": "s1"}])
    client.create_memory = AsyncMock(
        side_effect=[
            {"memoryId": "id-1"},
            {"memoryId": "id-2"},
        ]
    )

    coll = make_collection(client)
    keys = await coll.upsert([Note(content="First"), Note(content="Second")])

    assert keys == ["id-1", "id-2"]
    assert client.create_memory.await_count == 2


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------


async def test_get_by_key_calls_batch_get_and_deserializes():
    client = make_mock_client()
    client.list_spaces = AsyncMock(return_value=[{"name": "test-space", "spaceId": "s1"}])
    client.get_memories_batch = AsyncMock(
        return_value=[
            {
                "memoryId": "mem-42",
                "originalContent": "Test note content",
                "contentType": "text/plain",
                "metadata": {"tag": "important"},
            }
        ]
    )

    coll = make_collection(client)
    result = await coll.get(key="mem-42")

    assert result is not None
    assert result.id == "mem-42"
    assert result.content == "Test note content"
    assert result.tag == "important"
    client.get_memories_batch.assert_awaited_once_with(["mem-42"])


async def test_get_returns_none_when_key_not_found():
    client = make_mock_client()
    client.get_memories_batch = AsyncMock(return_value=[])

    coll = make_collection(client)
    result = await coll.get(key="nonexistent")

    assert result is None


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


async def test_delete_calls_delete_for_each_key():
    client = make_mock_client()

    coll = make_collection(client)
    await coll.delete(keys=["key-1", "key-2", "key-3"])

    assert client.delete_memory.await_count == 3
    client.delete_memory.assert_any_await("key-1")
    client.delete_memory.assert_any_await("key-2")
    client.delete_memory.assert_any_await("key-3")


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


async def test_search_text_returns_kernel_search_results_with_correct_scores():
    client = make_mock_client()
    client.list_spaces = AsyncMock(return_value=[{"name": "test-space", "spaceId": "s1"}])
    # retrieve_memories returns the correlated list that _client.py produces after
    # parsing the NDJSON stream.  Wire format:
    #   Top-level {"memoryDefinition": Memory} events → memory_list (indexed by position)
    #   {"retrievedItem": {"chunk": ChunkRef}} where ChunkRef.memoryIndex → memory_list[i]
    # _client.py negates relevanceScore so "score" here is already higher-is-better.
    client.retrieve_memories = AsyncMock(
        return_value=[
            {
                "chunk": {"memoryId": "m1", "chunkText": "Hello world"},
                "memory": {"memoryId": "m1", "originalContent": None, "metadata": {}},
                "score": 0.92,
            },
            {
                "chunk": {"memoryId": "m2", "chunkText": "Greetings"},
                "memory": {"memoryId": "m2", "originalContent": None, "metadata": {"tag": "greeting"}},
                "score": 0.78,
            },
        ]
    )

    coll = make_collection(client)
    results = await coll.search("hello", top=5)

    assert results.total_count == 2

    items = [r async for r in results.results]
    assert len(items) == 2
    assert items[0].score == pytest.approx(0.92)
    assert items[0].record.content == "Hello world"
    assert items[1].score == pytest.approx(0.78)
    assert items[1].record.tag == "greeting"


async def test_search_with_precomputed_vector_raises_not_supported():
    coll = make_collection()

    with pytest.raises(VectorStoreOperationNotSupportedException):
        await coll.search(vector=[0.1, 0.2, 0.3])


async def test_search_with_filter_raises_not_supported():
    coll = make_collection()

    with pytest.raises(VectorStoreOperationNotSupportedException):
        await coll.search("hello", filter=lambda x: x.tag == "test")


# ---------------------------------------------------------------------------
# Context manager closes client
# ---------------------------------------------------------------------------


async def test_context_manager_closes_owned_client():
    client = make_mock_client()
    # Owned client (no external client injected)
    coll = GoodMemCollection(
        record_type=Note,
        collection_name="test-space",
        settings=make_settings(),
    )
    # Swap in mock
    object.__setattr__(coll, "_http", client)
    object.__setattr__(coll, "managed_client", True)

    async with coll:
        pass

    client.aclose.assert_awaited_once()


async def test_context_manager_does_not_close_injected_client():
    client = make_mock_client()
    async with make_collection(client):
        pass

    # injected client → managed_client=False → should NOT be closed
    client.aclose.assert_not_awaited()
