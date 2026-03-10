"""Unit tests for GoodMemStore (mock httpx)."""

from dataclasses import dataclass
from typing import Annotated
from unittest.mock import AsyncMock, MagicMock

import pytest

from goodmem_semantic_kernel._client import GoodMemAsyncClient
from goodmem_semantic_kernel.collection import GoodMemCollection
from goodmem_semantic_kernel.settings import GoodMemSettings
from goodmem_semantic_kernel.store import GoodMemStore
from semantic_kernel.data.vector import VectorStoreField, vectorstoremodel


# ---------------------------------------------------------------------------
# Test data model
# ---------------------------------------------------------------------------


@vectorstoremodel
@dataclass
class Item:
    id: Annotated[str | None, VectorStoreField("key")] = None
    content: Annotated[str, VectorStoreField("data", type="str")] = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_settings(**overrides) -> GoodMemSettings:
    defaults = {
        "base_url": "http://localhost:8080",
        "api_key": "test-key",
        "embedder_id": "embedder-001",
    }
    defaults.update(overrides)
    return GoodMemSettings(**defaults)


def make_mock_client() -> GoodMemAsyncClient:
    client = MagicMock(spec=GoodMemAsyncClient)
    client.list_spaces = AsyncMock(return_value=[])
    client.aclose = AsyncMock(return_value=None)
    return client


# ---------------------------------------------------------------------------
# list_collection_names
# ---------------------------------------------------------------------------


async def test_list_collection_names_returns_space_names():
    client = make_mock_client()
    client.list_spaces = AsyncMock(
        return_value=[
            {"spaceId": "s1", "name": "alpha"},
            {"spaceId": "s2", "name": "beta"},
            {"spaceId": "s3", "name": "gamma"},
        ]
    )

    store = GoodMemStore(settings=make_settings(), client=client)
    names = await store.list_collection_names()

    assert list(names) == ["alpha", "beta", "gamma"]


async def test_list_collection_names_empty_when_no_spaces():
    client = make_mock_client()
    client.list_spaces = AsyncMock(return_value=[])

    store = GoodMemStore(settings=make_settings(), client=client)
    names = await store.list_collection_names()

    assert list(names) == []


# ---------------------------------------------------------------------------
# get_collection
# ---------------------------------------------------------------------------


async def test_get_collection_returns_goodmem_collection():
    client = make_mock_client()
    store = GoodMemStore(settings=make_settings(), client=client)

    coll = store.get_collection(Item, collection_name="my-space")

    assert isinstance(coll, GoodMemCollection)
    assert coll.collection_name == "my-space"


async def test_get_collection_shares_client_with_store():
    client = make_mock_client()
    store = GoodMemStore(settings=make_settings(), client=client)

    coll = store.get_collection(Item, collection_name="my-space")

    # The collection should use the same HTTP client as the store
    assert coll._http is client


async def test_get_collection_injected_client_not_managed():
    """Collection obtained from store must not close the shared client on exit."""
    client = make_mock_client()
    store = GoodMemStore(settings=make_settings(), client=client)

    coll = store.get_collection(Item, collection_name="my-space")
    async with coll:
        pass

    client.aclose.assert_not_awaited()


# ---------------------------------------------------------------------------
# Context manager lifecycle
# ---------------------------------------------------------------------------


async def test_store_context_manager_closes_owned_client():
    """Store closes its own client when used as a context manager."""
    client = make_mock_client()
    store = GoodMemStore(settings=make_settings())
    object.__setattr__(store, "_http", client)
    object.__setattr__(store, "managed_client", True)

    async with store:
        pass

    client.aclose.assert_awaited_once()


async def test_store_does_not_close_injected_client():
    """Store must not close an externally-provided client."""
    client = make_mock_client()
    store = GoodMemStore(settings=make_settings(), client=client)

    async with store:
        pass

    client.aclose.assert_not_awaited()
