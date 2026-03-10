"""Shared fixtures for GoodMem SK tests."""

from dataclasses import dataclass
from typing import Annotated

import pytest

from goodmem_semantic_kernel._client import GoodMemAsyncClient
from goodmem_semantic_kernel.settings import GoodMemSettings

# ---------------------------------------------------------------------------
# Shared data model used by unit tests
# ---------------------------------------------------------------------------

# Import these lazily to avoid import errors if SK is not installed yet
try:
    from semantic_kernel.data.vector import VectorStoreField, vectorstoremodel

    @vectorstoremodel
    @dataclass
    class NoteModel:
        """Simple test record type."""

        id: Annotated[str | None, VectorStoreField("key")] = None
        content: Annotated[str, VectorStoreField("data", type="str")] = ""
        tag: Annotated[str | None, VectorStoreField("data")] = None

except ImportError:
    NoteModel = None  # type: ignore[assignment,misc]


@pytest.fixture
def note_model():
    """Return the NoteModel class."""
    return NoteModel


@pytest.fixture
def goodmem_settings():
    """Return a GoodMemSettings instance with dummy values suitable for unit tests."""
    return GoodMemSettings(
        base_url="http://localhost:8080",
        api_key="test-api-key",
        embedder_id="embedder-uuid-001",
    )
