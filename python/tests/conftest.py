"""Pytest fixtures; the helpers live in :mod:`support`."""

import pytest
from goodmem import AsyncGoodmem
from support import Note, Recorder, build_client, build_settings

from goodmem_semantic_kernel import GoodMemCollection, GoodMemSettings


@pytest.fixture
def recorder() -> Recorder:
    return Recorder()


@pytest.fixture
def sdk_client(recorder: Recorder) -> AsyncGoodmem:
    return build_client(recorder)


@pytest.fixture
def settings() -> GoodMemSettings:
    return build_settings()


@pytest.fixture
def collection(sdk_client: AsyncGoodmem, settings: GoodMemSettings) -> GoodMemCollection:
    return GoodMemCollection(
        record_type=Note,
        collection_name="notes",
        settings=settings,
        client=sdk_client,
    )
