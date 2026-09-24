"""Regressions for every defect the 0.2.0 audit reproduced.

Each test names the pattern from the master checklist and fails against 0.2.0.
"""

from __future__ import annotations

import httpx
import pytest
from semantic_kernel.exceptions.vector_store_exceptions import (
    VectorStoreInitializationException,
    VectorStoreOperationException,
    VectorStoreOperationNotSupportedException,
)
from support import (
    EMBEDDER_ID,
    REAL_VECTOR_SCORE,
    Note,
    chunk_event,
    memory_event,
    memory_json,
    ndjson,
    space_json,
    status_event,
)

from goodmem_semantic_kernel import GoodMemCollection, GoodMemSettings, GoodMemUpsertError
from goodmem_semantic_kernel import filters as gm_filters

# asyncio_mode = "auto" in pyproject.toml runs the async tests; the few sync
# ones here need no marker.


def spaces_listing(*names: str, embedder_id: str = EMBEDDER_ID) -> httpx.Response:
    return httpx.Response(
        200, json={"spaces": [space_json(n, embedder_id=embedder_id) for n in names]}
    )


# ---------------------------------------------------------------------------
# P18 — the batch response envelope
# ---------------------------------------------------------------------------


async def test_get_reads_the_results_envelope(collection, recorder):
    """0.2.0 read ``memories`` from a payload that has ``results``, so get()
    returned nothing for records that existed."""
    recorder.route("GET", "/v1/spaces", spaces_listing("notes"))
    recorder.route(
        "POST",
        "/v1/memories:batchGet",
        httpx.Response(
            200,
            json={
                "results": [
                    {
                        "success": True,
                        "memory": memory_json("m-1", content="hello", metadata={"tag": "a"}),
                    }
                ]
            },
        ),
    )

    records = await collection.get(keys=["m-1"])

    assert records is not None, "an existing record must be readable"
    assert records[0].content == "hello"
    assert records[0].tag == "a"


async def test_get_asks_for_the_content(collection, recorder):
    """batchGet omits content unless asked, so the record would come back empty."""
    recorder.route("GET", "/v1/spaces", spaces_listing("notes"))
    recorder.route(
        "POST",
        "/v1/memories:batchGet",
        httpx.Response(
            200, json={"results": [{"success": True, "memory": memory_json("m-1", content="x")}]}
        ),
    )

    await collection.get(keys=["m-1"])

    body = recorder.bodies_for("POST", "/v1/memories:batchGet")[0]
    assert body.get("includeContent") is True


async def test_get_returns_records_in_the_requested_order(collection, recorder):
    recorder.route("GET", "/v1/spaces", spaces_listing("notes"))
    recorder.route(
        "POST",
        "/v1/memories:batchGet",
        httpx.Response(
            200,
            json={
                "results": [
                    {"success": True, "memory": memory_json("m-2", content="second")},
                    {"success": True, "memory": memory_json("m-1", content="first")},
                ]
            },
        ),
    )

    records = await collection.get(keys=["m-1", "m-2"])

    assert [r.content for r in records] == ["first", "second"]


async def test_get_skips_unsuccessful_results(collection, recorder):
    recorder.route("GET", "/v1/spaces", spaces_listing("notes"))
    recorder.route(
        "POST",
        "/v1/memories:batchGet",
        httpx.Response(
            200,
            json={
                "results": [
                    {"success": True, "memory": memory_json("m-1", content="kept")},
                    {"success": False, "errorMessage": "not found"},
                ]
            },
        ),
    )

    records = await collection.get(keys=["m-1", "m-missing"])

    assert [r.id for r in records] == ["m-1"]


# ---------------------------------------------------------------------------
# P9 — delete-then-create upsert
# ---------------------------------------------------------------------------


async def test_failed_update_restores_the_previous_version(collection, recorder):
    """0.2.0 deleted first and lost the record when the create failed."""
    recorder.route("GET", "/v1/spaces", spaces_listing("notes"))
    recorder.route(
        "GET",
        "/v1/memories/m-1",
        httpx.Response(200, json=memory_json("m-1", content="old", metadata={"tag": "a"})),
    )
    recorder.route(
        "GET",
        "/v1/memories/m-1/content",
        httpx.Response(200, content=b"old text", headers={"content-type": "text/plain"}),
    )
    recorder.route("DELETE", "/v1/memories/m-1", httpx.Response(204))

    creates: list[dict] = []

    def create(request: httpx.Request) -> httpx.Response:
        import json as _json

        body = _json.loads(request.content)
        creates.append(body)
        if len(creates) == 1:
            return httpx.Response(
                400, json={"errors": [{"field": "content", "message": "must be provided"}]}
            )
        return httpx.Response(201, json=memory_json(body["memoryId"], content="old text"))

    recorder.route("POST", "/v1/memories", create)

    with pytest.raises(VectorStoreOperationException) as caught:
        await collection.upsert(Note(id="m-1", content="", tag="a"))

    # Semantic Kernel wraps what a collection raises, so the detail is on the
    # cause. The message itself still says the record was restored.
    cause = caught.value.__cause__
    assert isinstance(cause, GoodMemUpsertError)
    assert cause.restored is True
    assert cause.lost_key is None
    assert "was restored" in str(caught.value)
    # The restore put the old content back under the same key.
    assert creates[1]["memoryId"] == "m-1"
    assert creates[1]["originalContent"] == "old text"


async def test_failed_update_that_cannot_be_restored_names_the_record(collection, recorder):
    recorder.route("GET", "/v1/spaces", spaces_listing("notes"))
    recorder.route(
        "GET", "/v1/memories/m-1", httpx.Response(200, json=memory_json("m-1", content="old"))
    )
    recorder.route("GET", "/v1/memories/m-1/content", httpx.Response(200, content=b"old text"))
    recorder.route("DELETE", "/v1/memories/m-1", httpx.Response(204))
    recorder.route("POST", "/v1/memories", httpx.Response(500, json={"error": "down"}))

    with pytest.raises(VectorStoreOperationException) as caught:
        await collection.upsert(Note(id="m-1", content="new"))

    cause = caught.value.__cause__
    assert isinstance(cause, GoodMemUpsertError)
    assert cause.restored is False
    assert cause.lost_key == "m-1"
    assert "could NOT be restored" in str(caught.value)


async def test_a_new_record_is_never_deleted_first(collection, recorder):
    """A key that does not exist yet must not trigger a delete."""
    recorder.route("GET", "/v1/spaces", spaces_listing("notes"))
    recorder.route("GET", "/v1/memories/m-new", httpx.Response(404, json={"error": "nope"}))
    recorder.route(
        "POST", "/v1/memories", httpx.Response(201, json=memory_json("m-new", content="hi"))
    )

    keys = await collection.upsert(Note(id="m-new", content="hi"))

    assert keys == "m-new"
    assert not [c for c in recorder.calls() if c[0] == "DELETE"]


async def test_upsert_failure_reports_the_keys_already_written(collection, recorder):
    recorder.route("GET", "/v1/spaces", spaces_listing("notes"))
    recorder.route("GET", "/v1/memories/m-2", httpx.Response(404, json={"error": "nope"}))

    calls = {"n": 0}

    def create(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(201, json=memory_json("m-1", content="first"))
        return httpx.Response(500, json={"error": "down"})

    recorder.route("POST", "/v1/memories", create)

    with pytest.raises(VectorStoreOperationException) as caught:
        await collection.upsert([Note(content="first"), Note(id="m-2", content="second")])

    assert caught.value.__cause__.written_keys == ["m-1"]


# ---------------------------------------------------------------------------
# P4 / P3 — retrieval statuses (the contract)
# ---------------------------------------------------------------------------


async def test_a_reported_problem_flags_the_results_but_keeps_them(collection, recorder):
    """Contract Q4a: problem plus hits → return the hits, flagged."""
    recorder.route("GET", "/v1/spaces", spaces_listing("notes"))
    recorder.route(
        "POST",
        "/v1/memories:retrieve",
        ndjson(
            status_event("RERANKING_FAILED", "reranker unavailable"),
            memory_event("m-1", {"tag": "a"}),
            chunk_event("c-1", "Ada wrote the first algorithm.", "m-1"),
        ),
    )

    results = await collection.search("algorithm")
    records = [r async for r in results.results]

    assert len(records) == 1
    assert results.metadata["goodmem_partial"] is True
    assert results.metadata["goodmem_statuses"][0]["code"] == "RERANKING_FAILED"


async def test_a_problem_with_no_hits_warns_and_returns_empty(collection, recorder):
    """Contract Q4b: never raise; return empty with a flag."""
    recorder.route("GET", "/v1/spaces", spaces_listing("notes"))
    recorder.route(
        "POST",
        "/v1/memories:retrieve",
        ndjson(status_event("VECTOR_SEARCH_FAILED", "index unavailable")),
    )

    with pytest.warns(UserWarning, match="no results"):
        results = await collection.search("algorithm")
        records = [r async for r in results.results]

    assert records == []
    assert results.metadata["goodmem_partial"] is True


async def test_an_unknown_status_code_is_surfaced_not_dropped(collection, recorder):
    """Contract Q3: a newer server's code becomes UNKNOWN, and keeps the hits."""
    recorder.route("GET", "/v1/spaces", spaces_listing("notes"))
    recorder.route(
        "POST",
        "/v1/memories:retrieve",
        ndjson(
            status_event("SOMETHING_NEW_IN_A_LATER_SERVER", "unrecognised"),
            memory_event("m-1"),
            chunk_event("c-1", "kept", "m-1"),
        ),
    )

    results = await collection.search("algorithm")
    records = [r async for r in results.results]

    assert len(records) == 1, "an unknown code must not discard results"
    assert results.metadata["goodmem_partial"] is True
    assert results.metadata["goodmem_statuses"][0]["code"] == "UNKNOWN"
    assert results.metadata["goodmem_statuses"][0]["unrecognized"] is True


async def test_feature_disabled_is_informational(collection, recorder):
    """Contract Q1: informational by code alone, with no details check."""
    recorder.route("GET", "/v1/spaces", spaces_listing("notes"))
    recorder.route(
        "POST",
        "/v1/memories:retrieve",
        ndjson(
            status_event("FEATURE_DISABLED", "no LLM configured"),
            memory_event("m-1"),
            chunk_event("c-1", "kept", "m-1"),
        ),
    )

    results = await collection.search("algorithm")
    records = [r async for r in results.results]

    assert len(records) == 1
    assert results.metadata["goodmem_partial"] is False
    assert "goodmem_statuses" not in results.metadata


async def test_a_truncated_stream_does_not_lose_the_events_before_it(collection, recorder):
    recorder.route("GET", "/v1/spaces", spaces_listing("notes"))
    good = ndjson(memory_event("m-1"), chunk_event("c-1", "kept", "m-1"))
    recorder.route(
        "POST",
        "/v1/memories:retrieve",
        httpx.Response(
            200,
            text=good.text + '\n{"retrievedItem": {"chunk": {"chu',
            headers={"content-type": "application/x-ndjson"},
        ),
    )

    results = await collection.search("algorithm")
    records = [r async for r in results.results]

    assert [r.record.content for r in records] == ["kept"]
    assert results.metadata["goodmem_partial"] is True
    assert results.metadata["goodmem_statuses"][-1]["code"] == "MALFORMED_STREAM"


# ---------------------------------------------------------------------------
# P30 — joining chunks to memories
# ---------------------------------------------------------------------------


async def test_chunks_join_to_memories_by_uuid_not_position(collection, recorder):
    """0.2.0 indexed memory definitions by arrival order, so a stream that
    emits them in a different order than ``memoryIndex`` attached the wrong
    metadata to a chunk."""
    recorder.route("GET", "/v1/spaces", spaces_listing("notes"))
    recorder.route(
        "POST",
        "/v1/memories:retrieve",
        ndjson(
            memory_event("m-A", {"tag": "alpha"}),
            memory_event("m-B", {"tag": "beta"}),
            # memoryIndex says 0 (m-A) but the chunk belongs to m-B.
            chunk_event("c-1", "text of B", "m-B", memory_index=0),
        ),
    )

    results = await collection.search("text")
    records = [r async for r in results.results]

    assert records[0].record.tag == "beta", "metadata must follow the chunk's memory UUID"


async def test_two_chunks_of_one_memory_are_two_results(collection, recorder):
    recorder.route("GET", "/v1/spaces", spaces_listing("notes"))
    recorder.route(
        "POST",
        "/v1/memories:retrieve",
        ndjson(
            memory_event("m-1", {"tag": "a"}),
            chunk_event("c-1", "first half", "m-1"),
            chunk_event("c-2", "second half", "m-1"),
        ),
    )

    results = await collection.search("half")
    records = [r async for r in results.results]

    assert [r.record.content for r in records] == ["first half", "second half"]


# ---------------------------------------------------------------------------
# P29 — score direction
# ---------------------------------------------------------------------------


async def test_vector_scores_are_negated_into_higher_is_better(collection, recorder):
    recorder.route("GET", "/v1/spaces", spaces_listing("notes"))
    recorder.route(
        "POST",
        "/v1/memories:retrieve",
        ndjson(memory_event("m-1"), chunk_event("c-1", "x", "m-1", score=REAL_VECTOR_SCORE)),
    )

    results = await collection.search("x")
    records = [r async for r in results.results]

    assert REAL_VECTOR_SCORE < 0
    assert records[0].score == pytest.approx(-REAL_VECTOR_SCORE)


async def test_reranker_scores_are_not_negated(sdk_client, recorder, settings):
    """A reranker score is already higher-is-better; negating it would invert
    the ranking. Reranker ranges are provider-dependent, not 0-1."""
    reranked = GoodMemCollection(
        record_type=Note,
        collection_name="notes",
        settings=settings.model_copy(update={"reranker_id": "r-1"}),
        client=sdk_client,
    )
    recorder.route("GET", "/v1/spaces", spaces_listing("notes"))
    recorder.route(
        "POST",
        "/v1/memories:retrieve",
        ndjson(memory_event("m-1"), chunk_event("c-1", "x", "m-1", score=0.42)),
    )

    results = await reranked.search("x")
    records = [r async for r in results.results]

    assert records[0].score == pytest.approx(0.42)
    body = recorder.bodies_for("POST", "/v1/memories:retrieve")[0]
    assert body["postProcessor"]["config"]["reranker_id"] == "r-1"


# ---------------------------------------------------------------------------
# P19 / P34 — metadata filters
# ---------------------------------------------------------------------------


async def test_a_filter_lambda_becomes_a_server_side_filter(collection, recorder):
    """0.2.0 raised NotSupported for any filter at all."""
    recorder.route("GET", "/v1/spaces", spaces_listing("notes"))
    recorder.route(
        "POST", "/v1/memories:retrieve", ndjson(memory_event("m-1"), chunk_event("c-1", "x", "m-1"))
    )

    await collection.search("x", filter=lambda n: n.tag == "ops")

    body = recorder.bodies_for("POST", "/v1/memories:retrieve")[0]
    assert body["spaceKeys"][0]["filter"] == "CAST(val('$.tag') AS TEXT) = 'ops'"


async def test_an_apostrophe_in_a_filter_value_is_escaped(collection, recorder):
    """Backslash escaping is what the server accepts; SQL-style '' is a 400."""
    recorder.route("GET", "/v1/spaces", spaces_listing("notes"))
    recorder.route("POST", "/v1/memories:retrieve", ndjson())

    await collection.search("x", filter=lambda n: n.tag == "o'brien")

    body = recorder.bodies_for("POST", "/v1/memories:retrieve")[0]
    assert body["spaceKeys"][0]["filter"] == "CAST(val('$.tag') AS TEXT) = 'o\\'brien'"


async def test_an_injection_payload_stays_a_value(collection, recorder):
    recorder.route("GET", "/v1/spaces", spaces_listing("notes"))
    recorder.route("POST", "/v1/memories:retrieve", ndjson())

    await collection.search("x", filter=lambda n: n.tag == "x' OR '1'='1")

    body = recorder.bodies_for("POST", "/v1/memories:retrieve")[0]
    assert body["spaceKeys"][0]["filter"] == ("CAST(val('$.tag') AS TEXT) = 'x\\' OR \\'1\\'=\\'1'")


async def test_numbers_and_booleans_use_the_casts_the_server_accepts(collection, recorder):
    """A boolean compared as TEXT is accepted by the server and matches nothing."""
    recorder.route("GET", "/v1/spaces", spaces_listing("notes"))
    recorder.route("POST", "/v1/memories:retrieve", ndjson())

    await collection.search("x", filter=lambda n: n.year > 2000)
    assert (
        recorder.bodies_for("POST", "/v1/memories:retrieve")[0]["spaceKeys"][0]["filter"]
        == "CAST(val('$.year') AS NUMERIC) > 2000"
    )

    await collection.search("x", filter=lambda n: n.active == True)  # noqa: E712
    assert (
        recorder.bodies_for("POST", "/v1/memories:retrieve")[1]["spaceKeys"][0]["filter"]
        == "CAST(val('$.active') AS BOOLEAN) = true"
    )


async def test_and_or_not_and_in_are_translated(collection, recorder):
    recorder.route("GET", "/v1/spaces", spaces_listing("notes"))
    recorder.route("POST", "/v1/memories:retrieve", ndjson())

    await collection.search("x", filter=lambda n: n.tag == "a" and n.year == 2026)
    await collection.search("x", filter=lambda n: n.tag == "a" or n.tag == "b")
    await collection.search("x", filter=lambda n: not (n.tag == "a"))
    await collection.search("x", filter=lambda n: n.tag in ["a", "b"])

    bodies = recorder.bodies_for("POST", "/v1/memories:retrieve")
    expressions = [b["spaceKeys"][0]["filter"] for b in bodies]
    assert expressions[0] == (
        "(CAST(val('$.tag') AS TEXT) = 'a') AND (CAST(val('$.year') AS NUMERIC) = 2026)"
    )
    assert expressions[1] == (
        "(CAST(val('$.tag') AS TEXT) = 'a') OR (CAST(val('$.tag') AS TEXT) = 'b')"
    )
    assert expressions[2] == "NOT (CAST(val('$.tag') AS TEXT) = 'a')"
    assert expressions[3] == "CAST(val('$.tag') AS TEXT) IN ('a', 'b')"


async def test_filtering_on_an_unknown_field_is_refused(collection, recorder):
    recorder.route("GET", "/v1/spaces", spaces_listing("notes"))

    with pytest.raises(VectorStoreOperationNotSupportedException, match="not a field"):
        await collection.search("x", filter=lambda n: n.nope == "a")


async def test_filtering_on_the_content_field_is_refused(collection, recorder):
    """Content is the embedded body, not metadata, so it cannot be filtered."""
    recorder.route("GET", "/v1/spaces", spaces_listing("notes"))

    with pytest.raises(VectorStoreOperationNotSupportedException, match="content"):
        await collection.search("x", filter=lambda n: n.content == "a")


def test_a_field_name_that_cannot_be_encoded_is_refused():
    with pytest.raises(ValueError, match="Unsupported metadata field"):
        gm_filters.compare("bad field", "=", "x")


def test_a_control_character_in_a_value_is_refused():
    """The grammar rejects a raw newline in a literal outright."""
    with pytest.raises(ValueError, match="control characters"):
        gm_filters.compare("tag", "=", "a\nb")


# ---------------------------------------------------------------------------
# P32 — embedder reuse
# ---------------------------------------------------------------------------


async def test_reusing_a_space_with_a_different_embedder_is_an_error(
    sdk_client, recorder, settings
):
    """0.2.0 wrote into whatever space matched the name."""
    other = GoodMemCollection(
        record_type=Note,
        collection_name="notes",
        settings=settings.model_copy(
            update={"embedder_id": "019cfd94-2844-7117-85ca-1b9919758a26"}
        ),
        client=sdk_client,
    )
    recorder.route("GET", "/v1/spaces", spaces_listing("notes", embedder_id=EMBEDDER_ID))

    with pytest.raises(VectorStoreInitializationException, match="cannot be changed"):
        await other.ensure_collection_exists()


async def test_creating_a_space_without_an_embedder_is_an_error(sdk_client, recorder, monkeypatch):
    """0.2.0 silently picked the first embedder the server listed.

    The env var has to be cleared rather than passed as ``None``: pydantic
    settings treat an explicit ``None`` as "unset" and fall back to it.
    """
    monkeypatch.delenv("GOODMEM_EMBEDDER_ID", raising=False)
    collection = GoodMemCollection(
        record_type=Note,
        collection_name="notes",
        settings=GoodMemSettings(base_url="https://goodmem.test", api_key="k"),
        client=sdk_client,
    )
    recorder.route("GET", "/v1/spaces", httpx.Response(200, json={"spaces": []}))

    with pytest.raises(VectorStoreInitializationException, match="GOODMEM_EMBEDDER_ID"):
        await collection.ensure_collection_exists()

    assert not [c for c in recorder.calls() if c == ("POST", "/v1/spaces")]


async def test_duplicate_space_names_are_an_error(collection, recorder):
    recorder.route(
        "GET",
        "/v1/spaces",
        httpx.Response(
            200,
            json={
                "spaces": [
                    space_json("notes", space_id="01a0d16b-bbcd-701c-bfb4-fa306021e078"),
                    space_json("notes", space_id="01a0d16b-bbcd-701c-bfb4-fa306021e079"),
                ]
            },
        ),
    )

    with pytest.raises(VectorStoreInitializationException, match="Delete the"):
        await collection.ensure_collection_exists()


# ---------------------------------------------------------------------------
# P6 / P7 / P5 / P22 / P38
# ---------------------------------------------------------------------------


async def test_listing_collections_follows_pagination(sdk_client, recorder, settings):
    """0.2.0's own client paginated; the store's listing is checked here too."""
    from goodmem_semantic_kernel import GoodMemStore

    pages = [
        httpx.Response(200, json={"spaces": [space_json("one")], "nextToken": "t1"}),
        httpx.Response(200, json={"spaces": [space_json("two")]}),
    ]
    calls = {"n": 0}

    def listing(request: httpx.Request) -> httpx.Response:
        response = pages[min(calls["n"], 1)]
        calls["n"] += 1
        return response

    recorder.route("GET", "/v1/spaces", listing)
    store = GoodMemStore(settings=settings, client=sdk_client)

    assert sorted(await store.list_collection_names()) == ["one", "two"]
    assert calls["n"] == 2


async def test_a_server_error_carries_the_servers_message(collection, recorder):
    recorder.route("GET", "/v1/spaces", spaces_listing("notes"))
    recorder.route("GET", "/v1/memories/m-1", httpx.Response(404, json={"error": "nope"}))
    recorder.route(
        "POST",
        "/v1/memories",
        httpx.Response(400, json={"errors": [{"field": "content", "message": "must be provided"}]}),
    )

    with pytest.raises(VectorStoreOperationException, match="must be provided"):
        await collection.upsert(Note(id="m-1", content=""))


async def test_upsert_waits_for_indexing_when_asked(sdk_client, recorder, settings):
    """GoodMem indexes asynchronously: without this a search right after a
    write can miss it."""
    waiting = GoodMemCollection(
        record_type=Note,
        collection_name="notes",
        settings=settings.model_copy(update={"wait_for_indexing": True}),
        client=sdk_client,
    )
    recorder.route("GET", "/v1/spaces", spaces_listing("notes"))
    recorder.route(
        "POST", "/v1/memories", httpx.Response(201, json=memory_json("m-1", status="PENDING"))
    )

    statuses = ["PENDING", "PROCESSING", "COMPLETED"]
    calls = {"n": 0}

    def get_memory(request: httpx.Request) -> httpx.Response:
        status = statuses[min(calls["n"], len(statuses) - 1)]
        calls["n"] += 1
        return httpx.Response(200, json=memory_json("m-1", status=status))

    recorder.route("GET", "/v1/memories/m-1", get_memory)

    await waiting.upsert(Note(content="hi"))

    assert calls["n"] >= 3, "must poll until indexing completes"


async def test_an_injected_client_is_not_closed(collection, sdk_client):
    """The caller keeps ownership of a client they supplied."""
    assert collection.managed_client is False
    await collection.__aexit__(None, None, None)
    assert sdk_client._http.is_closed is False


async def test_delete_tolerates_a_key_that_is_already_gone(collection, recorder):
    recorder.route("GET", "/v1/spaces", spaces_listing("notes"))
    recorder.route("DELETE", "/v1/memories/m-1", httpx.Response(404, json={"error": "nope"}))

    await collection.delete(["m-1"])  # must not raise


async def test_a_created_space_sends_a_chunking_config(collection, recorder):
    """The server rejects a space created without one (HTTP 400)."""
    recorder.route("GET", "/v1/spaces", httpx.Response(200, json={"spaces": []}))
    recorder.route("POST", "/v1/spaces", httpx.Response(201, json=space_json("notes")))

    await collection.ensure_collection_exists()

    body = recorder.bodies_for("POST", "/v1/spaces")[0]
    assert "defaultChunkingConfig" in body
    assert body["spaceEmbedders"][0]["embedderId"] == EMBEDDER_ID


async def test_a_precomputed_vector_is_refused_clearly(collection, recorder):
    recorder.route("GET", "/v1/spaces", spaces_listing("notes"))

    with pytest.raises(VectorStoreOperationNotSupportedException, match="embed it server-side"):
        await collection.search(vector=[0.1, 0.2])
