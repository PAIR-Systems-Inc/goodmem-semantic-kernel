"""Ids that can reach a URL path must be UUIDs, checked before any request.

The goodmem SDK builds paths as ``f"/v1/memories/{id}"`` and httpx resolves
dot segments before sending, so in 0.3.0 ``collection.delete("../spaces/<U>")``
sent ``DELETE /v1/spaces/<U>`` and deleted a whole space.

Nothing here is mocked. Each test drives the connector's real stack --
``GoodMemCollection`` -> goodmem SDK -> httpx -> TCP -- against a local HTTP
server that records the request line of everything it receives.
"""

# No `from __future__ import annotations`: see support.py.
import json
import threading
import uuid
from collections.abc import Awaitable, Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest
from semantic_kernel.exceptions.vector_store_exceptions import (
    VectorStoreException,
    VectorStoreInitializationException,
    VectorStoreOperationException,
)
from support import (
    EMBEDDER_ID,
    MEMORY_1,
    MEMORY_2,
    MEMORY_NEW,
    RERANKER_ID,
    SPACE_ID,
    Note,
    memory_json,
    space_json,
)

from goodmem_semantic_kernel import (
    GoodMemCollection,
    GoodMemSettings,
    GoodMemStore,
    GoodMemUpsertError,
)

U = "0198c3a2-7f4e-7c1a-9b2d-5e6f7a8b9c0d"

PAYLOADS = [
    f"../spaces/{U}",
    f"a/../../spaces/{U}",
    f"%2e%2e/spaces/{U}",
    f"..%2Fspaces%2F{U}",
    f"{U}/../../spaces/{U}",
    "",
    f" {U}",
    f"{U}?x=1",
    f"{U}#frag",
    f"{U}\n",
    "..",
]

# An empty setting means "not configured", as it does for any GOODMEM_* env
# var; it is never sent anywhere. Every other payload is refused.
SETTING_PAYLOADS = [p for p in PAYLOADS if p]

REFUSAL = "must be a GoodMem UUID"

# The one request made before an id from the server can be checked.
LISTING = ("GET", "/v1/spaces?name_filter=notes")


class RecordingServer:
    """A local HTTP server that records every request line it receives."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, str]] = []
        self.bodies: list[dict[str, Any]] = []
        self.listed_space_id = SPACE_ID
        # Ids a create answers with, in order; when empty it echoes the key.
        self.returned_memory_ids: list[str] = []
        self.created: set[str] = set()
        server = self

        class Handler(BaseHTTPRequestHandler):
            def _handle(self) -> None:
                length = int(self.headers.get("content-length") or 0)
                raw = self.rfile.read(length) if length else b""
                body = json.loads(raw) if raw else {}
                # self.path is the request target exactly as it arrived.
                server.requests.append((self.command, self.path))
                server.bodies.append(body)
                status, payload, content_type = server.respond(self.command, self.path, body)
                data = (
                    payload.encode() if isinstance(payload, str) else json.dumps(payload).encode()
                )
                self.send_response(status)
                self.send_header("content-type", content_type)
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            do_GET = do_POST = do_DELETE = do_PUT = do_PATCH = _handle

            def log_message(self, *args: Any) -> None:
                pass

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_address[1]}"

    def respond(self, method: str, path: str, body: dict[str, Any]) -> tuple[int, Any, str]:
        route = path.split("?", 1)[0]
        if method == "GET" and route == "/v1/spaces":
            return (
                200,
                {"spaces": [space_json("notes", space_id=self.listed_space_id)]},
                "application/json",
            )
        if method == "POST" and route == "/v1/spaces":
            return 201, space_json("notes"), "application/json"
        if method == "POST" and route == "/v1/memories:batchGet":
            return 200, {"results": []}, "application/json"
        if method == "POST" and route == "/v1/memories:retrieve":
            return 200, "", "application/x-ndjson"
        if method == "POST" and route == "/v1/memories":
            if self.returned_memory_ids:
                memory_id = self.returned_memory_ids.pop(0)
            else:
                memory_id = body.get("memoryId") or MEMORY_NEW
            self.created.add(memory_id.lower())
            return 201, memory_json(memory_id, content="x"), "application/json"
        if method == "GET" and route.removeprefix("/v1/memories/") in self.created:
            return 200, memory_json(route.removeprefix("/v1/memories/")), "application/json"
        if method == "DELETE":
            return 204, "", "application/json"
        return 404, {"error": f"no route for {method} {path}"}, "application/json"

    def __enter__(self) -> "RecordingServer":
        self._thread.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()


@pytest.fixture
def server() -> Iterator[RecordingServer]:
    with RecordingServer() as recording:
        yield recording


@pytest.fixture
async def make_collection(server: RecordingServer):
    """Build collections the way a user does: from settings, no injected client."""
    made: list[GoodMemCollection] = []

    def build(**overrides: Any) -> GoodMemCollection:
        values: dict[str, Any] = {
            "base_url": server.url,
            "api_key": "test-key",
            "embedder_id": EMBEDDER_ID,
            "wait_for_indexing": False,
        }
        values.update(overrides)
        collection = GoodMemCollection(
            record_type=Note, collection_name="notes", settings=GoodMemSettings(**values)
        )
        made.append(collection)
        return collection

    yield build
    for collection in made:
        await collection.__aexit__(None, None, None)


async def outcome(call: Callable[[], Awaitable[Any]]) -> Exception | None:
    """Run ``call`` and return what it raised, so the requests can be checked first."""
    try:
        await call()
    except Exception as exc:
        return exc
    return None


def assert_refused(error: Exception | None, server: RecordingServer, kind: type = Exception):
    # Requests first: on 0.3.0 this is the line that fails, and it shows the
    # path the server actually received.
    assert server.requests == [], f"the server received {server.requests}"
    assert isinstance(error, kind), f"expected {kind.__name__}, got {error!r}"
    assert REFUSAL in str(error), str(error)


# ---------------------------------------------------------------------------
# delete: the key is the path segment of DELETE /v1/memories/{key}
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("payload", PAYLOADS)
async def test_delete_refuses_a_non_uuid_key(payload, server, make_collection):
    collection = make_collection()

    error = await outcome(lambda: collection.delete(payload))

    assert_refused(error, server, VectorStoreOperationException)


@pytest.mark.parametrize("payload", PAYLOADS)
async def test_delete_refuses_the_whole_batch_before_deleting_any(payload, server, make_collection):
    collection = make_collection()

    error = await outcome(lambda: collection.delete([U, payload]))

    assert_refused(error, server, VectorStoreOperationException)


async def test_delete_with_a_uuid_reaches_exactly_that_memory(server, make_collection):
    await make_collection().delete(U)

    assert server.requests == [("DELETE", f"/v1/memories/{U}")]


async def test_an_uppercase_uuid_is_normalised(server, make_collection):
    await make_collection().delete(U.upper())

    assert server.requests == [("DELETE", f"/v1/memories/{U}")]


# ---------------------------------------------------------------------------
# upsert: the key reaches GET/DELETE /v1/memories/{key} on the replace path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("payload", PAYLOADS)
async def test_upsert_refuses_a_non_uuid_key(payload, server, make_collection):
    collection = make_collection()

    error = await outcome(lambda: collection.upsert(Note(id=payload, content="x")))

    assert_refused(error, server, VectorStoreOperationException)


@pytest.mark.parametrize("payload", PAYLOADS)
async def test_upsert_refuses_the_whole_batch_before_writing_any(payload, server, make_collection):
    collection = make_collection()
    batch = [Note(id=U, content="first"), Note(id=payload, content="second")]

    error = await outcome(lambda: collection.upsert(batch))

    assert_refused(error, server, VectorStoreOperationException)


async def test_upsert_with_a_uuid_key_reaches_exactly_that_memory(server, make_collection):
    key = await make_collection().upsert(Note(id=U, content="x"))

    assert key == U
    assert server.requests == [LISTING, ("GET", f"/v1/memories/{U}"), ("POST", "/v1/memories")]
    assert server.bodies[-1]["memoryId"] == U


async def test_upsert_without_a_key_lets_the_server_assign_one(server, make_collection):
    key = await make_collection().upsert(Note(content="x"))

    assert key == MEMORY_NEW
    assert "memoryId" not in server.bodies[-1]


# ---------------------------------------------------------------------------
# get: ids travel in the batchGet body, and are checked the same way
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("payload", PAYLOADS)
async def test_get_refuses_a_non_uuid_key(payload, server, make_collection):
    collection = make_collection()

    error = await outcome(lambda: collection.get(keys=[U, payload]))

    assert_refused(error, server, VectorStoreOperationException)


async def test_get_with_a_uuid_asks_for_exactly_that_memory(server, make_collection):
    await make_collection().get(key=U.upper())

    assert server.requests == [("POST", "/v1/memories:batchGet")]
    assert server.bodies[0]["memoryIds"] == [U]


# ---------------------------------------------------------------------------
# Ids from configuration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("payload", SETTING_PAYLOADS)
async def test_a_non_uuid_embedder_setting_is_refused(payload, server, make_collection):
    collection = make_collection(embedder_id=payload)

    error = await outcome(collection.ensure_collection_exists)

    assert_refused(error, server, VectorStoreInitializationException)
    assert "GOODMEM_EMBEDDER_ID" in str(error)


async def test_a_uuid_embedder_setting_is_used(server, make_collection):
    collection = make_collection()

    await collection.ensure_collection_exists()

    assert [r[0] for r in server.requests] == ["GET"]


@pytest.mark.parametrize("payload", SETTING_PAYLOADS)
async def test_a_non_uuid_reranker_setting_is_refused(payload, server, make_collection):
    collection = make_collection(reranker_id=payload)

    error = await outcome(lambda: collection.search("x"))

    assert_refused(error, server, VectorStoreException)


async def test_a_uuid_reranker_setting_is_sent(server, make_collection):
    results = await make_collection(reranker_id=RERANKER_ID.upper()).search("x")
    _ = [r async for r in results.results]

    retrieve = server.bodies[server.requests.index(("POST", "/v1/memories:retrieve"))]
    assert retrieve["postProcessor"]["config"]["reranker_id"] == RERANKER_ID


# ---------------------------------------------------------------------------
# Ids from the server: the space id in DELETE /v1/spaces/{id}
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("payload", SETTING_PAYLOADS)
async def test_a_non_uuid_space_id_from_the_server_is_never_deleted(
    payload, server, make_collection
):
    server.listed_space_id = payload
    collection = make_collection()

    error = await outcome(collection.ensure_collection_deleted)

    # Only the listing: nothing is sent with the listed id.
    assert server.requests == [LISTING], server.requests
    # A Semantic Kernel exception like every other refusal, not a bare ValueError.
    assert isinstance(error, VectorStoreOperationException), repr(error)
    assert REFUSAL in str(error) and "not deleted" in str(error), str(error)


@pytest.mark.parametrize("payload", SETTING_PAYLOADS)
async def test_the_store_does_not_hide_a_refused_space_delete(payload, server):
    """Semantic Kernel's VectorStore.ensure_collection_deleted swallows
    VectorStoreOperationException; a refused delete must not look like a done one."""
    server.listed_space_id = payload
    store = GoodMemStore(settings=GoodMemSettings(base_url=server.url, api_key="test-key"))
    try:
        error = await outcome(lambda: store.ensure_collection_deleted("notes"))
    finally:
        await store.__aexit__(None, None, None)

    assert server.requests == [LISTING], server.requests
    assert isinstance(error, VectorStoreOperationException), repr(error)
    assert REFUSAL in str(error), str(error)


async def test_ensure_collection_deleted_deletes_exactly_the_listed_space(server, make_collection):
    await make_collection().ensure_collection_deleted()

    assert server.requests[-1] == ("DELETE", f"/v1/spaces/{SPACE_ID}")


async def test_the_store_deletes_exactly_the_listed_space(server):
    store = GoodMemStore(settings=GoodMemSettings(base_url=server.url, api_key="test-key"))
    try:
        await store.ensure_collection_deleted("notes")
    finally:
        await store.__aexit__(None, None, None)

    assert server.requests == [LISTING, ("DELETE", f"/v1/spaces/{SPACE_ID}")]


# ---------------------------------------------------------------------------
# Ids from the server: the memory id a create answers with
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("wait_for_indexing", [True, False])
@pytest.mark.parametrize("payload", PAYLOADS)
async def test_a_non_uuid_memory_id_from_create_is_reported_as_a_partial_write(
    payload, wait_for_indexing, server, make_collection
):
    # The first record gets a real id; the server answers the second with the payload.
    server.returned_memory_ids = [MEMORY_1, payload]
    collection = make_collection(wait_for_indexing=wait_for_indexing)

    error = await outcome(
        lambda: collection.upsert([Note(content="first"), Note(content="second")])
    )

    # Both creates were sent, and nothing after them: the returned id is
    # never used in a path, not even to wait for the first record's indexing.
    assert server.requests == [
        LISTING,
        ("POST", "/v1/memories"),
        ("POST", "/v1/memories"),
    ], server.requests
    assert isinstance(error, VectorStoreOperationException), repr(error)
    detail = error.__cause__
    assert isinstance(detail, GoodMemUpsertError), repr(detail)
    # The first record is reported as written, and the message does not claim
    # the second one was not sent: it was, and it is on the server.
    assert detail.written_keys == [MEMORY_1]
    assert "not a UUID" in str(detail) and "record 2 of 2" in str(detail), str(detail)
    assert "was not sent" not in str(detail), str(detail)


@pytest.mark.parametrize("wait_for_indexing", [True, False])
async def test_uuid_memory_ids_from_create_are_returned_and_waited_on(
    wait_for_indexing, server, make_collection
):
    server.returned_memory_ids = [MEMORY_1.upper(), MEMORY_2]
    collection = make_collection(wait_for_indexing=wait_for_indexing)

    keys = await collection.upsert([Note(content="first"), Note(content="second")])

    assert keys == [MEMORY_1, MEMORY_2]
    expected = [LISTING, ("POST", "/v1/memories"), ("POST", "/v1/memories")]
    if wait_for_indexing:
        expected += [("GET", f"/v1/memories/{MEMORY_1}"), ("GET", f"/v1/memories/{MEMORY_2}")]
    assert server.requests == expected


# ---------------------------------------------------------------------------
# In-process objects that pass the check and then change what is sent
# ---------------------------------------------------------------------------

TRAVERSAL = f"../spaces/{U}"


class LowerLies(str):
    """Is a UUID when checked; lowercasing it gives a traversal."""

    def lower(self) -> str:
        return TRAVERSAL


class FormatLies(str):
    """Is a UUID when checked; lowercases to itself and formats as a traversal,
    which is what the SDK's f-string path calls."""

    def lower(self) -> str:
        return self

    def __format__(self, spec: str) -> str:
        return TRAVERSAL


class StrLies(str):
    """Its str() is a LowerLies, as a key goes through str() when serialized."""

    def __str__(self) -> str:
        return LowerLies(U)


class UuidLies(uuid.UUID):
    """A uuid.UUID whose str() is a LowerLies."""

    def __str__(self) -> str:
        return LowerLies(U)


HOSTILE = {
    "lower": lambda: LowerLies(U),
    "format": lambda: FormatLies(U),
    "str": lambda: StrLies(U),
    "uuid": lambda: UuidLies(U),
}


@pytest.mark.parametrize("make", HOSTILE.values(), ids=HOSTILE.keys())
async def test_delete_sends_the_checked_uuid_not_the_callers_object(make, server, make_collection):
    await make_collection().delete(make())

    assert server.requests == [("DELETE", f"/v1/memories/{U}")]


@pytest.mark.parametrize("make", HOSTILE.values(), ids=HOSTILE.keys())
async def test_batch_delete_sends_the_checked_uuids(make, server, make_collection):
    await make_collection().delete([make(), make()])

    assert server.requests == [("DELETE", f"/v1/memories/{U}")] * 2


@pytest.mark.parametrize("make", HOSTILE.values(), ids=HOSTILE.keys())
async def test_upsert_sends_the_checked_uuid(make, server, make_collection):
    await make_collection().upsert(Note(id=make(), content="x"))

    assert server.requests == [LISTING, ("GET", f"/v1/memories/{U}"), ("POST", "/v1/memories")]
    assert server.bodies[-1]["memoryId"] == U


@pytest.mark.parametrize("make", HOSTILE.values(), ids=HOSTILE.keys())
async def test_get_asks_for_the_checked_uuid(make, server, make_collection):
    await make_collection().get(keys=[make()])

    assert server.requests == [("POST", "/v1/memories:batchGet")]
    assert server.bodies[0]["memoryIds"] == [U]


class PretendsToBeAStr:
    """isinstance(x, str) is True for this, though it is not a str."""

    @property
    def __class__(self) -> type:
        return str


async def test_an_object_that_only_claims_to_be_a_str_is_refused(server, make_collection):
    error = await outcome(lambda: make_collection().delete(PretendsToBeAStr()))

    assert_refused(error, server, VectorStoreOperationException)


# ---------------------------------------------------------------------------
# The validator itself
# ---------------------------------------------------------------------------


def test_the_validator_normalises_uuids_and_refuses_everything_else():
    from goodmem_semantic_kernel._ids import require_uuid

    assert require_uuid(U.upper(), "key") == U
    assert require_uuid(uuid.UUID(U), "key") == U
    for payload in [*PAYLOADS, None, 42, f"{U} ", U.replace("-", ""), f"{{{U}}}"]:
        with pytest.raises(ValueError, match=f"key {REFUSAL}"):
            require_uuid(payload, "key")
    with pytest.raises(ValueError, match=f"key {REFUSAL}"):
        require_uuid(PretendsToBeAStr(), "key")


@pytest.mark.parametrize("make", HOSTILE.values(), ids=HOSTILE.keys())
def test_the_validator_returns_a_plain_str_whatever_it_was_given(make):
    from goodmem_semantic_kernel._ids import require_uuid

    checked = require_uuid(make(), "key")

    # An exact str: nothing the caller's object overrides can run on it later.
    assert type(checked) is str
    assert checked == U and f"{checked}" == U and checked.lower() == U
