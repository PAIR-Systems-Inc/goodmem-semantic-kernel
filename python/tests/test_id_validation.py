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
from support import EMBEDDER_ID, MEMORY_NEW, RERANKER_ID, SPACE_ID, Note, memory_json, space_json

from goodmem_semantic_kernel import GoodMemCollection, GoodMemSettings

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


class RecordingServer:
    """A local HTTP server that records every request line it receives."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, str]] = []
        self.bodies: list[dict[str, Any]] = []
        self.listed_space_id = SPACE_ID
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
            memory_id = body.get("memoryId") or MEMORY_NEW
            return 201, memory_json(memory_id, content="x"), "application/json"
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
    assert [r for r in server.requests if not r[1].startswith("/v1/spaces")] == [
        ("GET", f"/v1/memories/{U}"),
        ("POST", "/v1/memories"),
    ]
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

    assert not [r for r in server.requests if r[0] == "DELETE"], server.requests
    assert isinstance(error, ValueError) and REFUSAL in str(error), repr(error)


async def test_ensure_collection_deleted_deletes_exactly_the_listed_space(server, make_collection):
    await make_collection().ensure_collection_deleted()

    assert server.requests[-1] == ("DELETE", f"/v1/spaces/{SPACE_ID}")


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
