"""Test support: drive the real SDK over a mock HTTP transport.

Nothing inside the connector or the SDK is monkeypatched, so a passing test
means the wire behaviour is right rather than that a stub was called. Event
shapes come from ``fixtures/retrieve_real.ndjson``, captured from a live
GoodMem server (v1.0.320), not hand-written from a guess at the schema.
"""

# NOTE: no `from __future__ import annotations` here. Semantic Kernel reads
# VectorStoreField metadata off the class signature, and postponed
# annotations turn those into strings it cannot inspect.
import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import httpx
from goodmem import AsyncGoodmem
from semantic_kernel.data.vector import VectorStoreField, vectorstoremodel

from goodmem_semantic_kernel import GoodMemSettings

FIXTURES = Path(__file__).parent / "fixtures"

# A real vector relevance score, straight from the capture. It is negative:
# GoodMem vector scores are opaque similarities, not 0-1 relevance.
REAL_VECTOR_SCORE = -0.5911163091659546

SPACE_ID = "01a0d16b-bbcd-701c-bfb4-fa306021e078"
EMBEDDER_ID = "019cfd1c-c033-7517-b7de-f73941a0464b"


@vectorstoremodel
@dataclass
class Note:
    """The record type the tests store."""

    id: Annotated[str | None, VectorStoreField("key")] = None
    content: Annotated[str, VectorStoreField("data")] = ""
    tag: Annotated[str | None, VectorStoreField("data")] = None
    year: Annotated[int | None, VectorStoreField("data")] = None
    active: Annotated[bool | None, VectorStoreField("data")] = None


def real_events() -> list[dict[str, Any]]:
    text = (FIXTURES / "retrieve_real.ndjson").read_text()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _template(kind: str) -> dict[str, Any]:
    for event in real_events():
        if kind in event:
            return json.loads(json.dumps(event))
    raise AssertionError(f"fixture has no {kind} event")


def chunk_event(
    chunk_id: str,
    text: str,
    memory_id: str,
    score: float = REAL_VECTOR_SCORE,
    memory_index: int = 0,
) -> dict[str, Any]:
    """A retrievedItem with the real field set, overriding only what matters."""
    event = _template("retrievedItem")
    reference = event["retrievedItem"]["chunk"]
    reference["chunk"]["chunkId"] = chunk_id
    reference["chunk"]["chunkText"] = text
    reference["chunk"]["memoryId"] = memory_id
    reference["relevanceScore"] = score
    reference["memoryIndex"] = memory_index
    return event


def memory_event(memory_id: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    event = _template("memoryDefinition")
    event["memoryDefinition"]["memoryId"] = memory_id
    event["memoryDefinition"]["metadata"] = metadata or {}
    return event


def status_event(code: str | None, message: str, **details: str) -> dict[str, Any]:
    """A status event. ``code=None`` omits the field, as a newer server's would."""
    status: dict[str, Any] = {"message": message}
    if code is not None:
        status["code"] = code
    if details:
        status["details"] = details
    return {"status": status}


def ndjson(*objects: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        200,
        text="\n".join(json.dumps(o) for o in objects),
        headers={"content-type": "application/x-ndjson"},
    )


def space_json(
    name: str, space_id: str = SPACE_ID, embedder_id: str = EMBEDDER_ID
) -> dict[str, Any]:
    """A Space with the fields the server really returns."""
    return {
        "spaceId": space_id,
        "name": name,
        "labels": {},
        "spaceEmbedders": [
            {
                "spaceId": space_id,
                "embedderId": embedder_id,
                "defaultRetrievalWeight": 1.0,
                "createdAt": 1790219893735,
                "updatedAt": 1790219893735,
                "createdById": "019cfcff-37c7-75ef-be71-06c83dae99c3",
                "updatedById": "019cfcff-37c7-75ef-be71-06c83dae99c3",
            }
        ],
        "createdAt": 1790219893735,
        "updatedAt": 1790219893735,
        "createdById": "019cfcff-37c7-75ef-be71-06c83dae99c3",
        "updatedById": "019cfcff-37c7-75ef-be71-06c83dae99c3",
        "ownerId": "019cfcff-37c5-76d0-bd46-8525e29a9c82",
        "defaultChunkingConfig": {
            "recursive": {
                "chunkSize": 512,
                "chunkOverlap": 64,
                "keepStrategy": "KEEP_END",
                "lengthMeasurement": "CHARACTER_COUNT",
            }
        },
    }


def memory_json(
    memory_id: str,
    *,
    content: str | None = None,
    metadata: dict[str, Any] | None = None,
    status: str = "COMPLETED",
) -> dict[str, Any]:
    """A Memory as the server returns it. ``originalContent`` is base64."""
    payload: dict[str, Any] = {
        "memoryId": memory_id,
        "spaceId": SPACE_ID,
        "originalContentLength": len(content or ""),
        "contentType": "text/plain",
        "processingStatus": status,
        "pageImageStatus": "PENDING",
        "pageImageCount": 0,
        "metadata": metadata or {},
        "createdAt": 1790219893735,
        "updatedAt": 1790219895810,
        "createdById": "019cfcff-37c7-75ef-be71-06c83dae99c3",
        "updatedById": "019cfcff-37c7-75ef-be71-06c83dae99c3",
    }
    if content is not None:
        payload["originalContent"] = base64.b64encode(content.encode()).decode()
    return payload


class Recorder:
    """Records outgoing requests and serves queued responses."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.routes: list[tuple[str, str, Any]] = []
        self.default: Any = None

    def route(self, method: str, path_suffix: str, response: Any) -> None:
        self.routes.append((method.upper(), path_suffix, response))

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        for method, suffix, response in self.routes:
            if request.method == method and request.url.path.endswith(suffix):
                return response(request) if callable(response) else response
        if self.default is not None:
            return self.default(request) if callable(self.default) else self.default
        return httpx.Response(404, json={"error": f"unrouted {request.method} {request.url.path}"})

    def body(self, index: int = -1) -> dict[str, Any]:
        content = self.requests[index].content
        return json.loads(content) if content else {}

    def calls(self) -> list[tuple[str, str]]:
        return [(r.method, r.url.path) for r in self.requests]

    def bodies_for(self, method: str, suffix: str) -> list[dict[str, Any]]:
        out = []
        for request in self.requests:
            if request.method == method.upper() and request.url.path.endswith(suffix):
                out.append(json.loads(request.content) if request.content else {})
        return out


def build_client(recorder: "Recorder") -> AsyncGoodmem:
    """An SDK client whose transport is the recorder.

    base_url and the API key live on the httpx client: the SDK refuses to take
    them alongside an injected http_client.
    """
    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(recorder.handler),
        base_url="https://goodmem.test",
        headers={"x-api-key": "test-key"},
    )
    return AsyncGoodmem(http_client=http_client)


def build_settings() -> GoodMemSettings:
    return GoodMemSettings(
        base_url="https://goodmem.test",
        api_key="test-key",
        embedder_id=EMBEDDER_ID,
        wait_for_indexing=False,
    )
