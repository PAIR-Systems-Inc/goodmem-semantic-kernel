"""The README's snippets run, and its facts match the code.

The Python snippets are executed as written against a local HTTP server that
stands in for GoodMem (and, for the agent snippet, for OpenAI's chat
endpoint), so they go through the real connector, SDK, httpx and TCP. Only
the base URLs and keys come from the environment, which is how a reader runs
them. The other checks compare README text with the settings class, the
.NET and Java option classes and the collected test counts.
"""

# No `from __future__ import annotations`: see support.py.
import ast
import inspect
import json
import os
import re
import subprocess
import sys
import textwrap
import threading
import uuid
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from semantic_kernel.kernel_pydantic import KernelBaseSettings
from support import EMBEDDER_ID, chunk_event, memory_event, memory_json, space_json

from goodmem_semantic_kernel import GoodMemCollection, GoodMemSettings

ROOT = Path(__file__).resolve().parents[2]
README = (ROOT / "README.md").read_text()


def blocks(lang: str) -> list[str]:
    """Fenced blocks of one language, list-item indentation removed."""
    found = re.findall(r"^( *)```(\w*)\n(.*?)^\1```", README, re.S | re.M)
    return [textwrap.dedent(body) for _, kind, body in found if kind == lang]


def python_block(marker: str) -> str:
    matches = [b for b in blocks("python") if marker in b]
    assert len(matches) == 1, f"expected one python block containing {marker!r}"
    return matches[0]


class StandIn:
    """GoodMem's memory and space endpoints, plus a scripted chat model.

    Shapes come from support.py, which takes them from a live capture:
    ``originalContent`` is base64 on reads and vector scores are negative.
    """

    def __init__(self) -> None:
        self.requests: list[tuple[str, str, dict[str, Any]]] = []
        self.spaces: dict[str, str] = {}
        self.memories: dict[str, dict[str, Any]] = {}
        stand_in = self

        class Handler(BaseHTTPRequestHandler):
            def _handle(self) -> None:
                length = int(self.headers.get("content-length") or 0)
                raw = self.rfile.read(length) if length else b""
                body = json.loads(raw) if raw else {}
                stand_in.requests.append((self.command, self.path, body))
                status, payload, content_type = stand_in.respond(self.command, self.path, body)
                data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
                if isinstance(payload, str):
                    data = payload.encode()
                self.send_response(status)
                self.send_header("content-type", content_type)
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            do_GET = do_POST = do_DELETE = _handle

            def log_message(self, *args: Any) -> None:
                pass

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_address[1]}"

    def calls(self, method: str, route: str) -> list[dict[str, Any]]:
        return [b for m, p, b in self.requests if m == method and p.split("?")[0] == route]

    def respond(self, method: str, path: str, body: dict[str, Any]) -> tuple[int, Any, str]:
        route = path.split("?", 1)[0]
        json_type = "application/json"
        if route == "/v1/chat/completions":
            return 200, self.chat(body), json_type
        if method == "GET" and route == "/v1/spaces":
            spaces = [space_json(name, space_id=sid) for sid, name in self.spaces.items()]
            return 200, {"spaces": spaces}, json_type
        if method == "POST" and route == "/v1/spaces":
            space_id = str(uuid.uuid4())
            self.spaces[space_id] = body["name"]
            return 201, space_json(body["name"], space_id=space_id), json_type
        if method == "POST" and route == "/v1/memories":
            if not body.get("originalContent"):
                message = "Either originalContent or originalContentB64 must be provided"
                return 400, {"error": message, "status": 400}, json_type
            memory_id = body.get("memoryId") or str(uuid.uuid4())
            self.memories[memory_id] = body
            return 201, memory_json(memory_id, metadata=body.get("metadata")), json_type
        if method == "POST" and route == "/v1/memories:batchGet":
            results = [
                {
                    "memoryId": key,
                    "success": True,
                    "memory": memory_json(
                        key,
                        content=self.memories[key]["originalContent"],
                        metadata=self.memories[key].get("metadata"),
                    ),
                }
                for key in body["memoryIds"]
                if key in self.memories
            ]
            return 200, {"results": results}, json_type
        if method == "POST" and route == "/v1/memories:retrieve":
            events = []
            for index, (key, memory) in enumerate(self.memories.items()):
                events.append(memory_event(key, memory.get("metadata")))
                text = memory["originalContent"]
                events.append(chunk_event(str(uuid.uuid4()), text, key, memory_index=index))
            return 200, "\n".join(json.dumps(e) for e in events), "application/x-ndjson"
        match = re.fullmatch(r"/v1/memories/([0-9a-f-]{36})(/content)?", route)
        if match and match.group(1) in self.memories:
            key = match.group(1)
            if method == "DELETE":
                del self.memories[key]
                return 204, b"", json_type
            if match.group(2):
                return 200, self.memories[key]["originalContent"].encode(), "text/plain"
            return 200, memory_json(key, metadata=self.memories[key].get("metadata")), json_type
        return 404, {"error": f"no route for {method} {path}"}, json_type

    @staticmethod
    def chat(body: dict[str, Any]) -> dict[str, Any]:
        """Call the first tool with the user's text, then answer with its output."""
        messages = body["messages"]
        tool_output = [m["content"] for m in messages if m.get("role") == "tool"]
        base = {
            "id": "chatcmpl-readme",
            "object": "chat.completion",
            "created": 0,
            "model": body["model"],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        if body.get("tools") and not tool_output:
            function = body["tools"][0]["function"]
            argument = next(iter(function["parameters"]["properties"]))
            question = next(m["content"] for m in messages if m.get("role") == "user")
            if isinstance(question, list):
                question = " ".join(part.get("text", "") for part in question)
            call = {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": function["name"],
                    "arguments": json.dumps({argument: question}),
                },
            }
            message: dict[str, Any] = {"role": "assistant", "content": None, "tool_calls": [call]}
            finish = "tool_calls"
        else:
            message = {"role": "assistant", "content": f"From memory: {tool_output}"}
            finish = "stop"
        return {**base, "choices": [{"index": 0, "finish_reason": finish, "message": message}]}

    def __enter__(self) -> "StandIn":
        self._thread.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()


@pytest.fixture
def stand_in(monkeypatch: pytest.MonkeyPatch) -> Iterator[StandIn]:
    """The environment a reader sets, pointed at the local stand-in."""
    with StandIn() as server:
        for name in list(os.environ):
            if name.startswith(("GOODMEM_", "OPENAI_")):
                monkeypatch.delenv(name)
        monkeypatch.setenv("GOODMEM_BASE_URL", server.url)
        monkeypatch.setenv("GOODMEM_API_KEY", "test-key")
        monkeypatch.setenv("GOODMEM_EMBEDDER_ID", EMBEDDER_ID)
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
        monkeypatch.setenv("OPENAI_BASE_URL", f"{server.url}/v1")
        yield server


# ---------------------------------------------------------------------------
# The snippets, executed
# ---------------------------------------------------------------------------


def test_every_python_block_parses() -> None:
    for block in blocks("python"):
        ast.parse(block, mode="exec", type_comments=False, feature_version=(3, 10))


def test_the_agent_snippet_runs_with_only_the_documented_environment(
    stand_in: StandIn, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Data model + Option A, as one file, run the way a reader runs it.
    script = tmp_path / "option_a.py"
    script.write_text(python_block("class Note") + "\n" + python_block("ChatCompletionAgent"))
    exec(compile(script.read_text(), str(script), "exec"), {"__name__": "__main__"})

    seeded = [b["originalContent"] for b in stand_in.calls("POST", "/v1/memories")]
    assert seeded, "the snippet must seed at least one record"
    assert len(stand_in.calls("POST", "/v1/memories:retrieve")) == 1, "the agent must search"
    assert len(stand_in.calls("POST", "/v1/chat/completions")) == 2
    printed = capsys.readouterr().out
    assert seeded[0] in printed, printed


async def test_the_upsert_failure_snippet_runs(stand_in: StandIn) -> None:
    namespace: dict[str, Any] = {}
    exec(python_block("class Note"), namespace)
    note_type = namespace["Note"]
    snippet = python_block("detail.restored")
    exec("async def _snippet(collection, note):\n" + textwrap.indent(snippet, "    "), namespace)

    async with GoodMemCollection(record_type=note_type, collection_name="notes") as collection:
        await collection.ensure_collection_exists()
        key = str(uuid.uuid4())
        await collection.upsert(note_type(id=key, content="old text", source="v1"))
        # Empty content is refused by the server, so the update fails.
        await namespace["_snippet"](collection, note_type(id=key, content=""))
        restored = await collection.get(key=key)

    assert restored.content == "old text"
    creates = [b["originalContent"] for b in stand_in.calls("POST", "/v1/memories")]
    assert creates == ["old text", "", "old text"], creates


def test_the_filter_snippet_runs_on_the_record_type_it_describes(
    stand_in: StandIn, tmp_path: Path
) -> None:
    (lead_in,) = re.findall(r"^- \*\*Filters.*$", README, re.M)
    assert "For a record type with `tag` and `year` data fields" in lead_in, lead_in

    # Semantic Kernel reads the lambda's source, so this has to be a file.
    script = tmp_path / "filter_snippet.py"
    script.write_text(
        textwrap.dedent(
            """
            import asyncio
            from dataclasses import dataclass
            from typing import Annotated
            from semantic_kernel.data.vector import VectorStoreField, vectorstoremodel
            from goodmem_semantic_kernel import GoodMemCollection

            @vectorstoremodel
            @dataclass
            class Report:
                id: Annotated[str | None, VectorStoreField("key")] = None
                content: Annotated[str, VectorStoreField("data", type="str")] = ""
                tag: Annotated[str | None, VectorStoreField("data")] = None
                year: Annotated[int | None, VectorStoreField("data")] = None

            async def main():
                async with GoodMemCollection(record_type=Report, collection_name="r") as collection:
                    await collection.ensure_collection_exists()
            """
        )
        + textwrap.indent(python_block("filter=lambda"), " " * 8)
        + "\nasyncio.run(main())\n"
    )
    exec(compile(script.read_text(), str(script), "exec"), {"__name__": "__main__"})

    (retrieve,) = stand_in.calls("POST", "/v1/memories:retrieve")
    assert retrieve["spaceKeys"][0]["filter"] == (
        "(CAST(val('$.tag') AS TEXT) = 'finance') AND (CAST(val('$.year') AS NUMERIC) > 2000)"
    )


def test_the_constructor_snippets_build_what_they_name(stand_in: StandIn) -> None:
    namespace: dict[str, Any] = {}
    exec(python_block("class Note"), namespace)
    exec("from goodmem_semantic_kernel import *", namespace)
    namespace["MyModel"] = namespace["Note"]
    for name in ("GoodMemCollection(", "GoodMemStore(", "GoodMemSettings(\n"):
        (block,) = [b for b in blocks("python") if b.startswith(name)]
        built = eval(block, namespace)
        assert type(built).__name__ == name.strip("(\n")


# ---------------------------------------------------------------------------
# The facts, compared with the code
# ---------------------------------------------------------------------------


def test_the_injectable_client_type_exists() -> None:
    (named,) = re.findall(r"inject a pre-built ([\w.]+)", README)
    module, _, attribute = named.rpartition(".")
    assert module, f"{named} does not say where it comes from"
    assert hasattr(__import__(module), attribute), f"{named} does not exist"
    signature = inspect.signature(GoodMemCollection.__init__)
    assert attribute in str(signature.parameters["client"].annotation)


def test_the_documented_search_default_is_the_real_one() -> None:
    (documented,) = re.findall(r"`search\(query, top=(\d+)\)`", README)
    actual = inspect.signature(GoodMemCollection.search).parameters["top"].default
    assert int(documented) == actual


def test_no_bash_line_assigns_a_variable_without_exporting_it() -> None:
    for block in blocks("bash"):
        for line in block.splitlines():
            assert not re.fullmatch(r"\s*[A-Z_][A-Z0-9_]*=\S*\s*", line), (
                f"{line.strip()!r} sets a shell variable a program will not see"
            )


def test_the_configuration_block_exports_what_creating_a_collection_needs() -> None:
    (config,) = [b for b in blocks("bash") if "export GOODMEM_API_KEY" in b]
    exported = set(re.findall(r"export (GOODMEM_\w+)=", config))
    assert {"GOODMEM_API_KEY", "GOODMEM_BASE_URL", "GOODMEM_EMBEDDER_ID"} <= exported


def config_table() -> dict[str, dict[str, str]]:
    section = README.split("## Configuration", 1)[1].split("\n## ", 1)[0]
    lines = [line for line in section.splitlines() if line.startswith("|")]
    header = [cell.strip() for cell in lines[0].strip("|").split("|")]
    rows = {}
    for line in lines[2:]:
        cells = dict(zip(header, (c.strip() for c in line.strip("|").split("|")), strict=True))
        rows[cells["Variable"].strip("`")] = cells
    return rows


def test_the_configuration_table_matches_the_settings_class() -> None:
    rows = config_table()
    # KernelBaseSettings contributes env_file_path/encoding, which are not settings.
    inherited = KernelBaseSettings.model_fields
    fields = {k: v for k, v in GoodMemSettings.model_fields.items() if k not in inherited}
    assert set(rows) == {f"GOODMEM_{name.upper()}" for name in fields}
    for name, field in fields.items():
        documented = rows[f"GOODMEM_{name.upper()}"]["Default"].strip("`")
        default = None if field.is_required() else field.default
        if default is None:
            assert documented == "—", name
        elif isinstance(default, bool):
            assert documented == str(default).lower(), name
        elif isinstance(default, float):
            assert float(documented) == default, name
        else:
            assert documented == default, name


def test_the_configuration_table_says_which_implementation_reads_each_variable() -> None:
    dotnet = (ROOT / "dotnet/GoodMem.SemanticKernel/GoodMemOptions.cs").read_text()
    java = (
        ROOT / "java/goodmem-semantic-kernel/src/main/java/ai/goodmem/semantickernel"
        "/GoodMemOptions.java"
    ).read_text()
    for variable, row in config_table().items():
        read_by = row["Read by"]
        in_dotnet, in_java = f'"{variable}"' in dotnet, f'"{variable}"' in java
        if read_by == "all":
            assert in_dotnet and in_java, f"{variable} is not read by .NET and Java"
        else:
            assert read_by == "Python", variable
            assert not in_dotnet and not in_java, f"{variable} is read outside Python"


def test_the_python_test_counts_match_the_suite() -> None:
    collected = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", str(ROOT / "python/tests")],
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
    ).stdout
    per_file: dict[str, int] = {}
    for line in collected.splitlines():
        if "::" in line:
            name = line.split("::")[0].rsplit("/", 1)[-1]
            per_file[name] = per_file.get(name, 0) + 1
    live = per_file.pop("test_e2e.py")
    (offline,) = re.findall(r"# Python: (\d+) offline tests", README)
    (documented_live,) = re.findall(r"# Python: (\d+) more live tests", README)
    assert int(offline) == sum(per_file.values()), per_file
    assert int(documented_live) == live
