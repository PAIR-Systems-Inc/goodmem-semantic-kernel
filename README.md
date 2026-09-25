# goodmem-semantic-kernel

A [GoodMem](https://goodmem.ai) connector for [Microsoft Semantic Kernel](https://github.com/microsoft/semantic-kernel).

Implements Semantic Kernel's `VectorStoreCollection` and `VectorStore` interfaces so agents built on Semantic Kernel can store and retrieve memories from a GoodMem server without having to configure your own data processing pipeline

## What is GoodMem?

GoodMem is a centralized memory API for AI agents and LLMs. The point of GoodMem is so that you can easily and efficiently store and retrieve your data/memories through semantic searching, ai summaries, and context-aware results.

GoodMem stores text memories as semantic embeddings in PostgreSQL (via `pgvector`) and retrieves them by semantic similarity. Because it runs as a shared service, multiple agents can read and write to the same memory spaces simultaneously.

> Embeddings are computed **server-side**, so this connector never needs an `embedding_generator`.

### Conceptual Overview

In GoodMem all data is hosted in a "**Space**", an abstract storage unit in GoodMem.
Each **Space** can be configured with embedders and/or chunking strategies. Each Space holds "**Memories**".

**Memories** are stored content with associated metadata that are automatically chunked and embedded for efficient retrieval. All **Memories** belong to a **Space**.

**Embedders** convert your data into a vectorized format. GoodMem supports multiple embedding models & providers.

---

## Quickstart

1. [installation](#installation)
2. [configuration](#configuration)
3. [run sample files](#running-the-samples)
4. create your own integration

## Installation

### Python (recommended)

**Requirements:** Python 3.10+ and a running GoodMem server.

```bash
pip install goodmem-semantic-kernel
```

To install from source:

```bash
git clone https://github.com/PAIR-Systems-Inc/goodmem-semantic-kernel
cd goodmem-semantic-kernel
pip install -e .
```

### .NET (debian/ubuntu)

```bash
sudo apt install dotnet-sdk-8.0
```

Build the connector from source:

```bash
dotnet build dotnet/GoodMem.SemanticKernel/GoodMem.SemanticKernel.csproj
```

### Java (debian/ubuntu)

**Requirements:** JDK 17+ (JDK 21 recommended) and Maven 3.6+.

Install JDK 21 via SDKMAN (recommended):

```bash
sdk install java 21.0.5-tem
```

Or via apt:

```bash
sudo apt install openjdk-21-jdk
```

Build and install the connector into your local Maven repository:

```bash
mvn install -f java/pom.xml -DskipTests
```

## Configuration

All settings are read from environment variables with the `GOODMEM_` prefix, or passed directly via `GoodMemSettings`.

```bash
export GOODMEM_API_KEY=your_key_here
export GOODMEM_BASE_URL=https://your_goodmem_server:8080
export GOODMEM_VERIFY_SSL=true_or_false
```

| Variable | Required | Default | Description |
|---|---|---|---|
| `GOODMEM_API_KEY` | Yes | — | API key for the GoodMem server |
| `GOODMEM_BASE_URL` | No | `http://localhost:8080` | GoodMem server base URL |
| `GOODMEM_EMBEDDER_ID` | Yes, to create a collection | — | UUID of the embedder the space is indexed with. The connector will not choose one for you: the choice is permanent for a space |
| `GOODMEM_RERANKER_ID` | No | — | UUID of a reranker to apply to searches |
| `GOODMEM_VERIFY_SSL` | No | `true` | Set to `false` for self-signed certs |
| `GOODMEM_TIMEOUT` | No | `30` | Per-request timeout in seconds |
| `GOODMEM_WAIT_FOR_INDEXING` | No | `true` | Wait for each written memory to finish indexing, so a search straight after a write can find it |
| `GOODMEM_INDEXING_TIMEOUT` | No | `60` | How long that wait lasts |

## Running the samples

### Python

```bash
cd samples/python

# Option A — agent with memory tool (also requires OPENAI_API_KEY)
OPENAI_API_KEY=your_openai_key_here
python example_agent.py

# Option B — single collection
python example_single_collection.py

# Option C — store with multiple collections
python example_store.py
```

If a sample fails, double-check [Configuration](#configuration) or run inside a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

### .NET

```bash
cd samples/dotnet/ExampleAgent
dotnet run
```

Each sample lists its required environment variables at the top of `Program.cs`.

### Java

Build the connector once before running any sample:

```bash
mvn install -f java/pom.xml -DskipTests
```

Then run any sample:

```bash
cd samples/java/ExampleAgent
mvn compile exec:java
```

Each sample lists its required environment variables in the file header.

## Testing

These are the same commands CI runs.

```bash
# Python: 186 offline tests. 36 drive the real SDK over a mock HTTP transport,
# using event shapes captured from a live server. The other 150 drive the whole
# stack over TCP to a local server that records every request, to check that
# no id reaches a request path unless it is a UUID.
pip install -e ".[dev]"
ruff check python/ && ruff format --check python/
mypy
pytest python/tests -q

# Python: 13 more live tests run when a server is configured. Without these
# variables they skip, which is also how we check no credential is baked in.
GOODMEM_BASE_URL=https://localhost:8080 \
GOODMEM_API_KEY=your_key_here \
GOODMEM_EMBEDDER_ID=your_embedder_uuid \
GOODMEM_VERIFY_SSL=false \
  pytest python/tests -q

# .NET: 167 offline tests (3 integration tests skip without GOODMEM_API_KEY)
dotnet test dotnet/GoodMem.SemanticKernel.Tests/GoodMem.SemanticKernel.Tests.csproj

# Java: 143 tests, against WireMock and a local recording server
mvn -f java/pom.xml test
```

### Define a data model

```python
from dataclasses import dataclass
from typing import Annotated
from semantic_kernel.data.vector import VectorStoreField, vectorstoremodel

@vectorstoremodel
@dataclass
class Note:
    id: Annotated[str | None, VectorStoreField("key")] = None
    content: Annotated[str, VectorStoreField("data", type="str")] = ""
    source: Annotated[str | None, VectorStoreField("data")] = None
```

- Exactly one `"key"` field (the memory ID — `None` lets the server generate a UUID; any other value must be a UUID).
- One `"data"` field named `content` becomes the embedded text (`originalContent` in GoodMem).
- All other `"data"` fields are stored as metadata and returned on search results.
- `"vector"` fields are accepted for interface compatibility but ignored — GoodMem embeds server-side.

We have three example patterns provided in the samples directory. We recommend option A, but choose what works for you.

Option A (`samples/python/example_agent.py`) is the recommended pattern for production agents since the LLM decides when to call memory and what to search for, rather than the application hardcoding those decisions.

### Option A: Wired into a Semantic Kernel agent

```python
from semantic_kernel.agents import AgentThread, ChatCompletionAgent
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.functions import KernelParameterMetadata, KernelPlugin
from goodmem_semantic_kernel import GoodMemCollection

async def main():
    async with GoodMemCollection(record_type=Note, collection_name="agent-memory") as coll:
        await coll.ensure_collection_exists()
        await coll.upsert([...])  # seed your memories

        memory_plugin = KernelPlugin(
            name="memory",
            functions=[
                coll.create_search_function(
                    function_name="recall",
                    description="Search long-term memory for relevant facts.",
                    string_mapper=lambda r: r.record.content,
                )
            ],
        )

        agent = ChatCompletionAgent(
            name="MemoryAgent",
            service=OpenAIChatCompletion(),
            instructions="Always search memory before answering factual questions.",
            function_choice_behavior=FunctionChoiceBehavior.Auto(),
            plugins=[memory_plugin],
        )

        thread: AgentThread | None = None
        result = await agent.get_response(messages="Where is the Golden Gate Bridge?", thread=thread)
        print(result.content)
```

### Option B: Single collection

see [example_single_collection.py](samples/python/example_single_collection.py)

### Option C: Store (multiple collections, shared connection)

see [example_store.py](samples/python/example_store.py)

## Behavior notes

- **Ids must be UUIDs.** Record keys are GoodMem memory ids, and ids go into request URLs, so a key such as `../spaces/<id>` could otherwise send a delete to a different resource. The connector refuses any key, and any configured embedder or reranker id, that is not a canonical UUID, before it sends anything. It also never puts a space or memory id that the server returned into a URL unless that id is a UUID. .NET raises `ArgumentException` and Java raises `IllegalArgumentException`. To let the server assign a key, pass `None` (Python) or `null` (.NET, Java). In Python the exception depends on what was refused:
  - A key passed to `get`, `upsert` or `delete` raises `VectorStoreOperationException`. The connector raises `ValueError`, and Semantic Kernel wraps it.
  - `GOODMEM_RERANKER_ID` raises `VectorSearchExecutionException` from `search`. That is a subclass of `VectorStoreOperationException`.
  - `GOODMEM_EMBEDDER_ID` raises `VectorStoreInitializationException` from `ensure_collection_exists`. That is **not** a `VectorStoreOperationException`, so `except VectorStoreOperationException` does not catch it. Semantic Kernel wraps it in `VectorStoreOperationException` when `upsert` hits it first, and in `VectorSearchExecutionException` when `search` does.
  - A space id the server listed that is not a UUID makes `ensure_collection_deleted` raise `VectorStoreOperationException` and delete nothing. `GoodMemStore.ensure_collection_deleted(name)` raises it too, where Semantic Kernel's default would have swallowed it.
  - A memory id that is not a UUID in the server's answer to a create makes `upsert` raise `VectorStoreOperationException`. Its `__cause__` is a `GoodMemUpsertError`: that record was written, and `written_keys` lists the records written before it.
- **No local embedding.** Never pass an `embedding_generator` — GoodMem embeds content server-side. The parameter is accepted for interface compatibility and silently ignored.
- **Upsert semantics.** GoodMem memories are immutable — there is no update endpoint — so upserting a record that already exists deletes the old memory and creates a new one. The connector reads the current version **before** the delete and writes it back if the create fails, then raises `GoodMemUpsertError` (Python) / `GoodMemUpsertException` (.NET, Java) saying whether the restore succeeded. Semantic Kernel wraps what a collection raises, so in Python the detail is on `__cause__`:

  ```python
  try:
      await collection.upsert(note)
  except VectorStoreOperationException as exc:
      detail = exc.__cause__          # GoodMemUpsertError
      detail.restored                 # True when the old version is back
      detail.lost_key                 # set only if it could not be restored
      detail.written_keys             # records written before the failure
  ```
- **`content` is write-only in GoodMem.** The server does not return `originalContent` in search responses. Retrieved text comes from `chunkText` (a chunk of the original), which the connector maps back to your `content` field transparently.
- **Score convention.** A GoodMem vector `relevanceScore` is a raw pgvector value where lower means more similar, so the connector negates it and Semantic Kernel's higher-is-better convention holds. A **reranker** score is already higher-is-better and is passed through unchanged — reranker ranges are provider-dependent (Voyage rerank-2.5 returns roughly `0.27..0.93`, Jina v3 `-0.14..0.43`), so do not assume 0–1 when choosing a threshold.
- **Filters.** `search(filter=...)` is translated to a GoodMem filter expression and evaluated server-side:

  ```python
  await collection.search("quarterly", filter=lambda n: n.tag == "finance" and n.year > 2000)
  ```

  `==`, `!=`, `<`, `<=`, `>`, `>=`, `in`, `not in`, `and`, `or` and `not` are supported. Values are quoted and cast for you — a value containing an apostrophe is a value, not syntax — and a boolean is compared with a `BOOLEAN` cast, because comparing one as text is accepted by the server and matches nothing. Two limits come from Semantic Kernel itself, which re-parses the lambda's own source: the value must be a **literal** rather than a variable, and the call has to fit on one line. Only metadata fields can be filtered; the `content` field is the embedded body, not metadata.
-  **Pre-computed vectors not supported.** Passing `vector=` to `search()` raises the same exception. Pass text only.

## Project structure

```
goodmem-semantic-kernel/           ← repo root
├── python/
│   ├── goodmem_semantic_kernel/   ← importable Python package
│   │   ├── __init__.py        # Public exports: GoodMemCollection, GoodMemStore, GoodMemSettings
│   │   ├── _connection.py     # Owns (or borrows) the official goodmem SDK client
│   │   ├── _ids.py            # Refuses any id that is not a UUID before it is sent
│   │   ├── _results.py        # Retrieval statuses, chunk→memory join, score direction
│   │   ├── _typing.py         # Protocols for the SDK surface this package calls
│   │   ├── collection.py      # VectorStoreCollection + VectorSearch implementation
│   │   ├── filters.py         # Builds GoodMem filter expressions safely
│   │   ├── settings.py        # GoodMemSettings (Pydantic, reads GOODMEM_* env vars)
│   │   └── store.py           # VectorStore implementation
│   └── tests/                 # support.py + test_regressions.py + test_id_validation.py + test_e2e.py
├── dotnet/
│   └── GoodMem.SemanticKernel/    ← .NET connector library
│   └── GoodMem.SemanticKernel.Tests/
├── java/
│   ├── pom.xml                    ← parent Maven POM
│   └── goodmem-semantic-kernel/   ← Java connector library
│       └── src/main/java/ai/goodmem/semantickernel/
│           ├── GoodMemCollection.java   # Typed CRUD + semantic search (Reactive)
│           ├── GoodMemVectorStore.java  # Factory for multiple collections
│           ├── GoodMemPlugin.java       # SK KernelPlugin: save + recall functions
│           ├── GoodMemSchema.java       # Reflection engine for @GoodMemKey/@GoodMemData
│           ├── GoodMemKey.java          # Annotation: marks the memory ID field
│           ├── GoodMemData.java         # Annotation: marks content/metadata fields
│           ├── GoodMemClient.java       # Async HTTP client (GoodMem REST API)
│           ├── GoodMemOptions.java      # Configuration (reads GOODMEM_* env vars)
│           ├── GoodMemIds.java          # Refuses any id that is not a UUID before it is sent
│           └── GoodMemException.java    # Runtime exception wrapper
├── samples/
│   ├── python/                    ← Runnable Python samples
│   ├── dotnet/                    ← Runnable .NET samples
│   └── java/                      ← Runnable Java samples
└── pyproject.toml
```

## API reference

### `GoodMemCollection`

The core class. Implements `VectorStoreCollection[str, TModel]` and `VectorSearch[str, TModel]`.

```python
GoodMemCollection(
    record_type=MyModel,
    collection_name="my-space",    # maps to a GoodMem Space
    settings=GoodMemSettings(),    # optional; reads GOODMEM_* env vars by default
    client=None,                   # optional; inject a pre-built GoodMemAsyncClient
)
```

| Method | Description |
|---|---|
| `ensure_collection_exists()` | Create the GoodMem space if it doesn't exist |
| `ensure_collection_deleted()` | Delete the space and all its memories |
| `collection_exists()` | Return `True` if the space exists |
| `upsert(records)` | Write one or a list of records; returns the memory ID(s) |
| `get(key=...)` / `get(keys=[...])` | Fetch memories by ID (each a UUID) |
| `delete(keys=[...])` | Delete memories by ID (each a UUID) |
| `search(query, top=5)` | Semantic search; returns `KernelSearchResults` |
| `create_search_function(...)` | Wrap search as a `KernelFunction` for use in agent plugins |

### `GoodMemStore`

Factory for collections. All collections from the same store share one HTTP connection.

```python
GoodMemStore(settings=GoodMemSettings())
```

| Method | Description |
|---|---|
| `get_collection(record_type, collection_name=...)` | Return a `GoodMemCollection` |
| `list_collection_names()` | List all GoodMem spaces visible to this API key |
| `ensure_collection_deleted(collection_name)` | Delete the named space, if it exists. A refused delete raises instead of passing silently |

### `GoodMemSettings`

Pydantic settings class; reads `GOODMEM_*` environment variables.

```python
GoodMemSettings(
    base_url="https://localhost:8080",
    api_key="your_key_here",
    embedder_id="your_embedder_uuid",  # required to create a space
    reranker_id=None,
    verify_ssl=True,
    timeout=30.0,
    wait_for_indexing=True,
    indexing_timeout=60.0,
)
```
