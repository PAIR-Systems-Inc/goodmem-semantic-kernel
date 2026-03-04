# goodmem-semantic-kernel

A [GoodMem](https://goodmem.ai) connector for [Microsoft Semantic Kernel](https://github.com/microsoft/semantic-kernel).

Implements Semantic Kernel's `VectorStoreCollection` and `VectorStore` interfaces so agents built on Semantic Kernel can store and retrieve memories from a GoodMem server — no local embedding model required.

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

## Installation

**Requirements:** Python 3.10+ and a running GoodMem server.

```bash
pip install goodmem-semantic-kernel
```

Or from source (note the hyphens, rather than underscores):

```bash
git clone https://github.com/PAIR-Systems-Inc/goodmem-semantic-kernel
pip install -e goodmem-semantic-kernel
# alternative methods if facing issues
# pip install -e ".[dev]"
# pip install -e .
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
| `GOODMEM_EMBEDDER_ID` | No | auto-detected | UUID of the embedder to use |
| `GOODMEM_VERIFY_SSL` | No | `false` | Set to `false` for self-signed certs, set to `true` if you setup custom TLS certs|

## Quickstart

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

- Exactly one `"key"` field (the memory ID — `None` lets the server generate a UUID).
- One `"data"` field named `content` becomes the embedded text (`originalContent` in GoodMem).
- All other `"data"` fields are stored as metadata and returned on search results.
- `"vector"` fields are accepted for interface compatibility but ignored — GoodMem embeds server-side.

We have three example patterns provided.

Option A is the recommended pattern for production agents since the LLM decides when to call memory and what to search for, rather than the application hardcoding those decisions.

### Option A — Wired into a Semantic Kernel agent

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
        result = await agent.get_response(messages="Where is the Eiffel Tower?", thread=thread)
        print(result.content)
```

### Option B — Single collection

```python
import asyncio
from goodmem_semantic_kernel import GoodMemCollection

async def main():
    async with GoodMemCollection(record_type=Note, collection_name="my-notes") as coll:
        await coll.ensure_collection_exists()

        keys = await coll.upsert([
            Note(content="The Eiffel Tower is in Paris", source="facts"),
            Note(content="Buy milk and eggs", source="shopping"),
        ])

        await asyncio.sleep(3)  # wait for server-side embedding

        results = await coll.search("grocery list", top=5)
        async for r in results.results:
            print(f"[{r.score:.3f}] {r.record.content}  (source={r.record.source})")

asyncio.run(main())
```

### Option C — Store (multiple collections, shared connection)

```python
from goodmem_semantic_kernel import GoodMemStore

async def main():
    async with GoodMemStore() as store:
        print(await store.list_collection_names())

        notes = store.get_collection(Note, collection_name="notes")
        todos = store.get_collection(Note, collection_name="todos")

        await notes.ensure_collection_exists()
        await todos.ensure_collection_exists()

        await notes.upsert(Note(content="Mount Fuji is in Japan"))
        await todos.upsert(Note(content="Call the dentist"))
```

## Running the examples

```bash
cd goodmem_semantic_kernel/examples/python

# Option A — agent with memory tool (also requires OPENAI_API_KEY)
OPENAI_API_KEY=your_openai_key_here
python example_agent.py

# Option B — single collection
python example_single_collection.py

# Option C — store with multiple collections
python example_store.py
```

if you failed to run an example successfully, double check [Configuration](#configuration)

or try running inside a python virtual environment if there is a dependency conflict

```bash
python3 -m venv venv
source venv/bin/activate
```

## Testing

```bash
# Unit tests (no server required)
pytest tests/unit/

# Integration tests (requires a live GoodMem server)
GOODMEM_API_KEY=your_key_here pytest -m integration
```

## Behavior notes

**No local embedding.** Never pass an `embedding_generator` — GoodMem embeds content server-side. The parameter is accepted for interface compatibility and silently ignored.

**Upsert semantics.** GoodMem memories are immutable. If you `upsert` a record with an existing `id`, the connector deletes the old memory and creates a new one.

**`content` is write-only in GoodMem.** The server does not return `originalContent` in search responses. Retrieved text comes from `chunkText` (a chunk of the original), which the connector maps back to your `content` field transparently.

**Score convention.** `relevanceScore` from the GoodMem API is a raw pgvector value where lower means more similar. The connector negates it before returning, so SK's standard convention (higher = more relevant) is preserved.

**Filters not supported.** Passing `filter=` to `search()` raises `VectorStoreOperationNotSupportedException`. Post-filter results in application code if needed.

**Pre-computed vectors not supported.** Passing `vector=` to `search()` raises the same exception. Pass text only.

## Project structure

```
goodmem-semantic-kernel/           ← repo root
├── goodmem_semantic_kernel/       ← importable package
│   ├── __init__.py        # Public exports: GoodMemCollection, GoodMemStore, GoodMemSettings
│   ├── _client.py         # Async HTTP wrapper around the GoodMem REST API
│   ├── collection.py      # VectorStoreCollection + VectorSearch implementation
│   ├── settings.py        # GoodMemSettings (Pydantic, reads GOODMEM_* env vars)
│   ├── store.py           # VectorStore implementation
├── tests/
│   ├── unit/              # Mocked unit tests (no server required)
│   └── integration/       # Live integration tests (require GoodMem server)
├── examples/              # Runnable examples (Options A, B, C)
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
| `get(key=...)` / `get(keys=[...])` | Fetch memories by ID |
| `delete(keys=[...])` | Delete memories by ID |
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

### `GoodMemSettings`

Pydantic settings class; reads `GOODMEM_*` environment variables.

```python
GoodMemSettings(
    base_url="https://localhost:8080",
    api_key="your_key_here",
    embedder_id=None,   # auto-detected if omitted
    verify_ssl=false,
)
```
