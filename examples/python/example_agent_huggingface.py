#!/usr/bin/env python3
"""Option A example — GoodMem collection wired into a Semantic Kernel agent (HuggingFace).

The agent has a memory search tool backed by GoodMem.  When the user asks
a question, the LLM decides whether to call the tool to look up relevant
memories before composing its answer.

This is the HuggingFace variant of example_agent.py.  Compare the two files
side-by-side: everything from the GoodMem import onward is identical — only
the import and service= line differ.  That is intentional: GoodMem is
provider-agnostic; swapping the LLM requires changing exactly one line.

NOTE — why this uses OpenAIChatCompletion rather than HuggingFaceTextCompletion:
    SK's built-in HuggingFace connector (HuggingFaceTextCompletion) runs models
    locally via the transformers pipeline and implements TextCompletionClientBase,
    not ChatCompletionClientBase.  ChatCompletionAgent requires a chat interface,
    so the local connector cannot be used here directly.

    The HuggingFace Inference API (https://api-inference.huggingface.co/v1/) is
    OpenAI-compatible, so we point OpenAIChatCompletion at it instead.  The model
    runs on HuggingFace infrastructure; no local GPU is required.  Any model on
    the Hub that supports the Messages API works.

Requirements (in addition to goodmem-sk):
    pip install semantic-kernel openai

Environment variables:
    GOODMEM_BASE_URL    — GoodMem server URL  (default: http://localhost:8080)
    GOODMEM_VERIFY_SSL  — Set to 'false' for self-signed certs
    GOODMEM_API_KEY     — GoodMem API key
    HF_TOKEN            — HuggingFace API token (https://huggingface.co/settings/tokens)
    HF_MODEL            — HuggingFace model ID (default: meta-llama/Llama-3.3-70B-Instruct)
                          Must support the Messages API (chat/instruction-tuned models).

Usage:
    GOODMEM_BASE_URL=https://localhost:8080 \
    GOODMEM_VERIFY_SSL=false \
    GOODMEM_API_KEY=your-key \
    HF_TOKEN=hf_... \
    HF_MODEL=meta-llama/Llama-3.3-70B-Instruct \
    python example_agent_huggingface.py
"""

import asyncio
import os
from dataclasses import dataclass
from typing import Annotated

from openai import AsyncOpenAI

from semantic_kernel.agents import AgentThread, ChatCompletionAgent
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.data.vector import VectorStoreField, vectorstoremodel
from semantic_kernel.functions import KernelParameterMetadata, KernelPlugin

from goodmem_semantic_kernel import GoodMemCollection


_HF_INFERENCE_BASE_URL = "https://api-inference.huggingface.co/v1/"
_DEFAULT_HF_MODEL = "meta-llama/Llama-3.3-70B-Instruct"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@vectorstoremodel
@dataclass
class Memory:
    id: Annotated[str | None, VectorStoreField("key")] = None
    content: Annotated[str, VectorStoreField("data", type="str")] = ""
    topic: Annotated[str | None, VectorStoreField("data")] = None


# ---------------------------------------------------------------------------
# Seed data
# ---------------------------------------------------------------------------

SEED_MEMORIES = [
    Memory(content="The Eiffel Tower is located in Paris, France.", topic="geography"),
    Memory(content="Python was created by Guido van Rossum and first released in 1991.", topic="technology"),
    Memory(content="The speed of light is approximately 299,792 km/s.", topic="science"),
    Memory(content="Shakespeare wrote Hamlet, Macbeth, and Romeo and Juliet.", topic="literature"),
    Memory(content="The Pacific Ocean is the largest ocean on Earth.", topic="geography"),
    Memory(content="Semantic Kernel is a Microsoft SDK for building AI agents.", topic="technology"),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    for var in ("GOODMEM_API_KEY", "HF_TOKEN"):
        if not os.environ.get(var):
            raise SystemExit(f"Set {var} before running this script.")

    hf_model = os.environ.get("HF_MODEL", _DEFAULT_HF_MODEL)

    # Point OpenAIChatCompletion at HuggingFace's OpenAI-compatible Inference API.
    # The AsyncOpenAI client is passed in so SK does not try to read OPENAI_API_KEY.
    hf_client = AsyncOpenAI(
        base_url=_HF_INFERENCE_BASE_URL,
        api_key=os.environ["HF_TOKEN"],
    )
    service = OpenAIChatCompletion(ai_model_id=hf_model, async_client=hf_client)

    async with GoodMemCollection(record_type=Memory, collection_name="agent-memory") as collection:
        # 1. Fresh space with seed data
        await collection.ensure_collection_deleted()
        await collection.ensure_collection_exists()
        await collection.upsert(SEED_MEMORIES)
        print(f"Seeded {len(SEED_MEMORIES)} memories into 'agent-memory'.")

        print("Waiting for embeddings...")
        await asyncio.sleep(3)

        # 2. Turn the collection into a kernel plugin the agent can call.
        #    create_search_function() builds a callable that the LLM sees as a
        #    tool — it calls collection.search(query, top=N) when invoked.
        memory_plugin = KernelPlugin(
            name="memory",
            description="Long-term memory store. Search it to recall facts.",
            functions=[
                collection.create_search_function(
                    function_name="recall",
                    description=(
                        "Search long-term memory for facts relevant to a query. "
                        "Call this before answering questions about facts, geography, "
                        "science, technology, or literature."
                    ),
                    parameters=[
                        KernelParameterMetadata(
                            name="query",
                            description="What to search for in memory.",
                            type="str",
                            is_required=True,
                            type_object=str,
                        ),
                        KernelParameterMetadata(
                            name="top",
                            description="Number of memories to retrieve (default 3).",
                            type="int",
                            default_value=3,
                            type_object=int,
                        ),
                    ],
                    # Format each result as a single line shown to the LLM
                    string_mapper=lambda r: r.record.content,
                ),
            ],
        )

        # 3. Build the agent
        agent = ChatCompletionAgent(
            name="MemoryAgent",
            service=service,
            instructions=(
                "You are a helpful assistant with access to a long-term memory store. "
                "Always search memory before answering factual questions. "
                "Cite what you found in memory when it is relevant."
            ),
            function_choice_behavior=FunctionChoiceBehavior.Auto(),
            plugins=[memory_plugin],
        )

        # 4. Interactive chat loop
        print(f"\nMemory agent ready (model: {hf_model}). Type 'exit' to quit.\n")
        thread: AgentThread | None = None
        while True:
            try:
                user_input = input("You: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nBye!")
                break

            if not user_input or user_input.lower() == "exit":
                break

            result = await agent.get_response(messages=user_input, thread=thread)
            thread = result.thread
            print(f"Agent: {result.content}\n")

        # 5. Uncomment to clean up:
        # await collection.ensure_collection_deleted()
        # print("Memory space deleted.")


if __name__ == "__main__":
    asyncio.run(main())
