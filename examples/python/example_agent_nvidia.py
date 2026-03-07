#!/usr/bin/env python3
"""NVIDIA NIM variant of example_agent.py — GoodMem collection wired into a Semantic Kernel agent.

The agent has a memory search tool backed by GoodMem. When the user asks
a question, the LLM decides whether to call the tool to look up relevant
memories before composing its answer.

Note: SK's NvidiaChatCompletion sets SUPPORTS_FUNCTION_CALLING=False, so it
cannot be used with ChatCompletionAgent and tool/function calling. Instead, we
use NVIDIA's OpenAI-compatible NIM endpoint via OpenAIChatCompletion with a
custom AsyncOpenAI client, which does support tool calling. You can get your
NVIDIA API key and find available models at https://build.nvidia.com

Requirements (in addition to goodmem-semantic-kernel):
    pip install semantic-kernel openai

Environment variables:
    GOODMEM_BASE_URL    — GoodMem server URL  (default: https://localhost:8080)
    GOODMEM_VERIFY_SSL  — Set to 'false' for self-signed certs
    GOODMEM_API_KEY     — GoodMem API key
    NVIDIA_API_KEY      — NVIDIA API key (https://build.nvidia.com)
    NVIDIA_MODEL        — NVIDIA NIM model ID (optional, default: meta/llama-3.1-8b-instruct)
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


_NVIDIA_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
# note: model must support tool/function calling — check https://build.nvidia.com for availability
_DEFAULT_NVIDIA_MODEL = "meta/llama-3.1-8b-instruct"

# Data model - id is key, content is text to store in GoodMem, topic is for easier searching
@vectorstoremodel
@dataclass
class Memory:
    id: Annotated[str | None, VectorStoreField("key")] = None
    content: Annotated[str, VectorStoreField("data", type="str")] = ""
    topic: Annotated[str | None, VectorStoreField("data")] = None

async def main() -> None:
    for var in ("GOODMEM_API_KEY", "GOODMEM_BASE_URL", "GOODMEM_VERIFY_SSL", "NVIDIA_API_KEY"):
        if not os.getenv(var):
            raise SystemExit(f"Set {var} before running this script.")

    nvidia_model = os.getenv("NVIDIA_MODEL", _DEFAULT_NVIDIA_MODEL)

    async with GoodMemCollection(record_type=Memory, collection_name="agent-memory") as collection:
        # 1. Fresh space with seed data
        await collection.ensure_collection_deleted()
        await collection.ensure_collection_exists()
        await collection.upsert([
            Memory(content="The Pacific Ocean is the largest ocean on Earth.", topic="geography"),
            Memory(content="Python was created by Guido van Rossum and first released in 1991.", topic="technology"),
            Memory(content="The speed of light is approximately 299,792 km/s.", topic="science"),
            Memory(content="Shakespeare wrote Hamlet, Macbeth, and Romeo and Juliet.", topic="literature"),
            Memory(content="Semantic Kernel is a Microsoft SDK for building AI agents.", topic="technology"),
        ])
        print(f"Seeded 5 memories into the 'agent-memory' GoodMem space.")
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
                        "Call this before answering questions about science, technology, geography, or literature."
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
                    string_mapper=lambda r: r.record.content,
                ),
            ],
        )

        # 3. Build the agent using NVIDIA NIM (OpenAI-compatible endpoint).
        #    We point AsyncOpenAI at NVIDIA's NIM base URL and pass NVIDIA_API_KEY as the key.
        nvidia_client = AsyncOpenAI(
            base_url=_NVIDIA_NIM_BASE_URL,
            api_key=os.getenv("NVIDIA_API_KEY"),
        )
        agent = ChatCompletionAgent(
            name="MemoryAgent",
            service=OpenAIChatCompletion(ai_model_id=nvidia_model, async_client=nvidia_client),
            instructions=(
                "You are a helpful assistant with access to a long-term memory store. "
                "Always search memory before answering factual questions. "
                "Cite what you found in memory when it is relevant."
            ),
            function_choice_behavior=FunctionChoiceBehavior.Auto(),
            plugins=[memory_plugin],
        )

        # 4. Interactive chat loop
        print(f"\nMemory agent ready (model: {nvidia_model}). Type 'exit' to quit.\n")
        thread: AgentThread | None = None
        while True:
            try:
                user_input = input("You: ").strip()
            except (KeyboardInterrupt, EOFError):
                break

            if not user_input or user_input.lower() == "exit":
                break

            result = await agent.get_response(messages=user_input, thread=thread)
            thread = result.thread
            print(f"Agent: {result.content}\n")


if __name__ == "__main__":
    asyncio.run(main())
