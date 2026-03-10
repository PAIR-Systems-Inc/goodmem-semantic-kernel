#!/usr/bin/env python3
"""Azure OpenAI variant of example_agent.py — GoodMem collection wired into a Semantic Kernel agent.

The agent has a memory search tool backed by GoodMem. When the user asks
a question, the LLM decides whether to call the tool to look up relevant
memories before composing its answer.

Requirements (in addition to goodmem-semantic-kernel):
    pip install semantic-kernel openai azure-identity

Environment variables:
    GOODMEM_BASE_URL                  — GoodMem server URL  (default: https://localhost:8080)
    GOODMEM_VERIFY_SSL                — Set to 'false' for self-signed certs
    GOODMEM_API_KEY                   — GoodMem API key
    AZURE_OPENAI_API_KEY              — Azure OpenAI API key
    AZURE_OPENAI_ENDPOINT             — Azure OpenAI endpoint (e.g. https://my-resource.openai.azure.com)
    AZURE_OPENAI_CHAT_DEPLOYMENT_NAME — Deployment name for the chat model (e.g. gpt-4o-mini)
    AZURE_OPENAI_API_VERSION          — API version (optional, defaults to latest stable)
"""

import asyncio
import os
from dataclasses import dataclass
from typing import Annotated
from semantic_kernel.agents import AgentThread, ChatCompletionAgent
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.data.vector import VectorStoreField, vectorstoremodel
from semantic_kernel.functions import KernelParameterMetadata, KernelPlugin
from goodmem_semantic_kernel import GoodMemCollection

# Data model - id is key, content is text to store in GoodMem, topic is for easier searching
@vectorstoremodel
@dataclass
class Memory:
    id: Annotated[str | None, VectorStoreField("key")] = None
    content: Annotated[str, VectorStoreField("data", type="str")] = ""
    topic: Annotated[str | None, VectorStoreField("data")] = None

async def main() -> None:
    for var in ("GOODMEM_API_KEY", "GOODMEM_BASE_URL", "GOODMEM_VERIFY_SSL", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"):
        if not os.getenv(var):
            raise SystemExit(f"Set {var} before running this script.")

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

        # 3. Build the agent with AzureChatCompletion.
        #    AzureChatCompletion reads AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT
        #    from the environment automatically; we pass the deployment name explicitly.
        agent = ChatCompletionAgent(
            name="MemoryAgent",
            service=AzureChatCompletion(
                deployment_name=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
                **({"api_version": os.getenv("AZURE_OPENAI_API_VERSION")} if os.getenv("AZURE_OPENAI_API_VERSION") else {}),
                # if above line doesn't work you can try this:
                # api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            ),
            instructions=(
                "You are a helpful assistant with access to a long-term memory store. "
                "Always search memory before answering factual questions. "
                "Cite what you found in memory when it is relevant."
            ),
            function_choice_behavior=FunctionChoiceBehavior.Auto(),
            plugins=[memory_plugin],
        )

        # 4. Interactive chat loop
        print("\nMemory agent ready. Type 'exit' to quit.\n")
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
