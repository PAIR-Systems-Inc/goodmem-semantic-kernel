#!/usr/bin/env python3
"""Option A example (simplified) — GoodMem agent using collection.as_plugin().

This is a simpler version of example_agent.py. Instead of manually
constructing a KernelPlugin with KernelParameterMetadata and
create_search_function, it uses the GoodMemCollection.as_plugin()
convenience method which handles all of that with sensible defaults.

See example_agent.py for the explicit version that gives you full control
over the plugin name, function description, parameter metadata, and how
each result is formatted for the LLM.

Requirement (in addition to goodmem-semantic-kernel):
    pip install semantic-kernel openai

Environment variables:
    GOODMEM_BASE_URL    — GoodMem server URL  (default: http://localhost:8080)
    GOODMEM_VERIFY_SSL  — Set to 'false' for self-signed certs
    GOODMEM_API_KEY     — GoodMem API key
    OPENAI_API_KEY      — OpenAI key (used by the chat LLM)
"""

import asyncio
import os
from dataclasses import dataclass
from typing import Annotated
import openai
from semantic_kernel.agents import AgentThread, ChatCompletionAgent
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.data.vector import VectorStoreField, vectorstoremodel
from goodmem_semantic_kernel import GoodMemCollection

# Data model - id is key, content is text to store in GoodMem, topic is for easier searching
@vectorstoremodel
@dataclass
class Memory:
    id: Annotated[str | None, VectorStoreField("key")] = None
    content: Annotated[str, VectorStoreField("data", type="str")] = ""
    topic: Annotated[str | None, VectorStoreField("data")] = None

async def main() -> None:
    for var in ("GOODMEM_API_KEY", "OPENAI_API_KEY", "GOODMEM_BASE_URL", "GOODMEM_VERIFY_SSL"):
        if not os.environ.get(var):
            raise SystemExit(f"Set {var} before running this script.")

    # recommended models:
    # gpt-4o-mini (recommended - cheap and fast)
    # gpt-4o (general purpose)
    # gpt-3.5-turbo (legacy, widely supported)
    openai_model = input("Enter the OpenAI model to use (recommended: gpt-4o-mini): ").strip()
    try:
        async with openai.AsyncOpenAI() as client:
            await client.models.retrieve(openai_model)
    except openai.AuthenticationError:
        raise SystemExit("OPENAI_API_KEY is invalid or expired.")
    except (openai.NotFoundError, openai.PermissionDeniedError) as exc:
        raise SystemExit(f"Model {openai_model!r} is not available: {exc.message}")

    async with GoodMemCollection(record_type=Memory, collection_name="agent-memory") as collection:
        # 1. Fresh space with seed data
        await collection.ensure_collection_deleted()
        await collection.ensure_collection_exists()
        await collection.upsert([
            Memory(content="The Pacific Ocean is the largest ocean on Earth.", topic="geography"),
            Memory(content="Python was created by Guido van Rossum and first released in 1991.", topic="technology"),
            Memory(content="The speed of light is approximately 299,792 km/s.", topic="science"),
            Memory(content="Shakespeare wrote Hamlet, Macbeth, and Romeo and Juliet.", topic="literature"),
            Memory(content="Semantic Kernel is a Microsoft SDK for building AI agents.", topic="technology")
        ])
        print(f"Seeded 5 memories into the 'agent-memory' GoodMem space.")
        print("Waiting for embeddings...")
        await asyncio.sleep(3)

        # 2. Build the agent — as_plugin() handles all the SK plugin wiring
        agent = ChatCompletionAgent(
            name="MemoryAgent",
            service=OpenAIChatCompletion(ai_model_id=openai_model),
            instructions=(
                "You are a helpful assistant with access to a long-term memory store. "
                "Always search memory before answering factual questions. "
                "Cite what you found in memory when it is relevant."
            ),
            function_choice_behavior=FunctionChoiceBehavior.Auto(),
            plugins=[collection.as_plugin(name="memory")],
        )

        # 3. Interactive chat loop
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
