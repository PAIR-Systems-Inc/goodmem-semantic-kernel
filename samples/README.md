# Example files using GoodMem as a plugin for Semantic Kernel Agents/LLMs

### Note: Embeddings in GoodMem are asynchronous, so please keep this in mind when trying to retrieve memories from GoodMem

`python/example_agent.py` is the recommended approach.

It defines how you would like to store your data in GoodMem, then integrates it as a plugin through Semantic Kernel's functions for easy access.

The key connection is `collection.create_search_function()` > `KernelPlugin` > `plugins=[...]` on the agent.

This is similar to the idiomatic Semantic Kernel pattern shown in the [hotel sample](https://github.com/microsoft/semantic-kernel/blob/main/python/samples/concepts/memory/azure_ai_search_hotel_samples/README.md) in the [semantic-kernel repository](https://github.com/microsoft/semantic-kernel).

