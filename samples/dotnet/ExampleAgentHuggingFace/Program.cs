// ExampleAgentHuggingFace — GoodMem collection wired into a Semantic Kernel agent (Hugging Face).
//
// The agent has a memory search tool backed by GoodMem. When the user asks a
// question, the LLM decides whether to call the tool to look up relevant
// memories before composing its answer.
//
// Note: SK's built-in HuggingFace connector does not implement chat completion
// with function calling. This sample uses HuggingFace's OpenAI-compatible
// Inference API endpoint instead, via Microsoft.SemanticKernel.Connectors.OpenAI
// with a custom endpoint URI. Not all HuggingFace models support tool calling —
// instruction-tuned models (e.g. Llama-3.1-8B-Instruct) work best.
//
// Required environment variables:
//   GOODMEM_BASE_URL      — GoodMem server URL  (default: http://localhost:8080)
//   GOODMEM_VERIFY_SSL    — Set to 'false' for self-signed certs
//   GOODMEM_API_KEY       — GoodMem API key
//   HF_TOKEN              — Hugging Face API token (https://huggingface.co/settings/tokens)
//
// Optional environment variables:
//   HF_MODEL              — Hugging Face model ID (default: meta-llama/Llama-3.1-8B-Instruct)

using System.ComponentModel;
using GoodMem.SemanticKernel;
using Microsoft.Extensions.VectorData;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Agents;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.OpenAI;

const string HuggingFaceInferenceBaseUrl = "https://router.huggingface.co/v1/";
const string DefaultHfModel = "meta-llama/Llama-3.1-8B-Instruct";

// ── Validate required environment variables ───────────────────────────────────
foreach (var v in new[] { "GOODMEM_API_KEY", "GOODMEM_BASE_URL", "HF_TOKEN" })
    if (string.IsNullOrEmpty(Environment.GetEnvironmentVariable(v)))
        throw new InvalidOperationException($"Set {v} before running this example.");

var model = Environment.GetEnvironmentVariable("HF_MODEL") ?? DefaultHfModel;

// ── Main ──────────────────────────────────────────────────────────────────────
using var collection = new GoodMemCollection<Memory>("agent-memory");

// 1. Fresh space with seed data.
await collection.EnsureCollectionDeletedAsync();
await collection.EnsureCollectionExistsAsync();
await collection.UpsertAsync(
[
    new Memory { Content = "The Pacific Ocean is the largest ocean on Earth.", Source = "geography" },
    new Memory { Content = "Python was created by Guido van Rossum and first released in 1991.", Source = "technology" },
    new Memory { Content = "The speed of light is approximately 299,792 km/s.", Source = "science" },
    new Memory { Content = "Shakespeare wrote Hamlet, Macbeth, and Romeo and Juliet.", Source = "literature" },
    new Memory { Content = "Semantic Kernel is a Microsoft SDK for building AI agents.", Source = "technology" },
]);
Console.WriteLine("Seeded 5 memories into the 'agent-memory' GoodMem space.");

// GoodMem's embedding pipeline is asynchronous — wait a moment before searching.
Console.WriteLine("Waiting for embeddings...");
await Task.Delay(TimeSpan.FromSeconds(3));

// 2. Build the Kernel using HuggingFace's OpenAI-compatible Inference API.
//    HF_TOKEN is passed as the API key; the custom endpoint overrides the base URL.
var kernel = Kernel.CreateBuilder()
    .AddOpenAIChatCompletion(
        modelId: model,
        apiKey: Environment.GetEnvironmentVariable("HF_TOKEN")!,
        endpoint: new Uri(HuggingFaceInferenceBaseUrl))
    .Build();

// 3. Register the GoodMem search function as a kernel plugin.
kernel.Plugins.AddFromObject(new MemoryPlugin(collection), pluginName: "memory");

// 4. Build the agent.
//    FunctionChoiceBehavior.Auto() lets the LLM decide when to call memory.recall.
var agent = new ChatCompletionAgent
{
    Name = "MemoryAgent",
    Instructions =
        "You are a helpful assistant with access to a long-term memory store. " +
        "Always search memory before answering factual questions. " +
        "Cite what you found in memory when it is relevant.",
    Kernel = kernel,
    Arguments = new KernelArguments(
        new OpenAIPromptExecutionSettings { FunctionChoiceBehavior = FunctionChoiceBehavior.Auto() }),
};

// 5. Interactive chat loop.
Console.WriteLine($"\nMemory agent ready (model: {model}). Type 'exit' to quit.\n");
AgentThread? thread = null;

while (true)
{
    Console.Write("You: ");
    var input = Console.ReadLine()?.Trim();
    if (string.IsNullOrEmpty(input) || input.Equals("exit", StringComparison.OrdinalIgnoreCase))
        break;

    var userMessage = new ChatMessageContent(AuthorRole.User, input);

    Console.Write("Agent: ");
    await foreach (var response in agent.InvokeAsync(userMessage, thread))
    {
        thread = response.Thread;
        Console.Write(response.Message.Content);
    }
    Console.WriteLine();
}

// ── Data model ────────────────────────────────────────────────────────────────
// One property with [VectorStoreKey] maps to GoodMem's memoryId.
// The string [VectorStoreData] property named "Content" becomes originalContent.
// Additional [VectorStoreData] properties go into metadata.
// [VectorStoreVector] properties are optional — GoodMem embeds server-side.
public class Memory
{
    [VectorStoreKey] public string? Id { get; set; }
    [VectorStoreData] public string Content { get; set; } = "";
    [VectorStoreData] public string? Source { get; set; }
}

// ── Plugin class ─────────────────────────────────────────────────────────────
// Wraps GoodMemCollection as a KernelPlugin the LLM can call as a tool.
public sealed class MemoryPlugin(GoodMemCollection<Memory> collection)
{
    [KernelFunction]
    [Description(
        "Search long-term memory for facts relevant to a query. " +
        "Call this before answering questions about science, technology, geography, or literature.")]
    public async Task<string> RecallAsync(
        [Description("What to search for in memory.")] string query,
        [Description("Number of memories to retrieve (default 3).")] int top = 3)
    {
        var results = new List<string>();
        await foreach (var r in collection.SearchAsync(query, top))
            if (!string.IsNullOrWhiteSpace(r.Record.Content))
                results.Add(r.Record.Content);

        return results.Count > 0
            ? string.Join("\n", results)
            : "(no relevant memories found)";
    }
}
