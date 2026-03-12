// ExampleAgentAzure — GoodMem collection wired into a Semantic Kernel agent (Azure OpenAI).
//
// Functionally identical to ExampleAgent, but uses Azure OpenAI as the LLM provider.
//
// Required environment variables:
//   GOODMEM_BASE_URL                   — GoodMem server URL  (default: http://localhost:8080)
//   GOODMEM_VERIFY_SSL                 — Set to 'false' for self-signed certs
//   GOODMEM_API_KEY                    — GoodMem API key
//   AZURE_OPENAI_API_KEY               — Azure OpenAI API key
//   AZURE_OPENAI_ENDPOINT              — Azure OpenAI endpoint (e.g. https://my-resource.openai.azure.com)
//   AZURE_OPENAI_CHAT_DEPLOYMENT_NAME  — Chat model deployment name (e.g. gpt-4o-mini)
//   AZURE_OPENAI_API_VERSION           — (optional) API version override

using System.ComponentModel;
using GoodMem.SemanticKernel;
using Microsoft.Extensions.VectorData;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Agents;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.AzureOpenAI;

// ── Validate required environment variables ───────────────────────────────────
foreach (var v in new[]
{
    "GOODMEM_API_KEY", "GOODMEM_BASE_URL",
    "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME",
})
    if (string.IsNullOrEmpty(Environment.GetEnvironmentVariable(v)))
        throw new InvalidOperationException($"Set {v} before running this example.");

var deployment = Environment.GetEnvironmentVariable("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")!;
var endpoint   = Environment.GetEnvironmentVariable("AZURE_OPENAI_ENDPOINT")!;
var apiKey     = Environment.GetEnvironmentVariable("AZURE_OPENAI_API_KEY")!;
var apiVersion = Environment.GetEnvironmentVariable("AZURE_OPENAI_API_VERSION"); // optional

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

Console.WriteLine("Waiting for embeddings...");
await Task.Delay(TimeSpan.FromSeconds(3));

// 2. Build the Kernel with Azure OpenAI chat completion.
//    AzureChatCompletion reads the deployment name, endpoint, and API key from the constructor.
var builder = Kernel.CreateBuilder();

if (apiVersion is not null)
    builder.AddAzureOpenAIChatCompletion(deployment, endpoint, apiKey, apiVersion: apiVersion);
else
    builder.AddAzureOpenAIChatCompletion(deployment, endpoint, apiKey);

var kernel = builder.Build();

// 3. Register the GoodMem search function as a kernel plugin.
kernel.Plugins.AddFromObject(new MemoryPlugin(collection), pluginName: "memory");

// 4. Build the agent.
var agent = new ChatCompletionAgent
{
    Name = "MemoryAgent",
    Instructions =
        "You are a helpful assistant with access to a long-term memory store. " +
        "Always search memory before answering factual questions. " +
        "Cite what you found in memory when it is relevant.",
    Kernel = kernel,
    Arguments = new KernelArguments(
        new AzureOpenAIPromptExecutionSettings { FunctionChoiceBehavior = FunctionChoiceBehavior.Auto() }),
};

// 5. Interactive chat loop.
Console.WriteLine("\nMemory agent ready. Type 'exit' to quit.\n");
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
public class Memory
{
    [VectorStoreKey] public string? Id { get; set; }
    [VectorStoreData] public string Content { get; set; } = "";
    [VectorStoreData] public string? Source { get; set; }
}

// ── Plugin class ──────────────────────────────────────────────────────────────
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
