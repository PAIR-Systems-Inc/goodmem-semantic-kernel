// ExampleStore — GoodMemVectorStore managing multiple collections.
//
// A single store owns one shared HTTP connection used across all collections.
// This mirrors the pattern where you manage related spaces from one place.
//
// Required environment variables:
//   GOODMEM_BASE_URL   — GoodMem server URL  (default: http://localhost:8080)
//   GOODMEM_VERIFY_SSL — Set to 'false' for self-signed certs
//   GOODMEM_API_KEY    — GoodMem API key

using GoodMem.SemanticKernel;
using Microsoft.Extensions.VectorData;

// ── Validate required environment variables ───────────────────────────────────
foreach (var v in new[] { "GOODMEM_API_KEY", "GOODMEM_BASE_URL" })
    if (string.IsNullOrEmpty(Environment.GetEnvironmentVariable(v)))
        throw new InvalidOperationException($"Set {v} before running this example.");

// ── Main ──────────────────────────────────────────────────────────────────────
using var store = new GoodMemVectorStore();

// 1. List all spaces currently visible to this API key.
var names = new List<string>();
await foreach (var name in store.ListCollectionNamesAsync())
    names.Add(name);
Console.WriteLine($"Existing spaces: {(names.Count > 0 ? string.Join(", ", names) : "(none)")}");

// 2. Get two collections from the same store (shared HTTP connection).
var notes = store.GetCollection<string, Note>("store-notes");
var todos = store.GetCollection<string, Note>("store-todos");

// 3. Fresh slate for both.
await notes.EnsureCollectionDeletedAsync();
await notes.EnsureCollectionExistsAsync();
await todos.EnsureCollectionDeletedAsync();
await todos.EnsureCollectionExistsAsync();
Console.WriteLine("Both collections ready (fresh).");

// 4. Write into each.
await notes.UpsertAsync(
[
    new Note { Content = "The Eiffel Tower is in Paris", Source = "facts" },
    new Note { Content = "Mount Fuji is in Japan", Source = "facts" },
]);
await todos.UpsertAsync(
[
    new Note { Content = "Buy groceries", Source = "chat" },
    new Note { Content = "Call the dentist", Source = "chat" },
]);
Console.WriteLine("Upserted into both collections.");

// 5. Wait for server-side embeddings.
Console.WriteLine("Waiting for embeddings...");
await Task.Delay(TimeSpan.FromSeconds(3));

// 6. Search each independently.
Console.WriteLine("\n--- notes search: 'famous landmarks in europe' ---");
await foreach (var r in notes.SearchAsync("famous landmarks in europe", top: 3))
    Console.WriteLine($"  [{r.Score:F3}] {r.Record.Content}");

Console.WriteLine("\n--- todos search: 'health appointments' ---");
await foreach (var r in todos.SearchAsync("health appointments", top: 3))
    Console.WriteLine($"  [{r.Score:F3}] {r.Record.Content}");

// Uncomment to clean up:
// await notes.EnsureCollectionDeletedAsync();
// await todos.EnsureCollectionDeletedAsync();
// Console.WriteLine("\nCollections deleted.");

// ── Data model ────────────────────────────────────────────────────────────────
public class Note
{
    [VectorStoreKey] public string? Id { get; set; }
    [VectorStoreData] public string Content { get; set; } = "";
    [VectorStoreData] public string? Source { get; set; }
}
