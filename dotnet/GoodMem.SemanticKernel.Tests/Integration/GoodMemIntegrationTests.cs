using Microsoft.Extensions.VectorData;
using Xunit;

namespace GoodMem.SemanticKernel.Tests.Integration;

/// <summary>
/// [IntegrationFact] is a drop-in for [Fact] that auto-skips when GOODMEM_API_KEY is not set.
/// </summary>
file sealed class IntegrationFactAttribute : FactAttribute
{
    public IntegrationFactAttribute()
    {
        if (string.IsNullOrEmpty(Environment.GetEnvironmentVariable("GOODMEM_API_KEY")))
            Skip = "GOODMEM_API_KEY not set — integration tests require a live GoodMem server.";
    }
}

/// <summary>
/// End-to-end integration tests against a live GoodMem server.
///
/// Run with:
///   GOODMEM_API_KEY=your_key GOODMEM_BASE_URL=http://your_server:8080 \
///   dotnet test --filter Category=Integration
/// </summary>
[Trait("Category", "Integration")]
public sealed class GoodMemIntegrationTests
{
    // ── Data model ────────────────────────────────────────────────────────────

    private class IntegrationMemory
    {
        [VectorStoreKey] public string? Id { get; set; }
        [VectorStoreData] public string Content { get; set; } = "";
        [VectorStoreData] public string? Tag { get; set; }
    }

    // ── Tests ─────────────────────────────────────────────────────────────────

    /// <summary>Full round-trip: create space → upsert → search → verify → cleanup.</summary>
    [IntegrationFact]
    public async Task FullRoundTrip_UpsertSearchDelete()
    {
        const string collectionName = "sk-dotnet-integration-test";

        using var collection = new GoodMemCollection<IntegrationMemory>(collectionName);

        await collection.EnsureCollectionDeletedAsync();
        await collection.EnsureCollectionExistsAsync();

        try
        {
            // Upsert seed data.
            var memories = new[]
            {
                new IntegrationMemory { Content = "The Eiffel Tower is in Paris, France.", Tag = "landmarks" },
                new IntegrationMemory { Content = "Mount Fuji is a volcano in Japan.", Tag = "geography" },
                new IntegrationMemory { Content = "The Amazon river is the largest river by discharge.", Tag = "geography" },
            };
            await collection.UpsertAsync(memories);

            // All records should have IDs assigned by the server.
            Assert.All(memories, m => Assert.NotNull(m.Id));

            // Wait for server-side embedding (GoodMem is async).
            await Task.Delay(TimeSpan.FromSeconds(5));

            // Search should return relevant results.
            var results = new List<VectorSearchResult<IntegrationMemory>>();
            await foreach (var r in collection.SearchAsync("famous towers in Europe", top: 3))
                results.Add(r);

            Assert.NotEmpty(results);
            Assert.Contains(results, r => r.Record.Content.Contains("Eiffel"));

            // Scores should be non-negative (connector negates GoodMem's lower-is-better values).
            Assert.All(results, r => Assert.True(r.Score >= 0));

            // Get by key should return the upserted record.
            var fetched = await collection.GetAsync(memories[0].Id!);
            Assert.NotNull(fetched);

            // Delete a record and confirm it no longer appears in batchGet.
            await collection.DeleteAsync(memories[0].Id!);
            var deleted = await collection.GetAsync(memories[0].Id!);
            Assert.Null(deleted);
        }
        finally
        {
            await collection.EnsureCollectionDeletedAsync();
        }
    }

    /// <summary>CollectionExistsAsync returns false before creation, true after.</summary>
    [IntegrationFact]
    public async Task CollectionExists_FalseBeforeCreate_TrueAfter()
    {
        const string collectionName = "sk-dotnet-exists-test";
        using var collection = new GoodMemCollection<IntegrationMemory>(collectionName);

        await collection.EnsureCollectionDeletedAsync();

        try
        {
            Assert.False(await collection.CollectionExistsAsync());
            await collection.EnsureCollectionExistsAsync();
            Assert.True(await collection.CollectionExistsAsync());
        }
        finally
        {
            await collection.EnsureCollectionDeletedAsync();
        }
    }

    /// <summary>GoodMemVectorStore lists collection names including freshly created ones.</summary>
    [IntegrationFact]
    public async Task VectorStore_ListCollectionNames_IncludesCreatedSpace()
    {
        const string collectionName = "sk-dotnet-list-test";
        using var store = new GoodMemVectorStore();
        using var collection = store.GetCollection<string, IntegrationMemory>(collectionName);

        await collection.EnsureCollectionDeletedAsync();

        try
        {
            await collection.EnsureCollectionExistsAsync();

            var names = new List<string>();
            await foreach (var name in store.ListCollectionNamesAsync())
                names.Add(name);

            Assert.Contains(collectionName, names);
        }
        finally
        {
            await collection.EnsureCollectionDeletedAsync();
        }
    }
}
