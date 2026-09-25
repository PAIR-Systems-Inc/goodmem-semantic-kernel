using System.Net;
using System.Text;
using Microsoft.Extensions.VectorData;
using Xunit;

namespace GoodMem.SemanticKernel.Tests.Unit;

internal class Memory
{
    [VectorStoreKey] public string? Id { get; set; }
    [VectorStoreData] public string Content { get; set; } = "";
    [VectorStoreData] public string? Topic { get; set; }
}

public sealed class GoodMemCollectionTests : IDisposable
{
    // GoodMem ids are UUIDs and the connector refuses anything else, so the
    // keys, embedder and space ids these tests pass in are real UUIDs.
    private const string EmbedderId = "019cfd1c-c033-7517-b7de-f73941a0464b";
    private const string SpaceToDelete = "01a0d16b-bbcd-701c-bfb4-fa306021e078";
    private const string ExistingId = "019cfd1d-5a1e-7a4b-9c3e-2f6a1b0c0e01";
    private const string BrandNew = "019cfd1d-5a1e-7a4b-9c3e-2f6a1b0c0e02";
    private const string Mem1 = "019cfd1d-5a1e-7a4b-9c3e-2f6a1b0c0e03";
    private const string Mem2 = "019cfd1d-5a1e-7a4b-9c3e-2f6a1b0c0e04";
    private const string NoSuchId = "019cfd1d-5a1e-7a4b-9c3e-2f6a1b0c0e05";

    private readonly MockHttpMessageHandler _handler = new();
    private readonly GoodMemClient _client;
    private readonly GoodMemCollection<Memory> _collection;

    public GoodMemCollectionTests()
    {
        var httpClient = new HttpClient(_handler) { BaseAddress = new Uri("http://localhost:8080/") };
        _client = new GoodMemClient(httpClient, ownsClient: false);
        // EmbedderId set so ResolveEmbedderIdAsync never needs to call ListEmbedders.
        _collection = new GoodMemCollection<Memory>(
            "test-collection", _client, new GoodMemOptions { EmbedderId = EmbedderId });
    }

    public void Dispose()
    {
        _collection.Dispose();
        _client.Dispose();
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    private static string SpaceFoundJson(string name = "test-collection", string spaceId = "space-abc") =>
        $$"""{"spaces":[{"name":"{{name}}","spaceId":"{{spaceId}}"}]}""";

    private static string SpaceNotFoundJson() => """{"spaces":[]}""";

    private static string CreateSpaceJson(string spaceId = "space-new") =>
        $$"""{"spaceId":"{{spaceId}}","name":"test-collection"}""";

    private static string CreateMemoryJson(string memoryId = "mem-new") =>
        $$"""{"memoryId":"{{memoryId}}"}""";

    private static string BatchGetJson(string memoryId = "mem-1", string content = "hello", string topic = "t") =>
        $$$$"""{"results":[{"success":true,"memory":{"memoryId":"{{{{memoryId}}}}","originalContent":"{{{{content}}}}","metadata":{"Topic":"{{{{topic}}}}"}}}]}""";

    private static string EmptyBatchGetJson() => """{"results":[]}""";

    private static string NdjsonRetrieveBody(
        string chunkText, string memoryId, string topic, double rawScore) =>
        $$$$"""
        {"memoryDefinition":{"memoryId":"{{{{memoryId}}}}","metadata":{"Topic":"{{{{topic}}}}"}}}
        {"retrievedItem":{"chunk":{"chunk":{"chunkText":"{{{{chunkText}}}}","memoryId":"{{{{memoryId}}}}"},"relevanceScore":{{{{rawScore}}}},"memoryIndex":0}}}
        """;

    // ── CollectionExistsAsync ────────────────────────────────────────────────

    [Fact]
    public async Task CollectionExistsAsync_ReturnsTrue_WhenSpaceFound()
    {
        _handler.EnqueueOk(SpaceFoundJson());

        var exists = await _collection.CollectionExistsAsync();

        Assert.True(exists);
    }

    [Fact]
    public async Task CollectionExistsAsync_ReturnsFalse_WhenSpaceNotFound()
    {
        _handler.EnqueueOk(SpaceNotFoundJson());

        var exists = await _collection.CollectionExistsAsync();

        Assert.False(exists);
    }

    // ── EnsureCollectionExistsAsync ──────────────────────────────────────────

    [Fact]
    public async Task EnsureCollectionExistsAsync_UsesExistingSpace()
    {
        _handler.EnqueueOk(SpaceFoundJson(spaceId: "existing-space"));

        await _collection.EnsureCollectionExistsAsync();

        // Only one request made (list spaces) — no create call.
        Assert.Single(_handler.SentRequests);
        Assert.Contains("v1/spaces", _handler.SentRequests[0].RequestUri!.ToString());
    }

    [Fact]
    public async Task EnsureCollectionExistsAsync_CreatesSpace_WhenNotExists()
    {
        _handler.EnqueueOk(SpaceNotFoundJson());        // list → not found
        _handler.EnqueueOk(CreateSpaceJson());          // create space

        await _collection.EnsureCollectionExistsAsync();

        Assert.Equal(2, _handler.SentRequests.Count);
        Assert.Equal(HttpMethod.Post, _handler.SentRequests[1].Method);
    }

    // ── EnsureCollectionDeletedAsync ─────────────────────────────────────────

    [Fact]
    public async Task EnsureCollectionDeletedAsync_DeletesSpace_WhenExists()
    {
        _handler.EnqueueOk(SpaceFoundJson(spaceId: SpaceToDelete));
        _handler.EnqueueNoContent(); // delete response

        await _collection.EnsureCollectionDeletedAsync();

        Assert.Equal(2, _handler.SentRequests.Count);
        Assert.Equal(HttpMethod.Delete, _handler.SentRequests[1].Method);
        Assert.Contains(SpaceToDelete, _handler.SentRequests[1].RequestUri!.ToString());
    }

    [Fact]
    public async Task EnsureCollectionDeletedAsync_DoesNothing_WhenNotExists()
    {
        _handler.EnqueueOk(SpaceNotFoundJson());

        await _collection.EnsureCollectionDeletedAsync();

        Assert.Single(_handler.SentRequests); // only the list call
    }

    // ── UpsertAsync ──────────────────────────────────────────────────────────

    [Fact]
    public async Task UpsertAsync_NewRecord_CreatesMemory_WritesBackId()
    {
        _handler.EnqueueOk(SpaceFoundJson());           // ResolveSpaceId
        _handler.EnqueueOk(CreateMemoryJson("mem-99")); // CreateMemory

        var record = new Memory { Content = "hello" };
        await _collection.UpsertAsync(record);

        Assert.Equal("mem-99", record.Id); // server-generated id written back
        Assert.Equal(HttpMethod.Post, _handler.SentRequests[1].Method);
    }

    [Fact]
    public async Task UpsertAsync_ExistingRecord_ReadsItBeforeReplacingIt()
    {
        // The previous version has to be in hand before the delete: GoodMem has
        // no update endpoint, so a failed create after a delete would otherwise
        // destroy the record.
        _handler.EnqueueOk(SpaceFoundJson());                 // ResolveSpaceId
        _handler.EnqueueOk(BatchGetJson(ExistingId));      // read the current version
        _handler.EnqueueNoContent();                          // DeleteMemory
        _handler.EnqueueOk(CreateMemoryJson(ExistingId));  // CreateMemory

        var record = new Memory { Id = ExistingId, Content = "updated" };
        await _collection.UpsertAsync(record);

        Assert.Equal(4, _handler.SentRequests.Count);
        Assert.EndsWith("memories:batchGet", _handler.SentRequests[1].RequestUri!.AbsolutePath);
        Assert.Equal(HttpMethod.Delete, _handler.SentRequests[2].Method);
        Assert.Equal(HttpMethod.Post, _handler.SentRequests[3].Method);
    }

    [Fact]
    public async Task UpsertAsync_FailedUpdate_RestoresThePreviousVersion()
    {
        _handler.EnqueueOk(SpaceFoundJson());                        // ResolveSpaceId
        _handler.EnqueueOk(BatchGetJson(ExistingId, "aGVsbG8=")); // "hello", base64
        _handler.EnqueueNoContent();                                 // DeleteMemory
        _handler.EnqueueBadRequest();                                // CreateMemory fails
        _handler.EnqueueOk(CreateMemoryJson(ExistingId));         // restore succeeds

        var record = new Memory { Id = ExistingId, Content = "" };

        var error = await Assert.ThrowsAsync<GoodMemUpsertException>(
            () => _collection.UpsertAsync(record));

        Assert.True(error.Restored);
        Assert.Null(error.LostKey);
        Assert.Contains("was restored", error.Message);
        Assert.Equal(5, _handler.SentRequests.Count);
    }

    [Fact]
    public async Task UpsertAsync_FailedUpdateAndFailedRestore_NamesTheLostRecord()
    {
        _handler.EnqueueOk(SpaceFoundJson());
        _handler.EnqueueOk(BatchGetJson(ExistingId, "aGVsbG8="));
        _handler.EnqueueNoContent();
        _handler.EnqueueBadRequest();                 // CreateMemory fails
        _handler.EnqueueBadRequest();                 // and so does the restore

        var record = new Memory { Id = ExistingId, Content = "" };

        var error = await Assert.ThrowsAsync<GoodMemUpsertException>(
            () => _collection.UpsertAsync(record));

        Assert.False(error.Restored);
        Assert.Equal(ExistingId, error.LostKey);
        Assert.Contains("could NOT be restored", error.Message);
    }

    [Fact]
    public async Task UpsertAsync_NewRecordWithAKey_IsNeverDeletedFirst()
    {
        _handler.EnqueueOk(SpaceFoundJson());
        _handler.EnqueueOk(EmptyBatchGetJson());             // nothing to replace
        _handler.EnqueueOk(CreateMemoryJson(BrandNew));

        await _collection.UpsertAsync(new Memory { Id = BrandNew, Content = "hi" });

        Assert.DoesNotContain(_handler.SentRequests, r => r.Method == HttpMethod.Delete);
    }

    [Fact]
    public async Task UpsertAsync_Batch_WritesBackIdsToEachRecord()
    {
        _handler.EnqueueOk(SpaceFoundJson()); // ResolveSpaceId
        _handler.EnqueueOk(CreateMemoryJson("id-a"));
        _handler.EnqueueOk(CreateMemoryJson("id-b"));

        var r1 = new Memory { Content = "first" };
        var r2 = new Memory { Content = "second" };
        await _collection.UpsertAsync(new[] { r1, r2 });

        Assert.Equal("id-a", r1.Id);
        Assert.Equal("id-b", r2.Id);
    }

    // ── GetAsync ─────────────────────────────────────────────────────────────

    [Fact]
    public async Task GetAsync_SingleKey_ReturnsDeserializedRecord()
    {
        _handler.EnqueueOk(BatchGetJson(Mem1, "batch content", "geo"));

        var record = await _collection.GetAsync(Mem1);

        Assert.NotNull(record);
        Assert.Equal(Mem1, record!.Id);
        Assert.Equal("batch content", record.Content);
        Assert.Equal("geo", record.Topic);
    }

    [Fact]
    public async Task GetAsync_UnknownKey_ReturnsNull()
    {
        _handler.EnqueueOk(EmptyBatchGetJson());

        var record = await _collection.GetAsync(NoSuchId);

        Assert.Null(record);
    }

    [Fact]
    public async Task GetAsync_MultipleKeys_YieldsEachRecord()
    {
        _handler.EnqueueOk($$$$"""
            {"results":[
              {"success":true,"memory":{"memoryId":"{{{{Mem1}}}}","originalContent":"one","metadata":{}}},
              {"success":true,"memory":{"memoryId":"{{{{Mem2}}}}","originalContent":"two","metadata":{}}}
            ]}
            """);

        var results = new List<Memory>();
        await foreach (var r in _collection.GetAsync(new[] { Mem1, Mem2 }))
            results.Add(r);

        Assert.Equal(2, results.Count);
        Assert.Equal("one", results[0].Content);
        Assert.Equal("two", results[1].Content);
    }

    // ── DeleteAsync ──────────────────────────────────────────────────────────

    [Fact]
    public async Task DeleteAsync_SingleKey_CallsDeleteMemory()
    {
        _handler.EnqueueNoContent();

        await _collection.DeleteAsync(Mem1);

        Assert.Single(_handler.SentRequests);
        Assert.Equal(HttpMethod.Delete, _handler.SentRequests[0].Method);
        Assert.Equal($"/v1/memories/{Mem1}", _handler.SentRequests[0].RequestUri!.AbsolutePath);
    }

    [Fact]
    public async Task DeleteAsync_MultipleKeys_CallsDeleteForEach()
    {
        _handler.EnqueueNoContent();
        _handler.EnqueueNoContent();

        await _collection.DeleteAsync(new[] { Mem1, Mem2 });

        Assert.Equal(2, _handler.SentRequests.Count);
    }

    // ── SearchAsync ──────────────────────────────────────────────────────────

    [Fact]
    public async Task SearchAsync_NonStringInput_ThrowsNotSupported()
    {
        await Assert.ThrowsAsync<NotSupportedException>(async () =>
        {
            await foreach (var _ in _collection.SearchAsync(42, top: 3)) { }
        });
    }

    [Fact]
    public async Task SearchAsync_WithFilter_ThrowsNotSupported()
    {
        _handler.EnqueueOk(SpaceFoundJson()); // ResolveSpaceId might be called first

        var options = new VectorSearchOptions<Memory> { Filter = r => r.Topic == "geo" };
        await Assert.ThrowsAsync<NotSupportedException>(async () =>
        {
            await foreach (var _ in _collection.SearchAsync("query", top: 3, options)) { }
        });
    }

    [Fact]
    public async Task SearchAsync_Text_ParsesNdjsonAndReturnsResults()
    {
        _handler.EnqueueOk(SpaceFoundJson()); // ResolveSpaceId
        _handler.Enqueue(new HttpResponseMessage(HttpStatusCode.OK)
        {
            Content = new StringContent(
                NdjsonRetrieveBody("Pacific Ocean is large", "mem-1", "geography", rawScore: -0.8),
                Encoding.UTF8,
                "application/x-ndjson"),
        });

        var results = new List<VectorSearchResult<Memory>>();
        await foreach (var r in _collection.SearchAsync("ocean", top: 3))
            results.Add(r);

        Assert.Single(results);
        Assert.Equal("Pacific Ocean is large", results[0].Record.Content);
        Assert.Equal("geography", results[0].Record.Topic);
        Assert.Equal(0.8, results[0].Score!.Value, precision: 3); // negated: -(-0.8)
    }

    [Fact]
    public async Task SearchAsync_EmptyNdjson_ReturnsNoResults()
    {
        _handler.EnqueueOk(SpaceFoundJson());
        _handler.Enqueue(new HttpResponseMessage(HttpStatusCode.OK)
        {
            Content = new StringContent("", Encoding.UTF8, "application/x-ndjson"),
        });

        var results = new List<VectorSearchResult<Memory>>();
        await foreach (var r in _collection.SearchAsync("nothing", top: 3))
            results.Add(r);

        Assert.Empty(results);
    }

    // ── GetAsync(filter) — not supported ─────────────────────────────────────

    [Fact]
    public void GetAsync_FilterBased_ThrowsNotSupported()
    {
        Assert.Throws<NotSupportedException>(
            () => _collection.GetAsync(r => r.Topic == "geo", top: 5));
    }
}
