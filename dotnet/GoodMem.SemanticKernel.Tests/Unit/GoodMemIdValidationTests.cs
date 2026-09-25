using System.Text.Json.Nodes;
using Microsoft.Extensions.VectorData;
using Xunit;

namespace GoodMem.SemanticKernel.Tests.Unit;

internal class IdRecord
{
    [VectorStoreKey] public string? Id { get; set; }
    [VectorStoreData] public string Content { get; set; } = "";
}

/// <summary>
/// Ids that can reach a URL path must be UUIDs, checked before any request.
/// </summary>
/// <remarks>
/// Nothing here is mocked: each test drives the public connector API through
/// its own <see cref="HttpClient"/> over TCP to <see cref="RecordingHttpServer"/>,
/// which records every request line it receives.
/// </remarks>
public sealed class GoodMemIdValidationTests : IDisposable
{
    private const string U = "0198c3a2-7f4e-7c1a-9b2d-5e6f7a8b9c0d";
    private const string EmbedderId = "019cfd1c-c033-7517-b7de-f73941a0464b";
    private const string Refusal = "must be a GoodMem UUID";

    private static readonly string[] s_payloads =
    [
        $"../spaces/{U}",
        $"a/../../spaces/{U}",
        $"%2e%2e/spaces/{U}",
        $"..%2Fspaces%2F{U}",
        $"{U}/../../spaces/{U}",
        "",
        $" {U}",
        $"{U}?x=1",
        $"{U}#frag",
        $"{U}\n",
        "..",
    ];

    public static TheoryData<string> Payloads => new(s_payloads);

    /// <summary>
    /// An empty setting means "not configured", as it does for any GOODMEM_* env
    /// var; it is never sent anywhere. Every other payload is refused.
    /// </summary>
    public static TheoryData<string> SettingPayloads => new(s_payloads.Where(p => p.Length > 0));

    private readonly RecordingHttpServer _server = new();
    private readonly List<IDisposable> _made = [];

    public void Dispose()
    {
        foreach (var made in _made) made.Dispose();
        _server.Dispose();
    }

    private GoodMemOptions Options(string? embedderId = EmbedderId) => new()
    {
        BaseUrl = _server.Url,
        ApiKey = "test-key",
        EmbedderId = embedderId,
    };

    private GoodMemCollection<IdRecord> Collection(string? embedderId = EmbedderId)
    {
        var collection = new GoodMemCollection<IdRecord>("notes", Options(embedderId));
        _made.Add(collection);
        return collection;
    }

    private static async Task<Exception?> Outcome(Func<Task> call)
    {
        try { await call(); return null; }
        catch (Exception error) { return error; }
    }

    private void AssertRefused(Exception? error)
    {
        // Requests first: against the unfixed code this is the assertion that
        // fails, and its message shows the path the server actually received.
        Assert.True(_server.Requests.Count == 0, $"the server received {_server.Describe()}");
        var argument = Assert.IsAssignableFrom<ArgumentException>(error);
        Assert.Contains(Refusal, argument.Message);
    }

    // ── DeleteAsync: the key is the path segment of DELETE /v1/memories/{key} ──

    [Theory]
    [MemberData(nameof(Payloads))]
    public async Task DeleteAsync_RefusesANonUuidKey(string payload)
    {
        var error = await Outcome(() => Collection().DeleteAsync(payload));
        AssertRefused(error);
    }

    [Theory]
    [MemberData(nameof(Payloads))]
    public async Task DeleteAsync_RefusesTheWholeBatchBeforeDeletingAny(string payload)
    {
        var error = await Outcome(() => Collection().DeleteAsync(new[] { U, payload }));
        AssertRefused(error);
    }

    [Fact]
    public async Task DeleteAsync_WithAUuid_ReachesExactlyThatMemory()
    {
        await Collection().DeleteAsync(U);

        Assert.Equal($"[DELETE /v1/memories/{U}]", _server.Describe());
    }

    [Fact]
    public async Task DeleteAsync_NormalisesAnUppercaseUuid()
    {
        await Collection().DeleteAsync(U.ToUpperInvariant());

        Assert.Equal($"[DELETE /v1/memories/{U}]", _server.Describe());
    }

    // ── UpsertAsync: the key reaches DELETE /v1/memories/{key} on replace ──

    [Theory]
    [MemberData(nameof(Payloads))]
    public async Task UpsertAsync_RefusesANonUuidKey(string payload)
    {
        var error = await Outcome(() => Collection().UpsertAsync(new IdRecord { Id = payload, Content = "x" }));
        AssertRefused(error);
    }

    [Theory]
    [MemberData(nameof(Payloads))]
    public async Task UpsertAsync_RefusesTheWholeBatchBeforeWritingAny(string payload)
    {
        var batch = new[]
        {
            new IdRecord { Id = U, Content = "first" },
            new IdRecord { Id = payload, Content = "second" },
        };
        var error = await Outcome(() => Collection().UpsertAsync(batch));
        AssertRefused(error);
    }

    [Fact]
    public async Task UpsertAsync_WithAUuidKey_SendsExactlyThatId()
    {
        await Collection().UpsertAsync(new IdRecord { Id = U, Content = "x" });

        var requests = _server.Requests.Where(r => !r.Target.StartsWith("/v1/spaces")).ToList();
        Assert.Equal("POST /v1/memories:batchGet", requests[0].ToString());
        Assert.Equal(U, JsonNode.Parse(requests[0].Body)!["memoryIds"]![0]!.GetValue<string>());
        Assert.Equal("POST /v1/memories", requests[1].ToString());
        Assert.Equal(U, JsonNode.Parse(requests[1].Body)!["memoryId"]!.GetValue<string>());
    }

    [Fact]
    public async Task UpsertAsync_WithoutAKey_LetsTheServerAssignOne()
    {
        var record = new IdRecord { Content = "x" };

        await Collection().UpsertAsync(record);

        Assert.Equal(RecordingHttpServer.AssignedMemoryId, record.Id);
        Assert.Null(JsonNode.Parse(_server.Requests[^1].Body)!["memoryId"]);
    }

    // ── GetAsync: ids travel in the batchGet body, and are checked the same way ──

    [Theory]
    [MemberData(nameof(Payloads))]
    public async Task GetAsync_RefusesANonUuidKey(string payload)
    {
        var error = await Outcome(() => Collection().GetAsync(payload));
        AssertRefused(error);
    }

    [Theory]
    [MemberData(nameof(Payloads))]
    public async Task GetAsync_Batch_RefusesANonUuidKey(string payload)
    {
        var error = await Outcome(async () =>
        {
            await foreach (var _ in Collection().GetAsync(new[] { U, payload })) { }
        });
        AssertRefused(error);
    }

    [Fact]
    public async Task GetAsync_WithAUuid_AsksForExactlyThatMemory()
    {
        await Collection().GetAsync(U.ToUpperInvariant());

        var request = Assert.Single(_server.Requests);
        Assert.Equal("POST /v1/memories:batchGet", request.ToString());
        Assert.Equal(U, JsonNode.Parse(request.Body)!["memoryIds"]![0]!.GetValue<string>());
    }

    // ── Ids from configuration ──

    [Theory]
    [MemberData(nameof(SettingPayloads))]
    public async Task ANonUuidEmbedderSetting_IsRefused(string payload)
    {
        var error = await Outcome(() => Collection(embedderId: payload).EnsureCollectionExistsAsync());
        AssertRefused(error);
        Assert.Contains("EmbedderId", error!.Message);
    }

    [Fact]
    public async Task AUuidEmbedderSetting_IsUsed()
    {
        var collection = new GoodMemCollection<IdRecord>("other", Options());
        _made.Add(collection);

        await collection.EnsureCollectionExistsAsync();

        var create = _server.Requests.Single(r => r.ToString() == "POST /v1/spaces");
        Assert.Equal(EmbedderId, JsonNode.Parse(create.Body)!["spaceEmbedders"]![0]!["embedderId"]!.GetValue<string>());
    }

    // ── Ids from the server: the space id in DELETE /v1/spaces/{id} ──

    [Theory]
    [MemberData(nameof(SettingPayloads))]
    public async Task ANonUuidSpaceIdFromTheServer_IsNeverDeleted(string payload)
    {
        _server.ListedSpaceId = payload;

        var error = await Outcome(() => Collection().EnsureCollectionDeletedAsync());

        Assert.DoesNotContain(_server.Requests, r => r.Method == "DELETE");
        Assert.Contains(Refusal, Assert.IsAssignableFrom<ArgumentException>(error).Message);
    }

    [Theory]
    [MemberData(nameof(SettingPayloads))]
    public async Task ANonUuidSpaceIdFromTheServer_IsNeverDeletedByTheStore(string payload)
    {
        _server.ListedSpaceId = payload;
        using var store = new GoodMemVectorStore(Options());

        var error = await Outcome(() => store.EnsureCollectionDeletedAsync("notes"));

        Assert.DoesNotContain(_server.Requests, r => r.Method == "DELETE");
        Assert.Contains(Refusal, Assert.IsAssignableFrom<ArgumentException>(error).Message);
    }

    [Fact]
    public async Task EnsureCollectionDeletedAsync_DeletesExactlyTheListedSpace()
    {
        await Collection().EnsureCollectionDeletedAsync();

        Assert.Equal($"DELETE /v1/spaces/{RecordingHttpServer.DefaultSpaceId}", _server.Requests[^1].ToString());
    }
}
