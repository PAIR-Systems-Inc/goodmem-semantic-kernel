using Microsoft.Extensions.Logging;
using Microsoft.Extensions.VectorData;
using Xunit;

namespace GoodMem.SemanticKernel.Tests.Unit;

/// <summary>
/// The retrieval status contract, through <see cref="GoodMemCollection{TRecord}.SearchWithStatusAsync"/>.
/// </summary>
/// <remarks>
/// Q1: <c>FEATURE_DISABLED</c> and <c>LLM_CAPABILITY_INFERRED</c> are noise, by code alone.
/// Q3: an unrecognised code is surfaced as <c>UNKNOWN</c> and marks the result partial.
/// Q4a: a problem with hits returns the hits, partial, with the statuses.
/// Q4b: a problem with no hits returns empty and partial, logs a warning, and does not throw.
/// These drive the connector's real HTTP client over a mock transport, with retrieval streams
/// captured from a live server (see <see cref="RetrievalFixtures"/>).
/// </remarks>
public sealed class GoodMemSearchStatusTests : IDisposable
{
    private const string EmbedderId = "019cfd1c-c033-7517-b7de-f73941a0464b";
    private const string SpaceId = "01a0d44b-746f-775b-b91e-bc73d4058e27";
    private const string MissingReranker = "00000000-0000-7000-8000-000000000000";
    private const string Category = "GoodMem.SemanticKernel.GoodMemCollection";

    private readonly MockHttpMessageHandler _handler = new();
    private readonly CapturingLoggerFactory _logs = new();
    private readonly GoodMemClient _client;
    private readonly GoodMemCollection<Memory> _collection;

    public GoodMemSearchStatusTests()
    {
        var httpClient = new HttpClient(_handler) { BaseAddress = new Uri("http://localhost:8080/") };
        _client = new GoodMemClient(httpClient, ownsClient: false);
        _collection = new GoodMemCollection<Memory>(
            "test-collection", _client, new GoodMemOptions { EmbedderId = EmbedderId, LoggerFactory = _logs });
    }

    public void Dispose()
    {
        _collection.Dispose();
        _client.Dispose();
    }

    private void Serve(string stream)
    {
        _handler.EnqueueOk($$"""{"spaces":[{"name":"test-collection","spaceId":"{{SpaceId}}"}]}""");
        _handler.Enqueue(RetrievalFixtures.Ndjson(stream));
    }

    private Task<GoodMemSearchResults<Memory>> SearchAsync(string stream)
    {
        Serve(stream);
        return _collection.SearchWithStatusAsync("fixture canary", top: 5);
    }

    [Fact]
    public async Task HealthySearch_IsNotPartial()
    {
        var search = await SearchAsync(RetrievalFixtures.Load("retrieve_ok"));

        Assert.Equal(RetrievalFixtures.CanaryText, Assert.Single(search.Results).Record.Content);
        Assert.False(search.Partial);
        Assert.Empty(search.Statuses);
        Assert.Empty(_logs.Entries);
    }

    [Fact]
    public async Task Q1_InformationalCodes_AreNoise_ByCodeAlone()
    {
        // Details that name a reranker must not turn a notice into a problem.
        var search = await SearchAsync(RetrievalFixtures.OkWith(
            RetrievalFixtures.StatusLine("FEATURE_DISABLED", "no LLM configured", $$"""{"reranker_id":"{{MissingReranker}}"}"""),
            RetrievalFixtures.StatusLine("LLM_CAPABILITY_INFERRED", "capabilities inferred")));

        Assert.Single(search.Results);
        Assert.False(search.Partial);
        Assert.Empty(search.Statuses);
        Assert.Empty(_logs.Entries);
    }

    [Fact]
    public async Task Q3_UnknownCode_IsSurfacedAsUnknown_WithTheServersCode_AndKeepsTheHits()
    {
        var search = await SearchAsync(RetrievalFixtures.OkWith(
            RetrievalFixtures.StatusLine("SOMETHING_NEW_IN_A_LATER_SERVER", "unrecognised", """{"stage":"retrieve"}""")));

        Assert.Single(search.Results);
        Assert.True(search.Partial);
        var status = Assert.Single(search.Statuses);
        Assert.Equal("UNKNOWN", status.Code);
        Assert.Equal("SOMETHING_NEW_IN_A_LATER_SERVER", status.OriginalCode);
        Assert.True(status.Unrecognized);
        Assert.Equal("unrecognised", status.Message);
        Assert.Equal("retrieve", status.Details["stage"]);
    }

    [Fact]
    public async Task Q3_StatusWithNoCode_IsUnknown()
    {
        var search = await SearchAsync(RetrievalFixtures.OkWith(
            RetrievalFixtures.StatusLine(code: null, "a status with no code")));

        Assert.Single(search.Results);
        Assert.True(search.Partial);
        var status = Assert.Single(search.Statuses);
        Assert.Equal("UNKNOWN", status.Code);
        Assert.Null(status.OriginalCode);
        Assert.True(status.Unrecognized);
    }

    [Fact]
    public async Task Q4a_CapturedDegradedStream_ReturnsTheHit_Partial_WithTheStatuses()
    {
        var search = await SearchAsync(RetrievalFixtures.Load("retrieve_degraded_hits"));

        var hit = Assert.Single(search.Results);
        Assert.Equal(RetrievalFixtures.CanaryText, hit.Record.Content);
        // The reranker failed, so this is the vector score, negated into higher-is-better.
        Assert.Equal(-RetrievalFixtures.DegradedRawScore, hit.Score!.Value, precision: 6);
        Assert.True(search.Partial);
        Assert.Equal(new[] { "NOT_FOUND", "RERANKING_FAILED" }, search.Statuses.Select(s => s.Code));
        Assert.All(search.Statuses, s => Assert.Equal(MissingReranker, s.Details["reranker_id"]));
        Assert.All(search.Statuses, s => Assert.False(s.Unrecognized));
        // The flag is in the result, so there is nothing to log.
        Assert.Empty(_logs.Entries);
    }

    [Fact]
    public async Task Q4b_CapturedDegradedEmptyStream_ReturnsEmpty_Partial_LogsAWarning_AndDoesNotThrow()
    {
        var search = await SearchAsync(RetrievalFixtures.Load("retrieve_degraded_empty"));

        Assert.Empty(search.Results);
        Assert.True(search.Partial);
        Assert.Equal(new[] { "NOT_FOUND", "RERANKING_FAILED" }, search.Statuses.Select(s => s.Code));
        var entry = Assert.Single(_logs.Entries);
        Assert.Equal(LogLevel.Warning, entry.Level);
        Assert.Equal(Category, entry.Category);
        Assert.Contains("returned no results", entry.Message);
        Assert.Contains("NOT_FOUND", entry.Message);
        Assert.Contains("RERANKING_FAILED", entry.Message);
        Assert.Contains(MissingReranker, entry.Message);
    }

    [Fact]
    public async Task HealthyEmptySearch_IsNotPartial_AndLogsNothing()
    {
        var search = await SearchAsync("");

        Assert.Empty(search.Results);
        Assert.False(search.Partial);
        Assert.Empty(_logs.Entries);
    }

    [Fact]
    public async Task TruncatedLastLine_KeepsTheHitsBeforeIt_AndIsPartial()
    {
        // Four whole lines, then a fifth cut off mid-object.
        var search = await SearchAsync(
            RetrievalFixtures.Load("retrieve_ok").TrimEnd('\n') + "\n{\"retrievedItem\":{\"chunk\":{\"chu");

        Assert.Equal(RetrievalFixtures.CanaryText, Assert.Single(search.Results).Record.Content);
        Assert.True(search.Partial);
        var status = Assert.Single(search.Statuses);
        Assert.Equal("MALFORMED_STREAM", status.Code);
        Assert.Contains("Line 5", status.Message);
        Assert.False(status.Unrecognized);
    }

    [Fact]
    public async Task LineThatIsNotAnObject_IsReported_AndTheLinesAfterItAreStillRead()
    {
        var search = await SearchAsync("[1,2]\nnull\n" + RetrievalFixtures.Load("retrieve_ok"));

        Assert.Single(search.Results);
        Assert.Equal(new[] { "MALFORMED_STREAM", "MALFORMED_STREAM" }, search.Statuses.Select(s => s.Code));
    }

    [Fact]
    public async Task SearchAsync_LogsAWarning_EvenWhenTheProblemCameWithHits()
    {
        // SearchAsync has no slot for the flag, so the log is the only place it can show.
        Serve(RetrievalFixtures.Load("retrieve_degraded_hits"));

        var hits = new List<VectorSearchResult<Memory>>();
        await foreach (var hit in _collection.SearchAsync("fixture canary", top: 5))
            hits.Add(hit);

        Assert.Single(hits);
        var entry = Assert.Single(_logs.Entries);
        Assert.Equal(LogLevel.Warning, entry.Level);
        Assert.Contains("the 1 results it returned may be incomplete", entry.Message);
        Assert.Contains("RERANKING_FAILED", entry.Message);
    }

    [Fact]
    public async Task LoggedStatuses_StayOnOneLine()
    {
        await SearchAsync(RetrievalFixtures.StatusLine("VECTOR_SEARCH_FAILED", "first\\nwarn: forged line"));

        var entry = Assert.Single(_logs.Entries);
        Assert.DoesNotContain('\n', entry.Message);
        Assert.Contains("VECTOR_SEARCH_FAILED: first warn: forged line", entry.Message);
    }

    [Fact]
    public async Task VectorStore_GivesItsCollectionsTheLoggerFactory()
    {
        using var server = new RecordingHttpServer { RetrieveBody = RetrievalFixtures.Load("retrieve_degraded_empty") };
        var logs = new CapturingLoggerFactory();
        using var store = new GoodMemVectorStore(new GoodMemOptions
        {
            BaseUrl = server.Url,
            ApiKey = "test-key",
            EmbedderId = EmbedderId,
            LoggerFactory = logs,
        });
        var collection = (GoodMemCollection<Memory>)store.GetCollection<string, Memory>("notes");

        var search = await collection.SearchWithStatusAsync("fixture canary", top: 5);

        Assert.True(search.Partial);
        Assert.Contains("returned no results", Assert.Single(logs.Entries).Message);
    }
}
