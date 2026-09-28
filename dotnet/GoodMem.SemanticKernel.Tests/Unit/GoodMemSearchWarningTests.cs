using Microsoft.Extensions.VectorData;
using Xunit;

namespace GoodMem.SemanticKernel.Tests.Unit;

/// <summary>These tests swap <see cref="Console.Error"/>, so they never run in parallel.</summary>
[CollectionDefinition(nameof(StandardErrorCollection), DisableParallelization = true)]
public sealed class StandardErrorCollection
{
}

/// <summary>
/// <c>SearchAsync</c> is Semantic Kernel's interface and has no slot for a flag, so a problem
/// the server reported has to show in the log. With no <c>LoggerFactory</c> set, the log is
/// standard error. Before the fix every one of these searches looked healthy: the status
/// events were dropped and nothing was written anywhere.
/// </summary>
/// <remarks>
/// These drive the connector's real HTTP client over a mock transport, with retrieval streams
/// captured from a live server (see <see cref="RetrievalFixtures"/>).
/// </remarks>
[Collection(nameof(StandardErrorCollection))]
public sealed class GoodMemSearchWarningTests : IDisposable
{
    private const string EmbedderId = "019cfd1c-c033-7517-b7de-f73941a0464b";
    private const string SpaceId = "01a0d44b-746f-775b-b91e-bc73d4058e27";

    private readonly MockHttpMessageHandler _handler = new();
    private readonly GoodMemClient _client;
    private readonly GoodMemCollection<Memory> _collection;

    public GoodMemSearchWarningTests()
    {
        var httpClient = new HttpClient(_handler) { BaseAddress = new Uri("http://localhost:8080/") };
        _client = new GoodMemClient(httpClient, ownsClient: false);
        _collection = new GoodMemCollection<Memory>(
            "test-collection", _client, new GoodMemOptions { EmbedderId = EmbedderId });
    }

    public void Dispose()
    {
        _collection.Dispose();
        _client.Dispose();
    }

    private async Task<(List<VectorSearchResult<Memory>> Hits, string[] Lines)> SearchAsync(string stream)
    {
        _handler.EnqueueOk($$"""{"spaces":[{"name":"test-collection","spaceId":"{{SpaceId}}"}]}""");
        _handler.Enqueue(RetrievalFixtures.Ndjson(stream));

        var hits = new List<VectorSearchResult<Memory>>();
        var original = Console.Error;
        var captured = new StringWriter();
        Console.SetError(captured);
        try
        {
            await foreach (var hit in _collection.SearchAsync("fixture canary", top: 5))
                hits.Add(hit);
        }
        finally
        {
            Console.SetError(original);
        }
        return (hits, captured.ToString().Split('\n', StringSplitOptions.RemoveEmptyEntries));
    }

    [Fact]
    public async Task ProblemWithHits_YieldsTheHits_AndWarnsOnStandardError()
    {
        // Contract Q4a, on the captured stream of a search whose reranker does not exist.
        var (hits, lines) = await SearchAsync(RetrievalFixtures.Load("retrieve_degraded_hits"));

        var hit = Assert.Single(hits);
        Assert.Equal(RetrievalFixtures.CanaryText, hit.Record.Content);
        Assert.Equal(-RetrievalFixtures.DegradedRawScore, hit.Score!.Value, precision: 6);
        var line = Assert.Single(lines);
        Assert.StartsWith("warn: GoodMem.SemanticKernel.GoodMemCollection: ", line);
        Assert.Contains("may be incomplete", line);
        Assert.Contains("NOT_FOUND", line);
        Assert.Contains("RERANKING_FAILED", line);
        Assert.DoesNotContain("FEATURE_DISABLED", line);
    }

    [Fact]
    public async Task ProblemWithNoHits_ReturnsEmpty_WarnsAndDoesNotThrow()
    {
        // Contract Q4b: an empty result must not look like a search that found nothing.
        var (hits, lines) = await SearchAsync(RetrievalFixtures.Load("retrieve_degraded_empty"));

        Assert.Empty(hits);
        var line = Assert.Single(lines);
        Assert.StartsWith("warn: ", line);
        Assert.Contains("returned no results", line);
        Assert.Contains("RERANKING_FAILED", line);
    }

    [Fact]
    public async Task UnknownCode_KeepsTheHits_AndWarnsWithTheServersCode()
    {
        // Contract Q3: never crash, never drop.
        var (hits, lines) = await SearchAsync(RetrievalFixtures.OkWith(
            RetrievalFixtures.StatusLine("SOMETHING_NEW_IN_A_LATER_SERVER", "unrecognised")));

        Assert.Single(hits);
        Assert.Contains("UNKNOWN (server code SOMETHING_NEW_IN_A_LATER_SERVER): unrecognised", Assert.Single(lines));
    }

    [Fact]
    public async Task TruncatedLastLine_KeepsTheHits_AndWarns()
    {
        var (hits, lines) = await SearchAsync(
            RetrievalFixtures.Load("retrieve_ok") + "\n{\"retrievedItem\":{\"chunk\":{\"chu");

        Assert.Equal(RetrievalFixtures.CanaryText, Assert.Single(hits).Record.Content);
        Assert.Contains("MALFORMED_STREAM", Assert.Single(lines));
    }

    [Fact]
    public async Task InformationalNotices_WriteNothing()
    {
        // Contract Q1: by code alone, whatever the details say.
        var (hits, lines) = await SearchAsync(RetrievalFixtures.OkWith(
            RetrievalFixtures.StatusLine("FEATURE_DISABLED", "no LLM configured", """{"feature":"summarization"}"""),
            RetrievalFixtures.StatusLine("LLM_CAPABILITY_INFERRED", "capabilities inferred", """{"reranker_id":"x"}""")));

        Assert.Single(hits);
        Assert.Empty(lines);
    }

    [Fact]
    public async Task HealthySearch_WritesNothing()
    {
        var (hits, lines) = await SearchAsync(RetrievalFixtures.Load("retrieve_ok"));

        Assert.Single(hits);
        Assert.Empty(lines);
    }
}
