using System.Net.Http.Json;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Web;

namespace GoodMem.SemanticKernel;

/// <summary>
/// Async HTTP client wrapping the GoodMem REST API.
/// This is an internal class; consumers should use <see cref="GoodMemCollection{TRecord}"/>
/// or <see cref="GoodMemVectorStore"/> instead.
/// </summary>
internal sealed class GoodMemClient : IDisposable
{
    private static readonly JsonSerializerOptions s_jsonOptions = new(JsonSerializerDefaults.Web);

    private readonly HttpClient _http;
    private readonly bool _ownsClient;

    internal GoodMemClient(string baseUrl, string apiKey, bool verifySsl = true)
        : this(CreateHttpClient(baseUrl, apiKey, verifySsl), ownsClient: true)
    {
    }

    internal GoodMemClient(HttpClient httpClient, bool ownsClient = false)
    {
        _http = httpClient;
        _ownsClient = ownsClient;
    }

    private static HttpClient CreateHttpClient(string baseUrl, string apiKey, bool verifySsl)
    {
        HttpMessageHandler handler = verifySsl
            ? new HttpClientHandler()
            : new HttpClientHandler { ServerCertificateCustomValidationCallback = (_, _, _, _) => true };

        var client = new HttpClient(handler, disposeHandler: true)
        {
            BaseAddress = new Uri(baseUrl.TrimEnd('/') + "/"),
            Timeout = TimeSpan.FromSeconds(30),
        };
        client.DefaultRequestHeaders.Add("x-api-key", apiKey.Trim());
        return client;
    }

    public void Dispose()
    {
        if (_ownsClient)
            _http.Dispose();
    }

    // ------------------------------------------------------------------
    // Spaces
    // ------------------------------------------------------------------

    internal async Task<List<JsonObject>> ListSpacesAsync(
        string? nameFilter = null,
        CancellationToken ct = default)
    {
        var all = new List<JsonObject>();
        string? nextToken = null;

        do
        {
            var query = HttpUtility.ParseQueryString(string.Empty);
            query["maxResults"] = "1000";
            if (nextToken is not null) query["nextToken"] = nextToken;
            if (nameFilter is not null) query["nameFilter"] = nameFilter;

            var response = await _http.GetAsync($"v1/spaces?{query}", ct).ConfigureAwait(false);
            response.EnsureSuccessStatusCode();

            var data = await response.Content.ReadFromJsonAsync<JsonObject>(s_jsonOptions, ct).ConfigureAwait(false)
                       ?? new JsonObject();

            foreach (var item in data["spaces"]?.AsArray() ?? [])
                if (item is JsonObject obj) all.Add(obj);

            nextToken = data["nextToken"]?.GetValue<string>();
        }
        while (nextToken is not null);

        return all;
    }

    internal async Task<JsonObject> CreateSpaceAsync(
        string name,
        string embedderId,
        string? spaceId = null,
        CancellationToken ct = default)
    {
        var payload = new JsonObject
        {
            ["name"] = name,
            ["spaceEmbedders"] = new JsonArray(new JsonObject
            {
                ["embedderId"] = embedderId,
                ["defaultRetrievalWeight"] = 1.0,
            }),
            ["defaultChunkingConfig"] = new JsonObject
            {
                ["recursive"] = new JsonObject
                {
                    ["chunkSize"] = 512,
                    ["chunkOverlap"] = 64,
                    ["keepStrategy"] = "KEEP_END",
                    ["lengthMeasurement"] = "CHARACTER_COUNT",
                },
            },
        };
        if (spaceId is not null)
            payload["spaceId"] = spaceId;

        var response = await _http.PostAsJsonAsync("v1/spaces", payload, s_jsonOptions, ct).ConfigureAwait(false);
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadFromJsonAsync<JsonObject>(s_jsonOptions, ct).ConfigureAwait(false)
               ?? new JsonObject();
    }

    internal async Task DeleteSpaceAsync(string spaceId, CancellationToken ct = default)
    {
        var response = await _http.DeleteAsync($"v1/spaces/{Uri.EscapeDataString(spaceId)}", ct).ConfigureAwait(false);
        response.EnsureSuccessStatusCode();
    }

    // ------------------------------------------------------------------
    // Embedders
    // ------------------------------------------------------------------

    internal async Task<List<JsonObject>> ListEmbeddersAsync(CancellationToken ct = default)
    {
        var response = await _http.GetAsync("v1/embedders", ct).ConfigureAwait(false);
        response.EnsureSuccessStatusCode();

        var data = await response.Content.ReadFromJsonAsync<JsonObject>(s_jsonOptions, ct).ConfigureAwait(false)
                   ?? new JsonObject();

        var result = new List<JsonObject>();
        foreach (var item in data["embedders"]?.AsArray() ?? [])
            if (item is JsonObject obj) result.Add(obj);
        return result;
    }

    // ------------------------------------------------------------------
    // Memories
    // ------------------------------------------------------------------

    internal async Task<JsonObject> CreateMemoryAsync(
        string spaceId,
        string content,
        string contentType = "text/plain",
        Dictionary<string, object?>? metadata = null,
        string? memoryId = null,
        CancellationToken ct = default)
    {
        var payload = new JsonObject
        {
            ["spaceId"] = spaceId,
            ["originalContent"] = content,
            ["contentType"] = contentType,
        };
        if (metadata is { Count: > 0 })
            payload["metadata"] = JsonSerializer.SerializeToNode(metadata, s_jsonOptions);
        if (memoryId is not null)
            payload["memoryId"] = memoryId;

        var response = await _http.PostAsJsonAsync("v1/memories", payload, s_jsonOptions, ct).ConfigureAwait(false);
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadFromJsonAsync<JsonObject>(s_jsonOptions, ct).ConfigureAwait(false)
               ?? new JsonObject();
    }

    internal async Task<List<JsonObject>> BatchGetMemoriesAsync(
        IEnumerable<string> memoryIds,
        CancellationToken ct = default)
    {
        var ids = memoryIds.ToList();
        if (ids.Count == 0) return [];

        var payload = new JsonObject
        {
            ["memoryIds"] = new JsonArray(ids.Select(id => JsonValue.Create(id)).Cast<JsonNode?>().ToArray()),
            ["includeContent"] = true,
        };

        var response = await _http.PostAsJsonAsync("v1/memories:batchGet", payload, s_jsonOptions, ct).ConfigureAwait(false);
        response.EnsureSuccessStatusCode();

        var data = await response.Content.ReadFromJsonAsync<JsonObject>(s_jsonOptions, ct).ConfigureAwait(false)
                   ?? new JsonObject();

        // Response shape: { "results": [ { "success": true, "memory": {...} }, ... ] }
        var result = new List<JsonObject>();
        foreach (var item in data["results"]?.AsArray() ?? [])
        {
            if (item is JsonObject resultObj &&
                resultObj["success"]?.GetValue<bool>() == true &&
                resultObj["memory"] is JsonObject mem)
            {
                result.Add(mem);
            }
        }
        return result;
    }

    internal async Task DeleteMemoryAsync(string memoryId, CancellationToken ct = default)
    {
        var response = await _http.DeleteAsync(
            $"v1/memories/{Uri.EscapeDataString(memoryId)}", ct).ConfigureAwait(false);

        if (response.StatusCode == System.Net.HttpStatusCode.NotFound)
            return;

        response.EnsureSuccessStatusCode();
    }

    /// <summary>
    /// Semantic search over one or more spaces. Parses the NDJSON response and
    /// returns a unified list of results with chunk, memory, and score.
    /// </summary>
    internal async Task<List<RetrieveResult>> RetrieveMemoriesAsync(
        string query,
        IEnumerable<string> spaceIds,
        int top = 5,
        CancellationToken ct = default)
    {
        var payload = new JsonObject
        {
            ["message"] = query,
            ["spaceKeys"] = new JsonArray(
                spaceIds.Select(sid => (JsonNode?)new JsonObject { ["spaceId"] = sid }).ToArray()),
            ["requestedSize"] = top,
        };

        using var request = new HttpRequestMessage(HttpMethod.Post, "v1/memories:retrieve")
        {
            Content = JsonContent.Create(payload, options: s_jsonOptions),
        };
        request.Headers.Accept.ParseAdd("application/x-ndjson");

        using var response = await _http.SendAsync(request, ct).ConfigureAwait(false);
        response.EnsureSuccessStatusCode();

        var body = await response.Content.ReadAsStringAsync(ct).ConfigureAwait(false);

        // Parse NDJSON events.
        // memoryDefinition events arrive indexed by position (0, 1, 2, …).
        // retrievedItem events reference a memoryIndex into that list.
        var memoryList = new List<JsonObject>();
        var chunkRefs = new List<JsonObject>();

        foreach (var line in body.Split('\n', StringSplitOptions.RemoveEmptyEntries))
        {
            var trimmed = line.Trim();
            if (string.IsNullOrEmpty(trimmed)) continue;

            JsonObject? evt;
            try { evt = JsonSerializer.Deserialize<JsonObject>(trimmed, s_jsonOptions); }
            catch { continue; }
            if (evt is null) continue;

            if (evt["memoryDefinition"] is JsonObject memDef)
            {
                memoryList.Add(memDef);
                continue;
            }

            if (evt["retrievedItem"] is JsonObject item && item["chunk"] is JsonObject chunkRef)
                chunkRefs.Add(chunkRef);
        }

        var results = new List<RetrieveResult>(chunkRefs.Count);
        foreach (var chunkRef in chunkRefs)
        {
            var chunk = chunkRef["chunk"] as JsonObject ?? new JsonObject();
            var rawScore = chunkRef["relevanceScore"]?.GetValue<double>() ?? 0.0;
            var memoryIndex = chunkRef["memoryIndex"]?.GetValue<int>();

            var mem = memoryIndex is >= 0 && memoryIndex < memoryList.Count
                ? memoryList[memoryIndex.Value]
                : new JsonObject();

            results.Add(new RetrieveResult(chunk, mem, -rawScore)); // negate: lower-is-better → higher-is-better
        }

        return results;
    }

    /// <summary>A correlated retrieve result: chunk data, memory metadata, and similarity score.</summary>
    internal sealed record RetrieveResult(JsonObject Chunk, JsonObject Memory, double Score);
}
