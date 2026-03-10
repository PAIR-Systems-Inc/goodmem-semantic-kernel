using System.Net;
using System.Text;

namespace GoodMem.SemanticKernel.Tests;

/// <summary>
/// Test double for HttpMessageHandler. Pre-queue responses; captures sent requests.
/// </summary>
internal sealed class MockHttpMessageHandler : HttpMessageHandler
{
    private readonly Queue<HttpResponseMessage> _responses = new();

    /// <summary>All requests that were sent through this handler, in order.</summary>
    public List<HttpRequestMessage> SentRequests { get; } = [];

    /// <summary>Enqueue a response to return on the next request.</summary>
    public void Enqueue(HttpResponseMessage response) => _responses.Enqueue(response);

    /// <summary>Shorthand: enqueue a 200 OK with a JSON body.</summary>
    public void EnqueueOk(string json) => Enqueue(OkJson(json));

    /// <summary>Shorthand: enqueue a 204 No Content response.</summary>
    public void EnqueueNoContent() => Enqueue(new HttpResponseMessage(HttpStatusCode.NoContent));

    /// <summary>Shorthand: enqueue a 404 Not Found response.</summary>
    public void EnqueueNotFound() => Enqueue(new HttpResponseMessage(HttpStatusCode.NotFound));

    protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken ct)
    {
        SentRequests.Add(request);
        var response = _responses.TryDequeue(out var r) ? r : OkJson("{}");
        return Task.FromResult(response);
    }

    internal static HttpResponseMessage OkJson(string json) =>
        new(HttpStatusCode.OK)
        {
            Content = new StringContent(json, Encoding.UTF8, "application/json"),
        };
}
