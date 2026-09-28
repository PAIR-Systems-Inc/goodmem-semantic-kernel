using System.Net;
using System.Text;

namespace GoodMem.SemanticKernel.Tests;

/// <summary>
/// Retrieval streams captured from a live GoodMem server (v1.0.320), and variations of them.
/// </summary>
/// <remarks>
/// <c>retrieve_ok</c> is a healthy search with one hit. <c>retrieve_degraded_hits</c> is the same
/// search with a reranker id that does not exist: the server sends <c>NOT_FOUND</c>,
/// <c>FEATURE_DISABLED</c> and <c>RERANKING_FAILED</c>, then the vector hit.
/// <c>retrieve_degraded_empty</c> is that search on an empty space.
/// </remarks>
internal static class RetrievalFixtures
{
    /// <summary>The text of the one chunk in the captured streams.</summary>
    internal const string CanaryText = "The fixture canary is ORYX-2290. CAMEL toolkit audit.\n";

    /// <summary>Its raw vector score in the degraded stream (negative; lower is better).</summary>
    internal const double DegradedRawScore = -0.5845972299575806;

    internal static string Load(string name) =>
        File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Fixtures", name + ".ndjson"));

    /// <summary>A status event line, as the server writes one.</summary>
    internal static string StatusLine(string? code, string message, string? detailsJson = null)
    {
        var codePart = code is null ? "" : $"\"code\":\"{code}\",";
        var detailsPart = detailsJson is null ? "" : $",\"details\":{detailsJson}";
        return $"{{\"status\":{{{codePart}\"message\":\"{message}\"{detailsPart}}}}}";
    }

    /// <summary>The healthy captured stream with extra lines before it.</summary>
    internal static string OkWith(params string[] leadingLines) =>
        string.Join("\n", leadingLines) + "\n" + Load("retrieve_ok");

    internal static HttpResponseMessage Ndjson(string body) =>
        new(HttpStatusCode.OK)
        {
            Content = new StringContent(body, Encoding.UTF8, "application/x-ndjson"),
        };
}
