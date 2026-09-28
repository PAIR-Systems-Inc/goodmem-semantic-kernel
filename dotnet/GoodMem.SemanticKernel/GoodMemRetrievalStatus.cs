using System.Text.Json.Nodes;

namespace GoodMem.SemanticKernel;

/// <summary>
/// A problem the GoodMem server reported while it ran a search.
/// </summary>
/// <remarks>
/// Only statuses that mean something may be missing from the results are reported.
/// <c>FEATURE_DISABLED</c> and <c>LLM_CAPABILITY_INFERRED</c> are notices about optional
/// features the search did not ask for, and are left out by their code alone. A code this
/// connector does not know is reported as <c>UNKNOWN</c>, with the server's own code kept in
/// <see cref="OriginalCode"/>. A line of the response stream that cannot be read is reported as
/// <c>MALFORMED_STREAM</c>.
/// </remarks>
public sealed class GoodMemRetrievalStatus
{
    /// <summary>Creates a status.</summary>
    public GoodMemRetrievalStatus(
        string code,
        string message,
        IReadOnlyDictionary<string, string>? details = null,
        string? originalCode = null,
        bool unrecognized = false)
    {
        ArgumentNullException.ThrowIfNull(code);
        Code = code;
        Message = message ?? "";
        Details = details ?? new Dictionary<string, string>();
        OriginalCode = originalCode;
        Unrecognized = unrecognized;
    }

    /// <summary>
    /// GoodMem's status code, such as <c>RERANKING_FAILED</c>; <c>UNKNOWN</c> for a code this
    /// connector does not recognise; or <c>MALFORMED_STREAM</c> for a line it could not read.
    /// </summary>
    public string Code { get; }

    /// <summary>The server's human-readable message.</summary>
    public string Message { get; }

    /// <summary>The server's details, such as <c>reranker_id</c>. Empty when it sent none.</summary>
    public IReadOnlyDictionary<string, string> Details { get; }

    /// <summary>
    /// The code exactly as the server sent it, when <see cref="Code"/> is <c>UNKNOWN</c>.
    /// <see langword="null"/> otherwise, and when the server sent no code at all.
    /// </summary>
    public string? OriginalCode { get; }

    /// <summary>True when the server's code is not one this connector recognises.</summary>
    public bool Unrecognized { get; }

    /// <inheritdoc/>
    public override string ToString()
    {
        var code = OriginalCode is null ? Code : $"{Code} (server code {OriginalCode})";
        var details = Details.Count == 0
            ? ""
            : " {" + string.Join(", ", Details.Select(d => $"{d.Key}={d.Value}")) + "}";
        return $"{code}: {Message}{details}";
    }
}

/// <summary>
/// Sorts the <c>status</c> events of a retrieval stream: the retrieval status contract every
/// GoodMem integration follows.
/// </summary>
internal static class RetrievalStatuses
{
    internal const string Unknown = "UNKNOWN";
    internal const string MalformedStream = "MALFORMED_STREAM";

    // Notices that carry no loss of results, by their code alone (contract Q1).
    private static readonly HashSet<string> s_informational = new(StringComparer.Ordinal)
    {
        "LLM_CAPABILITY_INFERRED",
        "FEATURE_DISABLED",
    };

    // Every code GoodMem's API defines for a status (GoodMemStatus.code, server v1.0.320).
    private static readonly HashSet<string> s_known = new(StringComparer.Ordinal)
    {
        "GOODMEM_STATUS_CODE_UNSPECIFIED", "INVALID_ARGUMENT", "NOT_FOUND", "PERMISSION_DENIED",
        "FAILED_PRECONDITION", "EMBEDDER_FAILED", "EMBEDDER_UNAVAILABLE", "EMBEDDER_TIMEOUT",
        "VECTOR_SEARCH_FAILED", "VECTOR_SEARCH_PARTIAL", "VECTOR_SEARCH_TIMEOUT", "SPACE_INACCESSIBLE",
        "SPACE_NOT_FOUND", "SPACE_NO_EMBEDDERS", "CHUNK_NOT_FOUND", "MEMORY_LOAD_FAILED",
        "MEMORY_CONTENT_UNAVAILABLE", "RERANKING_FAILED", "SUMMARIZATION_FAILED", "SUMMARIZATION_TIMEOUT",
        "RATE_LIMITED", "RESOURCE_EXHAUSTED", "CONFIGURATION_ERROR", "LLM_CAPABILITY_INFERRED",
        "FEATURE_DISABLED",
    };

    /// <summary>
    /// Returns the status to report, or <see langword="null"/> for an informational notice.
    /// </summary>
    /// <remarks>
    /// Never throws: a code this connector does not know, or no code at all, is reported as
    /// <c>UNKNOWN</c> (contract Q3), so a newer server cannot make a failure look like success.
    /// </remarks>
    internal static GoodMemRetrievalStatus? Classify(JsonNode? status)
    {
        var fields = status as JsonObject;
        var code = Text(fields?["code"]);
        if (code is not null && s_informational.Contains(code))
            return null;

        var message = Text(fields?["message"]) ?? "";
        var details = new Dictionary<string, string>();
        if (fields?["details"] is JsonObject detailFields)
            foreach (var (key, value) in detailFields)
                details[key] = Text(value) ?? "";

        return code is not null && s_known.Contains(code)
            ? new GoodMemRetrievalStatus(code, message, details)
            : new GoodMemRetrievalStatus(Unknown, message, details, originalCode: code, unrecognized: true);
    }

    /// <summary>A line of the stream that could not be read, reported like a server status.</summary>
    internal static GoodMemRetrievalStatus Malformed(string message) =>
        new(MalformedStream, message);

    /// <summary>
    /// One line naming every status, for a log message. Line breaks in the server's text are
    /// flattened so a message cannot start a log line of its own.
    /// </summary>
    internal static string Describe(IEnumerable<GoodMemRetrievalStatus> statuses) =>
        ("[" + string.Join("; ", statuses) + "]").Replace('\r', ' ').Replace('\n', ' ');

    private static string? Text(JsonNode? node) => node switch
    {
        null => null,
        JsonValue value when value.TryGetValue<string>(out var text) => text,
        _ => node.ToJsonString(),
    };
}
