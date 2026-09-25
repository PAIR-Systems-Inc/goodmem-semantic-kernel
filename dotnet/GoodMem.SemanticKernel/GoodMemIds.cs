using System.Text.RegularExpressions;

namespace GoodMem.SemanticKernel;

/// <summary>
/// The one check every GoodMem id passes before the connector sends it.
/// </summary>
/// <remarks>
/// Ids become URL path segments (<c>v1/memories/{id}</c>). Escaping them is not
/// enough: the GoodMem server decodes <c>%2e%2e</c> back into <c>..</c>, so an id
/// such as <c>../spaces/&lt;uuid&gt;</c> could turn a memory delete into a space
/// delete. GoodMem ids (memories, spaces, embedders) are all UUIDs, so anything
/// that is not one is refused before a request is made with it.
/// </remarks>
internal static class GoodMemIds
{
    // \A and \z, not ^ and $: in .NET, $ also matches before a trailing newline.
    // Explicit hex classes, not \d, which matches any Unicode digit.
    private static readonly Regex s_uuid = new(
        @"\A[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\z",
        RegexOptions.CultureInvariant);

    /// <summary>
    /// Returns <paramref name="value"/> as a lowercase canonical UUID, or throws
    /// <see cref="ArgumentException"/> naming <paramref name="field"/>.
    /// </summary>
    internal static string RequireUuid(string? value, string field)
    {
        if (value is not null && s_uuid.IsMatch(value))
            return value.ToLowerInvariant();

        var shown = value is null ? "null" : $"'{(value.Length > 64 ? value[..61] + "..." : value)}'";
        throw new ArgumentException(
            $"{field} must be a GoodMem UUID, got {shown}. It was not sent: GoodMem ids are UUIDs, " +
            "and any other value can change which URL a request goes to.",
            field);
    }
}
