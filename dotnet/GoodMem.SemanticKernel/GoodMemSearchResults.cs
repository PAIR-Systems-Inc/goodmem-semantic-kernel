using Microsoft.Extensions.VectorData;

namespace GoodMem.SemanticKernel;

/// <summary>
/// The results of a search, with any problem the GoodMem server reported while it ran.
/// Returned by <see cref="GoodMemCollection{TRecord}.SearchWithStatusAsync"/>.
/// </summary>
/// <typeparam name="TRecord">The record type of the collection.</typeparam>
public sealed class GoodMemSearchResults<TRecord>
{
    /// <summary>Creates a result set.</summary>
    public GoodMemSearchResults(
        IReadOnlyList<VectorSearchResult<TRecord>> results,
        IReadOnlyList<GoodMemRetrievalStatus> statuses)
    {
        Results = results;
        Statuses = statuses;
    }

    /// <summary>
    /// Every result the server returned, best first. A reported problem never removes results.
    /// </summary>
    public IReadOnlyList<VectorSearchResult<TRecord>> Results { get; }

    /// <summary>
    /// The problems the server reported. Informational notices (<c>FEATURE_DISABLED</c>,
    /// <c>LLM_CAPABILITY_INFERRED</c>) are not included.
    /// </summary>
    public IReadOnlyList<GoodMemRetrievalStatus> Statuses { get; }

    /// <summary>
    /// True when the server reported a real problem during this search, so
    /// <see cref="Results"/> may be incomplete, or empty when they should not be. It does not
    /// depend on whether any results came back.
    /// </summary>
    public bool Partial => Statuses.Count > 0;
}
