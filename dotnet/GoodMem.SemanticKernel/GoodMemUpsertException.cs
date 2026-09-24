using System;

namespace GoodMem.SemanticKernel;

/// <summary>
/// Thrown when an upsert fails while replacing an existing record.
/// </summary>
/// <remarks>
/// GoodMem has no update endpoint (<c>PUT</c> and <c>PATCH</c> return 404) and
/// rejects a create that reuses an existing id with 409, so replacing a record
/// means deleting the old memory before writing the new one. When the write
/// fails, the connector puts the previous version back; <see cref="Restored"/>
/// says whether that succeeded and <see cref="LostKey"/> names the record if it
/// did not.
/// </remarks>
public sealed class GoodMemUpsertException : Exception
{
    /// <summary>Whether the previous version was successfully restored.</summary>
    public bool Restored { get; }

    /// <summary>The key whose previous version could not be restored, if any.</summary>
    public string? LostKey { get; }

    /// <summary>Initialises the exception.</summary>
    public GoodMemUpsertException(string message, bool restored, string? lostKey, Exception? inner = null)
        : base(message, inner)
    {
        Restored = restored;
        LostKey = lostKey;
    }
}
