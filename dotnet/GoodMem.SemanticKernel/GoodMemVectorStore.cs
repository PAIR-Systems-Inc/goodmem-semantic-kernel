using System.Runtime.CompilerServices;
using Microsoft.Extensions.VectorData;

namespace GoodMem.SemanticKernel;

/// <summary>
/// Semantic Kernel VectorStore backed by GoodMem.
/// Acts as a factory for <see cref="GoodMemCollection{TRecord}"/> instances and provides
/// enumeration of available spaces (collections). A single underlying HTTP client
/// is shared across all collections created from this store.
/// </summary>
/// <example>
/// <code>
/// using var store = new GoodMemVectorStore();
/// var notes = store.GetCollection&lt;string, Note&gt;("notes");
/// var todos = store.GetCollection&lt;string, Note&gt;("todos");
/// await notes.EnsureCollectionExistsAsync();
/// </code>
/// </example>
public sealed class GoodMemVectorStore : VectorStore
{
    private readonly GoodMemClient _client;
    private readonly GoodMemOptions _options;

    /// <summary>
    /// Initializes a new GoodMemVectorStore. Reads GOODMEM_* environment variables by default.
    /// </summary>
    /// <param name="options">Optional configuration. Defaults to reading GOODMEM_* env vars.</param>
    public GoodMemVectorStore(GoodMemOptions? options = null)
    {
        _options = options ?? new GoodMemOptions();
        _client = new GoodMemClient(_options.BaseUrl, _options.ApiKey, _options.VerifySsl);
    }

    /// <inheritdoc/>
    protected override void Dispose(bool disposing)
    {
        if (disposing) _client.Dispose();
        base.Dispose(disposing);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Only <see cref="string"/> keys are supported. Requesting any other key type throws
    /// <see cref="ArgumentException"/>.
    /// </remarks>
    public override VectorStoreCollection<TKey, TRecord> GetCollection<TKey, TRecord>(
        string name,
        VectorStoreCollectionDefinition? definition = null)
    {
        if (typeof(TKey) != typeof(string))
            throw new ArgumentException(
                $"GoodMem only supports string keys. Requested key type: {typeof(TKey).Name}.");

        if (typeof(TRecord) == typeof(System.Collections.Generic.Dictionary<string, object?>))
            throw new ArgumentException(
                "Dynamic (Dictionary) collections are not supported. Use GetCollection<string, YourRecordType>() instead.");

        var opts = CloneOptionsWithDefinition(definition);
        var collection = new GoodMemCollection<TRecord>(name, _client, opts);
        return (VectorStoreCollection<TKey, TRecord>)(object)collection;
    }

    /// <inheritdoc/>
    /// <exception cref="NotSupportedException">Dynamic collections are not supported by GoodMem.</exception>
    public override VectorStoreCollection<object, System.Collections.Generic.Dictionary<string, object?>> GetDynamicCollection(
        string name,
        VectorStoreCollectionDefinition definition)
        => throw new NotSupportedException(
            "Dynamic collections are not supported by the GoodMem SK connector. " +
            "Use GetCollection<string, YourRecordType>() with a typed record class.");

    /// <inheritdoc/>
    public override async IAsyncEnumerable<string> ListCollectionNamesAsync(
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var spaces = await _client.ListSpacesAsync(ct: cancellationToken).ConfigureAwait(false);
        foreach (var space in spaces)
            if (space["name"]?.GetValue<string>() is string name)
                yield return name;
    }

    /// <inheritdoc/>
    public override async Task<bool> CollectionExistsAsync(
        string name,
        CancellationToken cancellationToken = default)
    {
        var spaces = await _client.ListSpacesAsync(nameFilter: name, ct: cancellationToken).ConfigureAwait(false);
        return spaces.Any(s => s["name"]?.GetValue<string>() == name);
    }

    /// <inheritdoc/>
    public override async Task EnsureCollectionDeletedAsync(
        string name,
        CancellationToken cancellationToken = default)
    {
        var spaces = await _client.ListSpacesAsync(nameFilter: name, ct: cancellationToken).ConfigureAwait(false);
        foreach (var space in spaces)
        {
            if (space["name"]?.GetValue<string>() == name && space["spaceId"]?.GetValue<string>() is string sid)
            {
                await _client.DeleteSpaceAsync(sid, cancellationToken).ConfigureAwait(false);
                return;
            }
        }
    }

    /// <inheritdoc/>
    public override object? GetService(Type serviceType, object? serviceKey = null)
    {
        if (serviceKey is not null) return null;
        if (serviceType == typeof(VectorStoreMetadata))
            return new VectorStoreMetadata { VectorStoreSystemName = "GoodMem" };
        if (serviceType.IsInstanceOfType(this)) return this;
        return null;
    }

    private GoodMemOptions CloneOptionsWithDefinition(VectorStoreCollectionDefinition? definition)
    {
        return new GoodMemOptions
        {
            BaseUrl = _options.BaseUrl,
            ApiKey = _options.ApiKey,
            EmbedderId = _options.EmbedderId,
            VerifySsl = _options.VerifySsl,
            Definition = definition ?? _options.Definition,
        };
    }
}
