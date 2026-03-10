using System.Linq.Expressions;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text.Json.Nodes;
using Microsoft.Extensions.VectorData;

namespace GoodMem.SemanticKernel;

/// <summary>
/// Semantic Kernel VectorStoreCollection backed by GoodMem.
/// Maps SK collections to GoodMem Spaces (1:1). The collection name is the
/// human-readable space name. Embedding is always performed server-side.
/// </summary>
/// <typeparam name="TRecord">
/// The data model class. Must have:
/// <list type="bullet">
///   <item>One property marked <c>[VectorStoreKey]</c> of type <c>string?</c></item>
///   <item>At least one property marked <c>[VectorStoreData]</c> — the first <c>string</c>
///         property (or the one named <c>Content</c>) becomes the GoodMem <c>originalContent</c>;
///         remaining data properties are stored in <c>metadata</c></item>
///   <item>A public parameterless constructor (or settable properties)</item>
/// </list>
/// </typeparam>
/// <example>
/// <code>
/// public class Memory
/// {
///     [VectorStoreKey] public string? Id { get; set; }
///     [VectorStoreData] public string Content { get; set; } = "";
///     [VectorStoreData] public string? Topic { get; set; }
///     [VectorStoreVector(Dimensions: 1536)] public ReadOnlyMemory&lt;float&gt;? Embedding { get; set; }
/// }
///
/// using var collection = new GoodMemCollection&lt;Memory&gt;("agent-memory");
/// await collection.EnsureCollectionExistsAsync();
/// await collection.UpsertAsync(new Memory { Content = "Hello, world!", Topic = "greetings" });
/// </code>
/// </example>
public sealed class GoodMemCollection<TRecord> : VectorStoreCollection<string, TRecord>
    where TRecord : class
{
    private readonly GoodMemClient _client;
    private readonly bool _ownsClient;
    private readonly GoodMemOptions _options;
    private readonly GoodMemSchema _schema;

    // Lazily resolved space UUID for this collection name.
    private string? _spaceId;

    /// <inheritdoc/>
    public override string Name { get; }

    /// <summary>
    /// Initializes a new GoodMemCollection that creates and manages its own HTTP client.
    /// </summary>
    /// <param name="collectionName">The GoodMem space name (collection name).</param>
    /// <param name="options">Optional configuration. Reads GOODMEM_* env vars by default.</param>
    public GoodMemCollection(string collectionName, GoodMemOptions? options = null)
        : this(collectionName, client: null, options)
    {
    }

    /// <summary>
    /// Initializes a new GoodMemCollection using an existing <see cref="GoodMemClient"/>.
    /// The caller owns the client lifetime when injecting it directly.
    /// </summary>
    internal GoodMemCollection(string collectionName, GoodMemClient? client, GoodMemOptions? options = null)
    {
        Name = collectionName;
        _options = options ?? new GoodMemOptions();
        _schema = GoodMemSchema.Build(typeof(TRecord), _options.Definition);
        _ownsClient = client is null;
        _client = client ?? new GoodMemClient(_options.BaseUrl, _options.ApiKey, _options.VerifySsl);
    }

    /// <inheritdoc/>
    protected override void Dispose(bool disposing)
    {
        if (disposing && _ownsClient)
            _client.Dispose();
    }

    // ------------------------------------------------------------------
    // Collection lifecycle
    // ------------------------------------------------------------------

    /// <inheritdoc/>
    public override async Task<bool> CollectionExistsAsync(CancellationToken cancellationToken = default)
    {
        var spaces = await _client.ListSpacesAsync(nameFilter: Name, ct: cancellationToken).ConfigureAwait(false);
        return spaces.Any(s => s["name"]?.GetValue<string>() == Name);
    }

    /// <inheritdoc/>
    public override async Task EnsureCollectionExistsAsync(CancellationToken cancellationToken = default)
    {
        await ResolveSpaceIdAsync(cancellationToken).ConfigureAwait(false);
    }

    /// <inheritdoc/>
    public override async Task EnsureCollectionDeletedAsync(CancellationToken cancellationToken = default)
    {
        var spaces = await _client.ListSpacesAsync(nameFilter: Name, ct: cancellationToken).ConfigureAwait(false);
        foreach (var space in spaces)
        {
            if (space["name"]?.GetValue<string>() == Name && space["spaceId"]?.GetValue<string>() is string sid)
            {
                await _client.DeleteSpaceAsync(sid, cancellationToken).ConfigureAwait(false);
                _spaceId = null;
                return;
            }
        }
    }

    // ------------------------------------------------------------------
    // CRUD
    // ------------------------------------------------------------------

    /// <inheritdoc/>
    public override async Task<TRecord?> GetAsync(
        string key,
        RecordRetrievalOptions? options = default,
        CancellationToken cancellationToken = default)
    {
        var memories = await _client.BatchGetMemoriesAsync([key], cancellationToken).ConfigureAwait(false);
        if (memories.Count == 0) return null;
        return _schema.Deserialize<TRecord>(memories[0]);
    }

    /// <inheritdoc/>
    public override async IAsyncEnumerable<TRecord> GetAsync(
        IEnumerable<string> keys,
        RecordRetrievalOptions? options = default,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var keyList = keys.ToList();
        if (keyList.Count == 0) yield break;

        var memories = await _client.BatchGetMemoriesAsync(keyList, cancellationToken).ConfigureAwait(false);
        foreach (var mem in memories)
        {
            var record = _schema.Deserialize<TRecord>(mem);
            if (record is not null) yield return record;
        }
    }

    /// <inheritdoc/>
    public override async Task DeleteAsync(string key, CancellationToken cancellationToken = default)
    {
        await _client.DeleteMemoryAsync(key, cancellationToken).ConfigureAwait(false);
    }

    /// <inheritdoc/>
    public override async Task DeleteAsync(IEnumerable<string> keys, CancellationToken cancellationToken = default)
    {
        foreach (var key in keys)
            await _client.DeleteMemoryAsync(key, cancellationToken).ConfigureAwait(false);
    }

    /// <inheritdoc/>
    public override async Task UpsertAsync(TRecord record, CancellationToken cancellationToken = default)
    {
        var spaceId = await ResolveSpaceIdAsync(cancellationToken).ConfigureAwait(false);
        var (content, metadata, memoryId) = _schema.Serialize(record);

        if (memoryId is not null)
            await _client.DeleteMemoryAsync(memoryId, cancellationToken).ConfigureAwait(false);

        var result = await _client.CreateMemoryAsync(
            spaceId, content, metadata: metadata, memoryId: memoryId,
            ct: cancellationToken).ConfigureAwait(false);

        // Write back the server-generated key to the record if the property is settable.
        var returnedId = result["memoryId"]?.GetValue<string>() ?? memoryId;
        if (returnedId is not null)
            _schema.SetKey(record, returnedId);
    }

    /// <inheritdoc/>
    public override async Task UpsertAsync(IEnumerable<TRecord> records, CancellationToken cancellationToken = default)
    {
        var spaceId = await ResolveSpaceIdAsync(cancellationToken).ConfigureAwait(false);

        foreach (var record in records)
        {
            var (content, metadata, memoryId) = _schema.Serialize(record);

            if (memoryId is not null)
                await _client.DeleteMemoryAsync(memoryId, cancellationToken).ConfigureAwait(false);

            var result = await _client.CreateMemoryAsync(
                spaceId, content, metadata: metadata, memoryId: memoryId,
                ct: cancellationToken).ConfigureAwait(false);

            var returnedId = result["memoryId"]?.GetValue<string>() ?? memoryId;
            if (returnedId is not null)
                _schema.SetKey(record, returnedId);
        }
    }

    // ------------------------------------------------------------------
    // Search
    // ------------------------------------------------------------------

    /// <inheritdoc/>
    /// <remarks>
    /// Only text search (<typeparamref name="TInput"/> = <see cref="string"/>) is supported.
    /// GoodMem embeds the query server-side; pre-computed vectors are not accepted.
    /// </remarks>
    public override async IAsyncEnumerable<VectorSearchResult<TRecord>> SearchAsync<TInput>(
        TInput searchValue,
        int top,
        VectorSearchOptions<TRecord>? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        if (searchValue is not string query)
            throw new NotSupportedException(
                "GoodMem only supports text search. Pass a string value — GoodMem will embed it server-side.");

        if (options?.Filter is not null)
            throw new NotSupportedException(
                "Filter expressions are not supported by the GoodMem SK connector in v1. " +
                "Remove the filter or post-filter in application code.");

        var spaceId = await ResolveSpaceIdAsync(cancellationToken).ConfigureAwait(false);
        var results = await _client.RetrieveMemoriesAsync(query, [spaceId], top, cancellationToken).ConfigureAwait(false);

        foreach (var result in results)
        {
            var record = _schema.DeserializeFromRetrieve<TRecord>(result.Chunk, result.Memory);
            if (record is not null)
                yield return new VectorSearchResult<TRecord>(record, result.Score);
        }
    }

    // ------------------------------------------------------------------
    // Not supported in v1
    // ------------------------------------------------------------------

    /// <inheritdoc/>
    /// <exception cref="NotSupportedException">Always thrown — filter-based retrieval is not supported in v1.</exception>
    public override IAsyncEnumerable<TRecord> GetAsync(
        Expression<Func<TRecord, bool>> filter,
        int top,
        FilteredRecordRetrievalOptions<TRecord>? options = null,
        CancellationToken cancellationToken = default)
        => throw new NotSupportedException(
            "Filter-based record retrieval is not supported by the GoodMem SK connector in v1. " +
            "Use SearchAsync with a text query instead.");

    /// <inheritdoc/>
    public override object? GetService(Type serviceType, object? serviceKey = null)
    {
        if (serviceKey is not null) return null;
        if (serviceType.IsInstanceOfType(this)) return this;
        return null;
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    private async Task<string> ResolveSpaceIdAsync(CancellationToken ct)
    {
        if (_spaceId is not null) return _spaceId;

        var spaces = await _client.ListSpacesAsync(nameFilter: Name, ct: ct).ConfigureAwait(false);
        foreach (var space in spaces)
        {
            if (space["name"]?.GetValue<string>() == Name && space["spaceId"]?.GetValue<string>() is string existing)
            {
                _spaceId = existing;
                return _spaceId;
            }
        }

        // Space does not exist — create it.
        var embedderId = await ResolveEmbedderIdAsync(ct).ConfigureAwait(false);
        var created = await _client.CreateSpaceAsync(Name, embedderId, ct: ct).ConfigureAwait(false);
        _spaceId = created["spaceId"]?.GetValue<string>()
                   ?? throw new InvalidOperationException("GoodMem did not return a spaceId after creating the space.");
        return _spaceId;
    }

    private async Task<string> ResolveEmbedderIdAsync(CancellationToken ct)
    {
        if (_options.EmbedderId is { Length: > 0 } configured)
            return configured;

        var embedders = await _client.ListEmbeddersAsync(ct).ConfigureAwait(false);
        if (embedders.Count > 0 && embedders[0]["embedderId"]?.GetValue<string>() is string eid)
            return eid;

        throw new InvalidOperationException(
            "No embedders are configured in GoodMem. " +
            "Create at least one embedder via the GoodMem API or set GOODMEM_EMBEDDER_ID.");
    }
}

// ------------------------------------------------------------------
// Schema helper — inspects TRecord attributes and drives serialization.
// ------------------------------------------------------------------

internal sealed class GoodMemSchema
{
    private readonly PropertyInfo _keyProp;
    private readonly PropertyInfo _contentProp;
    private readonly IReadOnlyList<(PropertyInfo Prop, string StorageName)> _metaProps;

    private GoodMemSchema(
        PropertyInfo keyProp,
        PropertyInfo contentProp,
        IReadOnlyList<(PropertyInfo, string)> metaProps)
    {
        _keyProp = keyProp;
        _contentProp = contentProp;
        _metaProps = metaProps;
    }

    /// <summary>
    /// Builds a <see cref="GoodMemSchema"/> for <paramref name="recordType"/> using either
    /// the explicit <paramref name="definition"/> or attribute-based reflection.
    /// </summary>
    internal static GoodMemSchema Build(Type recordType, VectorStoreCollectionDefinition? definition)
    {
        if (definition is not null)
            return BuildFromDefinition(recordType, definition);

        return BuildFromAttributes(recordType);
    }

    private static GoodMemSchema BuildFromAttributes(Type recordType)
    {
        var props = recordType.GetProperties(BindingFlags.Public | BindingFlags.Instance);

        PropertyInfo? keyProp = null;
        var dataPropsList = new List<(PropertyInfo, string)>();

        foreach (var prop in props)
        {
            if (prop.GetCustomAttribute<VectorStoreKeyAttribute>() is not null)
            {
                keyProp = prop;
                continue;
            }

            if (prop.GetCustomAttribute<VectorStoreDataAttribute>() is VectorStoreDataAttribute dataAttr)
            {
                var storageName = dataAttr.StorageName ?? prop.Name;
                dataPropsList.Add((prop, storageName));
            }

            // VectorStoreVector properties are intentionally ignored — GoodMem embeds server-side.
        }

        if (keyProp is null)
            throw new InvalidOperationException(
                $"Record type '{recordType.Name}' has no property marked with [VectorStoreKey]. " +
                "Add [VectorStoreKey] to the property that should map to the GoodMem memory ID.");

        if (dataPropsList.Count == 0)
            throw new InvalidOperationException(
                $"Record type '{recordType.Name}' has no properties marked with [VectorStoreData]. " +
                "At least one [VectorStoreData] property is required to provide content for GoodMem.");

        var contentProp = ResolveContentProp(dataPropsList);
        var metaProps = dataPropsList
            .Where(p => p.Item1 != contentProp)
            .ToList();

        return new GoodMemSchema(keyProp, contentProp, metaProps);
    }

    private static GoodMemSchema BuildFromDefinition(Type recordType, VectorStoreCollectionDefinition definition)
    {
        var props = recordType.GetProperties(BindingFlags.Public | BindingFlags.Instance)
            .ToDictionary(p => p.Name, StringComparer.OrdinalIgnoreCase);

        var keyProperty = definition.Properties.OfType<VectorStoreKeyProperty>().FirstOrDefault()
            ?? throw new InvalidOperationException("VectorStoreCollectionDefinition has no key property.");

        if (!props.TryGetValue(keyProperty.Name, out var keyProp))
            throw new InvalidOperationException(
                $"Key property '{keyProperty.Name}' not found on '{recordType.Name}'.");

        var dataProperties = definition.Properties.OfType<VectorStoreDataProperty>().ToList();
        if (dataProperties.Count == 0)
            throw new InvalidOperationException("VectorStoreCollectionDefinition has no data properties.");

        var dataPropsList = new List<(PropertyInfo, string)>();
        foreach (var dp in dataProperties)
        {
            if (props.TryGetValue(dp.Name, out var prop))
                dataPropsList.Add((prop, dp.StorageName ?? dp.Name));
        }

        var contentProp = ResolveContentProp(dataPropsList);
        var metaProps = dataPropsList.Where(p => p.Item1 != contentProp).ToList();

        return new GoodMemSchema(keyProp, contentProp, metaProps);
    }

    // Resolution order: name=="content" → first string data prop → first data prop
    private static PropertyInfo ResolveContentProp(List<(PropertyInfo Prop, string StorageName)> dataPropsList)
    {
        foreach (var (prop, sname) in dataPropsList)
            if (string.Equals(sname, "content", StringComparison.OrdinalIgnoreCase) ||
                string.Equals(prop.Name, "content", StringComparison.OrdinalIgnoreCase))
                return prop;

        foreach (var (prop, _) in dataPropsList)
            if (prop.PropertyType == typeof(string))
                return prop;

        return dataPropsList[0].Prop;
    }

    // ------------------------------------------------------------------
    // Serialization
    // ------------------------------------------------------------------

    internal (string Content, Dictionary<string, object?>? Metadata, string? MemoryId) Serialize(object record)
    {
        var memoryId = _keyProp.GetValue(record) as string;
        var content = _contentProp.GetValue(record) as string ?? string.Empty;

        Dictionary<string, object?>? metadata = null;
        foreach (var (prop, storageName) in _metaProps)
        {
            var val = prop.GetValue(record);
            if (val is null) continue;
            metadata ??= [];
            metadata[storageName] = val;
        }

        return (content, metadata, memoryId);
    }

    internal void SetKey(object record, string key)
    {
        if (_keyProp.CanWrite)
            _keyProp.SetValue(record, key);
    }

    // ------------------------------------------------------------------
    // Deserialization from batchGet response
    // ------------------------------------------------------------------

    internal T? Deserialize<T>(JsonObject mem) where T : class
    {
        var instance = CreateInstance<T>();
        if (instance is null) return null;

        var memoryId = mem["memoryId"]?.GetValue<string>();
        if (memoryId is not null && _keyProp.CanWrite)
            _keyProp.SetValue(instance, memoryId);

        var content = mem["originalContent"]?.GetValue<string>() ?? string.Empty;
        if (_contentProp.CanWrite)
            _contentProp.SetValue(instance, content);

        var metadataNode = mem["metadata"];
        if (metadataNode is not null)
        {
            foreach (var (prop, storageName) in _metaProps)
            {
                if (!prop.CanWrite) continue;
                var valNode = metadataNode[storageName];
                if (valNode is null) continue;
                SetPropFromNode(instance, prop, valNode);
            }
        }

        return instance;
    }

    // ------------------------------------------------------------------
    // Deserialization from retrieve (search) response
    // GoodMem's originalContent is write-only; content comes from chunkText.
    // ------------------------------------------------------------------

    internal T? DeserializeFromRetrieve<T>(JsonObject chunk, JsonObject memory) where T : class
    {
        var instance = CreateInstance<T>();
        if (instance is null) return null;

        var memoryId = chunk["memoryId"]?.GetValue<string>() ?? memory["memoryId"]?.GetValue<string>();
        if (memoryId is not null && _keyProp.CanWrite)
            _keyProp.SetValue(instance, memoryId);

        // Content comes from chunk.chunkText, not originalContent.
        var content = chunk["chunkText"]?.GetValue<string>() ?? string.Empty;
        if (_contentProp.CanWrite)
            _contentProp.SetValue(instance, content);

        var metadataNode = memory["metadata"];
        if (metadataNode is not null)
        {
            foreach (var (prop, storageName) in _metaProps)
            {
                if (!prop.CanWrite) continue;
                var valNode = metadataNode[storageName];
                if (valNode is null) continue;
                SetPropFromNode(instance, prop, valNode);
            }
        }

        return instance;
    }

    private static T? CreateInstance<T>() where T : class
    {
        try { return (T?)RuntimeHelpers.GetUninitializedObject(typeof(T)); }
        catch { return null; }
    }

    private static void SetPropFromNode(object instance, PropertyInfo prop, JsonNode node)
    {
        try
        {
            var type = Nullable.GetUnderlyingType(prop.PropertyType) ?? prop.PropertyType;
            object? val = type switch
            {
                _ when type == typeof(string) => node.GetValue<string>(),
                _ when type == typeof(int) => node.GetValue<int>(),
                _ when type == typeof(long) => node.GetValue<long>(),
                _ when type == typeof(double) => node.GetValue<double>(),
                _ when type == typeof(float) => node.GetValue<float>(),
                _ when type == typeof(bool) => node.GetValue<bool>(),
                _ => null,
            };
            if (val is not null)
                prop.SetValue(instance, val);
        }
        catch { /* silently skip unparseable metadata fields */ }
    }
}
