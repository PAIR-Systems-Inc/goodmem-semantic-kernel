using Microsoft.Extensions.VectorData;
using Xunit;

namespace GoodMem.SemanticKernel.Tests.Unit;

file class NoteRecord
{
    [VectorStoreKey] public string? Id { get; set; }
    [VectorStoreData] public string Content { get; set; } = "";
}

public sealed class GoodMemVectorStoreTests : IDisposable
{
    // Use a no-op base URL; these tests don't make HTTP calls.
    private readonly GoodMemVectorStore _store = new(new GoodMemOptions
    {
        BaseUrl = "http://localhost:8080",
        ApiKey = "test-key",
    });

    public void Dispose() => _store.Dispose();

    // ── GetCollection ────────────────────────────────────────────────────────

    [Fact]
    public void GetCollection_StringKey_ReturnsGoodMemCollection()
    {
        var collection = _store.GetCollection<string, NoteRecord>("notes");

        Assert.NotNull(collection);
        Assert.IsType<GoodMemCollection<NoteRecord>>(collection);
        Assert.Equal("notes", collection.Name);
    }

    [Fact]
    public void GetCollection_NonStringKey_ThrowsArgumentException()
    {
        var ex = Assert.Throws<ArgumentException>(
            () => _store.GetCollection<int, NoteRecord>("notes"));
        Assert.Contains("string", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void GetCollection_DictionaryRecord_ThrowsArgumentException()
    {
        var ex = Assert.Throws<ArgumentException>(
            () => _store.GetCollection<string, Dictionary<string, object?>>("dynamic"));
        Assert.Contains("dynamic", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    // ── GetDynamicCollection ─────────────────────────────────────────────────

    [Fact]
    public void GetDynamicCollection_ThrowsNotSupported()
    {
        var definition = new VectorStoreCollectionDefinition
        {
            Properties = [new VectorStoreKeyProperty("Id", typeof(string))],
        };

        Assert.Throws<NotSupportedException>(
            () => _store.GetDynamicCollection("dynamic", definition));
    }

    // ── GetService ────────────────────────────────────────────────────────────

    [Fact]
    public void GetService_VectorStoreMetadata_ReturnsMetadata()
    {
        var meta = _store.GetService(typeof(VectorStoreMetadata)) as VectorStoreMetadata;

        Assert.NotNull(meta);
        Assert.Equal("GoodMem", meta!.VectorStoreSystemName);
    }

    [Fact]
    public void GetService_Self_ReturnsSelf()
    {
        var result = _store.GetService(typeof(GoodMemVectorStore));

        Assert.Same(_store, result);
    }

    [Fact]
    public void GetService_UnknownType_ReturnsNull()
    {
        var result = _store.GetService(typeof(string));

        Assert.Null(result);
    }

    [Fact]
    public void GetService_WithServiceKey_ReturnsNull()
    {
        var result = _store.GetService(typeof(VectorStoreMetadata), serviceKey: "some-key");

        Assert.Null(result);
    }
}
