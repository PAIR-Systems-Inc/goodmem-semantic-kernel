using Microsoft.Extensions.VectorData;
using Xunit;

namespace GoodMem.SemanticKernel.Tests.Unit;

// ── Test record types ────────────────────────────────────────────────────────

// Standard record: key + content + metadata
file class NoteModel
{
    [VectorStoreKey] public string? Id { get; set; }
    [VectorStoreData] public string Content { get; set; } = "";
    [VectorStoreData] public string? Source { get; set; }
}

// No [VectorStoreKey]
file class NoKeyModel
{
    [VectorStoreData] public string Content { get; set; } = "";
}

// No [VectorStoreData]
file class NoDataModel
{
    [VectorStoreKey] public string? Id { get; set; }
}

// Content field resolved by first-string fallback (no property named "content")
file class AltContentModel
{
    [VectorStoreKey] public string? Id { get; set; }
    [VectorStoreData] public string Body { get; set; } = "";
    [VectorStoreData] public string? Tag { get; set; }
}

// Content field resolved by StorageName attribute
file class StorageNameModel
{
    [VectorStoreKey] public string? Id { get; set; }
    [VectorStoreData(StorageName = "content")] public string Text { get; set; } = "";
    [VectorStoreData] public string? Tag { get; set; }
}

// [VectorStoreVector] present — should be ignored by schema
file class WithVectorModel
{
    [VectorStoreKey] public string? Id { get; set; }
    [VectorStoreData] public string Content { get; set; } = "";
    [VectorStoreVector(Dimensions: 1536)] public ReadOnlyMemory<float>? Embedding { get; set; }
}

// Non-string data properties — content falls back to first data prop
file class NonStringContentModel
{
    [VectorStoreKey] public string? Id { get; set; }
    [VectorStoreData] public int Score { get; set; }
    [VectorStoreData] public bool Flag { get; set; }
}

public sealed class GoodMemSchemaTests
{
    // ── Build validation ─────────────────────────────────────────────────────

    [Fact]
    public void Build_NoKeyProperty_Throws()
    {
        var ex = Assert.Throws<InvalidOperationException>(
            () => GoodMemSchema.Build(typeof(NoKeyModel), definition: null));
        Assert.Contains("[VectorStoreKey]", ex.Message);
    }

    [Fact]
    public void Build_NoDataProperty_Throws()
    {
        var ex = Assert.Throws<InvalidOperationException>(
            () => GoodMemSchema.Build(typeof(NoDataModel), definition: null));
        Assert.Contains("[VectorStoreData]", ex.Message);
    }

    [Fact]
    public void Build_VectorPropertiesIgnored()
    {
        // Should not throw — [VectorStoreVector] is silently ignored.
        var schema = GoodMemSchema.Build(typeof(WithVectorModel), definition: null);
        Assert.NotNull(schema);
    }

    // ── Content field resolution ─────────────────────────────────────────────

    [Fact]
    public void Build_ContentResolution_PropertyNamedContent_Wins()
    {
        var schema = GoodMemSchema.Build(typeof(NoteModel), definition: null);
        var record = new NoteModel { Content = "hello", Source = "wiki" };

        var (content, metadata, _) = schema.Serialize(record);

        Assert.Equal("hello", content);
        Assert.NotNull(metadata);
        Assert.True(metadata!.ContainsKey("Source"));
    }

    [Fact]
    public void Build_ContentResolution_FirstStringFallback()
    {
        // AltContentModel has no "content"-named prop; Body is the first string → content
        var schema = GoodMemSchema.Build(typeof(AltContentModel), definition: null);
        var record = new AltContentModel { Body = "body text", Tag = "tag1" };

        var (content, metadata, _) = schema.Serialize(record);

        Assert.Equal("body text", content);
        Assert.True(metadata!.ContainsKey("Tag"));
    }

    [Fact]
    public void Build_ContentResolution_StorageNameContent_Wins()
    {
        var schema = GoodMemSchema.Build(typeof(StorageNameModel), definition: null);
        var record = new StorageNameModel { Text = "stored text", Tag = "t" };

        var (content, metadata, _) = schema.Serialize(record);

        Assert.Equal("stored text", content);
    }

    [Fact]
    public void Build_ContentResolution_FirstDataPropFallback_WhenNoString()
    {
        // NonStringContentModel: first data prop is int Score — should be used as content
        var schema = GoodMemSchema.Build(typeof(NonStringContentModel), definition: null);
        var record = new NonStringContentModel { Score = 42, Flag = true };

        var (content, metadata, _) = schema.Serialize(record);

        // Content from int prop → empty string (GetValue as string returns null → "")
        Assert.Equal(string.Empty, content);
    }

    // ── Serialize ─────────────────────────────────────────────────────────────

    [Fact]
    public void Serialize_NullId_ProducesNullMemoryId()
    {
        var schema = GoodMemSchema.Build(typeof(NoteModel), definition: null);
        var record = new NoteModel { Id = null, Content = "text" };

        var (_, _, memoryId) = schema.Serialize(record);

        Assert.Null(memoryId);
    }

    [Fact]
    public void Serialize_ExistingId_ProducesMemoryId()
    {
        var schema = GoodMemSchema.Build(typeof(NoteModel), definition: null);
        var record = new NoteModel { Id = "existing-id", Content = "text" };

        var (_, _, memoryId) = schema.Serialize(record);

        Assert.Equal("existing-id", memoryId);
    }

    [Fact]
    public void Serialize_NullMetadataFieldsOmitted()
    {
        var schema = GoodMemSchema.Build(typeof(NoteModel), definition: null);
        var record = new NoteModel { Content = "text", Source = null };

        var (_, metadata, _) = schema.Serialize(record);

        Assert.Null(metadata); // all meta fields were null → metadata dict not created
    }

    [Fact]
    public void Serialize_MetadataContainsNonNullFields()
    {
        var schema = GoodMemSchema.Build(typeof(NoteModel), definition: null);
        var record = new NoteModel { Content = "text", Source = "wiki" };

        var (_, metadata, _) = schema.Serialize(record);

        Assert.NotNull(metadata);
        Assert.Equal("wiki", metadata!["Source"]);
    }

    // ── Deserialize (from batchGet — uses originalContent) ───────────────────

    [Fact]
    public void Deserialize_UsesOriginalContent()
    {
        var schema = GoodMemSchema.Build(typeof(NoteModel), definition: null);
        var json = System.Text.Json.Nodes.JsonNode.Parse(
            """{"memoryId":"mem-1","originalContent":"batch content","metadata":{"Source":"wiki"}}""")!
            .AsObject();

        var record = schema.Deserialize<NoteModel>(json);

        Assert.NotNull(record);
        Assert.Equal("mem-1", record!.Id);
        Assert.Equal("batch content", record.Content);
        Assert.Equal("wiki", record.Source);
    }

    [Fact]
    public void Deserialize_MissingMetadata_DoesNotThrow()
    {
        var schema = GoodMemSchema.Build(typeof(NoteModel), definition: null);
        var json = System.Text.Json.Nodes.JsonNode.Parse(
            """{"memoryId":"mem-1","originalContent":"text"}""")!
            .AsObject();

        var record = schema.Deserialize<NoteModel>(json);

        Assert.NotNull(record);
        Assert.Null(record!.Source);
    }

    // ── DeserializeFromRetrieve (from search — uses chunkText) ───────────────

    [Fact]
    public void DeserializeFromRetrieve_UsesChunkText()
    {
        var schema = GoodMemSchema.Build(typeof(NoteModel), definition: null);
        var chunk = System.Text.Json.Nodes.JsonNode.Parse(
            """{"chunkText":"chunk content","memoryId":"mem-1"}""")!.AsObject();
        var memory = System.Text.Json.Nodes.JsonNode.Parse(
            """{"memoryId":"mem-1","metadata":{"Source":"web"}}""")!.AsObject();

        var record = schema.DeserializeFromRetrieve<NoteModel>(chunk, memory);

        Assert.NotNull(record);
        Assert.Equal("chunk content", record!.Content); // chunkText, not originalContent
        Assert.Equal("mem-1", record.Id);
        Assert.Equal("web", record.Source);
    }

    [Fact]
    public void DeserializeFromRetrieve_MemoryIdFromChunk()
    {
        var schema = GoodMemSchema.Build(typeof(NoteModel), definition: null);
        var chunk = System.Text.Json.Nodes.JsonNode.Parse(
            """{"chunkText":"text","memoryId":"chunk-mem-id"}""")!.AsObject();
        var memory = System.Text.Json.Nodes.JsonNode.Parse(
            """{"memoryId":"memory-mem-id"}""")!.AsObject();

        var record = schema.DeserializeFromRetrieve<NoteModel>(chunk, memory);

        Assert.Equal("chunk-mem-id", record!.Id); // chunk["memoryId"] wins
    }

    // ── SetKey ────────────────────────────────────────────────────────────────

    [Fact]
    public void SetKey_WritesKeyToRecord()
    {
        var schema = GoodMemSchema.Build(typeof(NoteModel), definition: null);
        var record = new NoteModel { Content = "text" };

        schema.SetKey(record, "new-key");

        Assert.Equal("new-key", record.Id);
    }

    // ── Definition-based schema ───────────────────────────────────────────────

    [Fact]
    public void Build_FromDefinition_CorrectSchema()
    {
        var definition = new VectorStoreCollectionDefinition
        {
            Properties =
            [
                new VectorStoreKeyProperty("Id", typeof(string)),
                new VectorStoreDataProperty("Content", typeof(string)),
                new VectorStoreDataProperty("Source", typeof(string)),
            ],
        };

        var schema = GoodMemSchema.Build(typeof(NoteModel), definition);
        var record = new NoteModel { Id = "d-1", Content = "def content", Source = "src" };

        var (content, metadata, memoryId) = schema.Serialize(record);

        Assert.Equal("def content", content);
        Assert.Equal("d-1", memoryId);
        Assert.Equal("src", metadata!["Source"]);
    }
}
