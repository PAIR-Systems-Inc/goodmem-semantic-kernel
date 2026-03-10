using Microsoft.Extensions.VectorData;

namespace GoodMem.SemanticKernel;

/// <summary>
/// Configuration options for the GoodMem Semantic Kernel connector.
/// All settings can also be provided via environment variables with the GOODMEM_ prefix.
/// </summary>
public sealed class GoodMemOptions
{
    /// <summary>
    /// Base URL of the GoodMem server (no trailing slash).
    /// Defaults to the GOODMEM_BASE_URL environment variable, or http://localhost:8080.
    /// </summary>
    public string BaseUrl { get; set; } =
        Environment.GetEnvironmentVariable("GOODMEM_BASE_URL") ?? "http://localhost:8080";

    /// <summary>
    /// API key sent in the x-api-key header.
    /// Defaults to the GOODMEM_API_KEY environment variable.
    /// </summary>
    public string ApiKey { get; set; } =
        Environment.GetEnvironmentVariable("GOODMEM_API_KEY") ?? string.Empty;

    /// <summary>
    /// Embedder UUID to use when creating spaces. When null, the first available
    /// embedder is used automatically.
    /// Defaults to the GOODMEM_EMBEDDER_ID environment variable.
    /// </summary>
    public string? EmbedderId { get; set; } =
        Environment.GetEnvironmentVariable("GOODMEM_EMBEDDER_ID");

    /// <summary>
    /// Whether to validate TLS certificates. Set to false for local servers
    /// with self-signed certificates.
    /// Defaults to true, or the GOODMEM_VERIFY_SSL environment variable.
    /// </summary>
    public bool VerifySsl { get; set; } =
        !string.Equals(
            Environment.GetEnvironmentVariable("GOODMEM_VERIFY_SSL"),
            "false",
            StringComparison.OrdinalIgnoreCase);

    /// <summary>
    /// Optional schema definition for the collection. When provided, this takes
    /// precedence over attribute-based schema discovery on the record type.
    /// </summary>
    public VectorStoreCollectionDefinition? Definition { get; set; }
}
