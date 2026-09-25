using Xunit;

namespace GoodMem.SemanticKernel.Tests.Unit;

public sealed class GoodMemIdsTests
{
    private const string U = "0198c3a2-7f4e-7c1a-9b2d-5e6f7a8b9c0d";

    [Fact]
    public void RequireUuid_NormalisesToLowercase()
    {
        Assert.Equal(U, GoodMemIds.RequireUuid(U.ToUpperInvariant(), "key"));
    }

    [Theory]
    [MemberData(nameof(GoodMemIdValidationTests.Payloads), MemberType = typeof(GoodMemIdValidationTests))]
    [InlineData(null)]
    [InlineData(U + " ")]
    [InlineData("0198c3a27f4e7c1a9b2d5e6f7a8b9c0d")]
    [InlineData("{0198c3a2-7f4e-7c1a-9b2d-5e6f7a8b9c0d}")]
    [InlineData("0198c3a2-7f4e-7c1a-9b2d-5e6f7a8b9c0١")] // Arabic-Indic digit one
    public void RequireUuid_RefusesEverythingElse_NamingTheField(string? payload)
    {
        var error = Assert.Throws<ArgumentException>(() => GoodMemIds.RequireUuid(payload, "key"));
        Assert.StartsWith("key must be a GoodMem UUID", error.Message);
        Assert.Equal("key", error.ParamName);
    }
}
