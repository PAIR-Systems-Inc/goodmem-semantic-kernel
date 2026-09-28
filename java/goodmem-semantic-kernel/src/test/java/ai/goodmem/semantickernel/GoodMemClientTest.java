package ai.goodmem.semantickernel;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.github.tomakehurst.wiremock.junit5.WireMockRuntimeInfo;
import com.github.tomakehurst.wiremock.junit5.WireMockTest;
import org.junit.jupiter.api.Test;

import java.util.List;

import static com.github.tomakehurst.wiremock.client.WireMock.*;
import static org.assertj.core.api.Assertions.*;

/**
 * Unit tests for {@link GoodMemClient} — each test spins up a WireMock HTTP
 * server on a random port and verifies that the client sends the correct
 * requests and correctly parses the responses.
 */
@WireMockTest
class GoodMemClientTest {

    // GoodMem ids are UUIDs and the client refuses anything else in a path.
    private static final String SPACE_ID = "01a0d16b-bbcd-701c-bfb4-fa306021e078";
    private static final String MEMORY_ID = "019cfd1d-5a1e-7a4b-9c3e-2f6a1b0c0e01";

    // ── Factory ───────────────────────────────────────────────────────────────

    private GoodMemClient client(WireMockRuntimeInfo wm) {
        return new GoodMemClient(
                GoodMemOptions.builder()
                        .baseUrl("http://localhost:" + wm.getHttpPort())
                        .apiKey("test-key")
                        .build());
    }

    // ── listSpaces ────────────────────────────────────────────────────────────

    @Test
    void listSpaces_empty(WireMockRuntimeInfo wm) {
        stubFor(get(urlPathEqualTo("/v1/spaces"))
                .willReturn(okJson("""
                        {"spaces":[]}
                        """)));

        List<ObjectNode> spaces = client(wm).listSpaces(null).block();

        assertThat(spaces).isEmpty();
    }

    @Test
    void listSpaces_returnsItems(WireMockRuntimeInfo wm) {
        stubFor(get(urlPathEqualTo("/v1/spaces"))
                .willReturn(okJson("""
                        {"spaces":[{"spaceId":"s1","name":"alpha"},{"spaceId":"s2","name":"beta"}]}
                        """)));

        List<ObjectNode> spaces = client(wm).listSpaces(null).block();

        assertThat(spaces).hasSize(2);
        assertThat(spaces.get(0).path("name").asText()).isEqualTo("alpha");
        assertThat(spaces.get(1).path("spaceId").asText()).isEqualTo("s2");
    }

    @Test
    void listSpaces_withNameFilter_sendsQueryParam(WireMockRuntimeInfo wm) {
        stubFor(get(urlPathEqualTo("/v1/spaces"))
                .withQueryParam("nameFilter", equalTo("my-space"))
                .willReturn(okJson("""
                        {"spaces":[{"spaceId":"s1","name":"my-space"}]}
                        """)));

        List<ObjectNode> spaces = client(wm).listSpaces("my-space").block();

        assertThat(spaces).hasSize(1);
    }

    @Test
    void listSpaces_pagination(WireMockRuntimeInfo wm) {
        // First page: has nextToken. Second page: no nextToken.
        stubFor(get(urlPathEqualTo("/v1/spaces"))
                .withQueryParam("maxResults", equalTo("1000"))
                .withQueryParam("nextToken", absent())
                .willReturn(okJson("""
                        {"spaces":[{"spaceId":"s1","name":"a"}],"nextToken":"tok-1"}
                        """)));
        stubFor(get(urlPathEqualTo("/v1/spaces"))
                .withQueryParam("nextToken", equalTo("tok-1"))
                .willReturn(okJson("""
                        {"spaces":[{"spaceId":"s2","name":"b"}]}
                        """)));

        List<ObjectNode> spaces = client(wm).listSpaces(null).block();

        assertThat(spaces).hasSize(2);
    }

    // ── createSpace ───────────────────────────────────────────────────────────

    @Test
    void createSpace_postsCorrectPayload(WireMockRuntimeInfo wm) {
        stubFor(post(urlEqualTo("/v1/spaces"))
                .withRequestBody(matchingJsonPath("$.name", equalTo("new-space")))
                .withRequestBody(matchingJsonPath("$.spaceEmbedders[0].embedderId", equalTo("emb-1")))
                .willReturn(okJson("""
                        {"spaceId":"space-new","name":"new-space"}
                        """)));

        ObjectNode result = client(wm).createSpace("new-space", "emb-1", null).block();

        assertThat(result).isNotNull();
        assertThat(result.path("spaceId").asText()).isEqualTo("space-new");
    }

    // ── deleteSpace ───────────────────────────────────────────────────────────

    @Test
    void deleteSpace_sends204(WireMockRuntimeInfo wm) {
        stubFor(delete(urlEqualTo("/v1/spaces/" + SPACE_ID))
                .willReturn(noContent()));

        assertThatCode(() -> client(wm).deleteSpace(SPACE_ID).block())
                .doesNotThrowAnyException();
    }

    // ── listEmbedders ─────────────────────────────────────────────────────────

    @Test
    void listEmbedders_returnsItems(WireMockRuntimeInfo wm) {
        stubFor(get(urlEqualTo("/v1/embedders"))
                .willReturn(okJson("""
                        {"embedders":[{"embedderId":"emb-1","displayName":"text-embedding-3-small"}]}
                        """)));

        List<ObjectNode> embedders = client(wm).listEmbedders().block();

        assertThat(embedders).hasSize(1);
        assertThat(embedders.get(0).path("embedderId").asText()).isEqualTo("emb-1");
    }

    // ── createMemory ──────────────────────────────────────────────────────────

    @Test
    void createMemory_postsCorrectPayload(WireMockRuntimeInfo wm) {
        stubFor(post(urlEqualTo("/v1/memories"))
                .withRequestBody(matchingJsonPath("$.spaceId", equalTo("space-1")))
                .withRequestBody(matchingJsonPath("$.originalContent", equalTo("hello world")))
                .withRequestBody(matchingJsonPath("$.contentType", equalTo("text/plain")))
                .willReturn(okJson("""
                        {"memoryId":"mem-new"}
                        """)));

        ObjectNode result = client(wm).createMemory("space-1", "hello world", null, null, null).block();

        assertThat(result).isNotNull();
        assertThat(result.path("memoryId").asText()).isEqualTo("mem-new");
    }

    @Test
    void createMemory_includesMetadata(WireMockRuntimeInfo wm) {
        stubFor(post(urlEqualTo("/v1/memories"))
                .withRequestBody(matchingJsonPath("$.metadata.Topic", equalTo("science")))
                .willReturn(okJson("""
                        {"memoryId":"mem-meta"}
                        """)));

        ObjectNode result = client(wm)
                .createMemory("space-1", "content", null, java.util.Map.of("Topic", "science"), null)
                .block();

        assertThat(result.path("memoryId").asText()).isEqualTo("mem-meta");
    }

    // ── batchGetMemories ──────────────────────────────────────────────────────

    @Test
    void batchGetMemories_returnsEmptyList_whenInputEmpty(WireMockRuntimeInfo wm) {
        // No HTTP call should be made.
        List<ObjectNode> result = client(wm).batchGetMemories(List.of()).block();
        assertThat(result).isEmpty();
    }

    @Test
    void batchGetMemories_unwrapsResultsArray(WireMockRuntimeInfo wm) {
        stubFor(post(urlEqualTo("/v1/memories:batchGet"))
                .willReturn(okJson("""
                        {"results":[
                          {"success":true,"memory":{"memoryId":"m1","originalContent":"hello","metadata":{"Topic":"geo"}}},
                          {"success":false,"memoryId":"m2","error":{"message":"not found"}}
                        ]}
                        """)));

        List<ObjectNode> memories = client(wm).batchGetMemories(List.of("m1", "m2")).block();

        // Only successful items with a "memory" are returned.
        assertThat(memories).hasSize(1);
        assertThat(memories.get(0).path("memoryId").asText()).isEqualTo("m1");
        assertThat(memories.get(0).path("originalContent").asText()).isEqualTo("hello");
    }

    @Test
    void batchGetMemories_sendsIncludeContent(WireMockRuntimeInfo wm) {
        stubFor(post(urlEqualTo("/v1/memories:batchGet"))
                .withRequestBody(matchingJsonPath("$.includeContent", equalTo("true")))
                .willReturn(okJson("{\"results\":[]}")));

        List<ObjectNode> result = client(wm).batchGetMemories(List.of("m1")).block();
        assertThat(result).isEmpty();
    }

    // ── deleteMemory ──────────────────────────────────────────────────────────

    @Test
    void deleteMemory_success(WireMockRuntimeInfo wm) {
        stubFor(delete(urlEqualTo("/v1/memories/" + MEMORY_ID))
                .willReturn(noContent()));

        assertThatCode(() -> client(wm).deleteMemory(MEMORY_ID).block())
                .doesNotThrowAnyException();
    }

    @Test
    void deleteMemory_notFound_doesNotThrow(WireMockRuntimeInfo wm) {
        stubFor(delete(urlEqualTo("/v1/memories/" + MEMORY_ID))
                .willReturn(notFound()));

        // 404 is silently swallowed — idempotent delete.
        assertThatCode(() -> client(wm).deleteMemory(MEMORY_ID).block())
                .doesNotThrowAnyException();
    }

    // ── retrieveMemories (NDJSON) ─────────────────────────────────────────────

    @Test
    void retrieveMemories_parsesNdjson(WireMockRuntimeInfo wm) {
        String ndjson = """
                {"memoryDefinition":{"memoryId":"m1","metadata":{"Topic":"geography"}}}
                {"retrievedItem":{"chunk":{"chunk":{"chunkText":"Paris is in France","memoryId":"m1"},"relevanceScore":0.2,"memoryIndex":0}}}
                """;

        stubFor(post(urlEqualTo("/v1/memories:retrieve"))
                .withHeader("Accept", equalTo("application/x-ndjson"))
                .willReturn(ok(ndjson).withHeader("Content-Type", "application/x-ndjson")));

        List<GoodMemClient.RetrieveResult> results =
                client(wm).retrieveMemories("capital cities", List.of("space-1"), 3).block();

        assertThat(results).hasSize(1);
        GoodMemClient.RetrieveResult r = results.get(0);
        assertThat(r.chunk().path("chunkText").asText()).isEqualTo("Paris is in France");
        assertThat(r.memory().path("memoryId").asText()).isEqualTo("m1");
        // Score is negated: 0.2 (distance) → -0.2 (similarity)
        assertThat(r.score()).isEqualTo(-0.2, within(1e-6));
    }

    @Test
    void retrieveMemories_emptyResponse_returnsEmptyList(WireMockRuntimeInfo wm) {
        stubFor(post(urlEqualTo("/v1/memories:retrieve"))
                .willReturn(ok("").withHeader("Content-Type", "application/x-ndjson")));

        List<GoodMemClient.RetrieveResult> results =
                client(wm).retrieveMemories("query", List.of("space-1"), 5).block();

        assertThat(results).isEmpty();
    }

    // ── Error handling ────────────────────────────────────────────────────────

    @Test
    void apiError_throwsGoodMemException(WireMockRuntimeInfo wm) {
        stubFor(get(urlPathEqualTo("/v1/spaces"))
                .willReturn(serverError().withBody("internal error")));

        assertThatThrownBy(() -> client(wm).listSpaces(null).block())
                .isInstanceOf(GoodMemException.class)
                .hasMessageContaining("500");
    }

    @Test
    void deleteSpace_nonNotFoundError_throws(WireMockRuntimeInfo wm) {
        stubFor(delete(urlPathMatching("/v1/spaces/.*"))
                .willReturn(serverError()));

        assertThatThrownBy(() -> client(wm).deleteSpace(SPACE_ID).block())
                .isInstanceOf(GoodMemException.class);
    }
}
