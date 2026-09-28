package ai.goodmem.semantickernel;

import com.github.tomakehurst.wiremock.junit5.WireMockRuntimeInfo;
import com.github.tomakehurst.wiremock.junit5.WireMockTest;
import org.junit.jupiter.api.Test;

import static com.github.tomakehurst.wiremock.client.WireMock.*;
import static org.assertj.core.api.Assertions.*;

/**
 * Upsert regressions for {@link GoodMemCollection}.
 *
 * <p>GoodMem memories cannot be updated in place, so an upsert of an existing
 * record deletes and re-creates it. Before this change the delete went first
 * unconditionally, so a create that failed afterwards destroyed the record.
 */
@WireMockTest
class GoodMemCollectionUpsertTest {

    // GoodMem ids are UUIDs and the connector refuses any other key.
    private static final String M1 = "019cfd1d-5a1e-7a4b-9c3e-2f6a1b0c0e01";
    private static final String M_NEW = "019cfd1d-5a1e-7a4b-9c3e-2f6a1b0c0e03";

    public static class Note {
        @GoodMemKey
        public String id;

        @GoodMemData
        public String content;

        @GoodMemData("tag")
        public String tag;

        Note(String id, String content, String tag) {
            this.id = id;
            this.content = content;
            this.tag = tag;
        }
    }

    private GoodMemCollection<Note> collection(WireMockRuntimeInfo wm) {
        return GoodMemCollection.of(
                "notes",
                Note.class,
                GoodMemOptions.builder()
                        .baseUrl("http://localhost:" + wm.getHttpPort())
                        .apiKey("test-key")
                        .build());
    }

    private void stubSpace() {
        stubFor(get(urlPathEqualTo("/v1/spaces"))
                .willReturn(okJson("""
                        {"spaces":[{"spaceId":"space-1","name":"notes"}]}
                        """)));
    }

    /** "old text", base64 — the shape batchGet really returns. */
    private static final String EXISTING = """
            {"results":[{"success":true,"memory":{"memoryId":"019cfd1d-5a1e-7a4b-9c3e-2f6a1b0c0e01",
            "originalContent":"b2xkIHRleHQ=","contentType":"text/plain",
            "metadata":{"tag":"hr"}}}]}
            """;

    @Test
    void existingRecordIsReadBeforeItIsDeleted(WireMockRuntimeInfo wm) {
        stubSpace();
        stubFor(post(urlPathEqualTo("/v1/memories:batchGet")).willReturn(okJson(EXISTING)));
        stubFor(delete(urlPathEqualTo("/v1/memories/" + M1)).willReturn(noContent()));
        stubFor(post(urlPathEqualTo("/v1/memories"))
                .willReturn(okJson("{\"memoryId\":\"" + M1 + "\"}")));

        String key = collection(wm).upsert(new Note(M1, "new text", "hr")).block();

        assertThat(key).isEqualTo(M1);
        verify(postRequestedFor(urlPathEqualTo("/v1/memories:batchGet")));
        verify(deleteRequestedFor(urlPathEqualTo("/v1/memories/" + M1)));
    }

    @Test
    void aFailedUpdateRestoresThePreviousVersion(WireMockRuntimeInfo wm) {
        stubSpace();
        stubFor(post(urlPathEqualTo("/v1/memories:batchGet")).willReturn(okJson(EXISTING)));
        stubFor(delete(urlPathEqualTo("/v1/memories/" + M1)).willReturn(noContent()));

        // The write fails, then the restore succeeds.
        stubFor(post(urlPathEqualTo("/v1/memories")).inScenario("write")
                .whenScenarioStateIs(com.github.tomakehurst.wiremock.stubbing.Scenario.STARTED)
                .willReturn(aResponse().withStatus(400)
                        .withBody("{\"errors\":[{\"message\":\"must be provided\"}]}"))
                .willSetStateTo("restoring"));
        stubFor(post(urlPathEqualTo("/v1/memories")).inScenario("write")
                .whenScenarioStateIs("restoring")
                .willReturn(okJson("{\"memoryId\":\"" + M1 + "\"}")));

        assertThatThrownBy(() -> collection(wm).upsert(new Note(M1, "", "hr")).block())
                .isInstanceOf(GoodMemUpsertException.class)
                .hasMessageContaining("was restored")
                .satisfies(error -> {
                    GoodMemUpsertException upsert = (GoodMemUpsertException) error;
                    assertThat(upsert.isRestored()).isTrue();
                    assertThat(upsert.getLostKey()).isNull();
                });

        // Two writes: the failed one and the restore.
        verify(2, postRequestedFor(urlPathEqualTo("/v1/memories")));
    }

    @Test
    void aFailedRestoreNamesTheLostRecord(WireMockRuntimeInfo wm) {
        stubSpace();
        stubFor(post(urlPathEqualTo("/v1/memories:batchGet")).willReturn(okJson(EXISTING)));
        stubFor(delete(urlPathEqualTo("/v1/memories/" + M1)).willReturn(noContent()));
        stubFor(post(urlPathEqualTo("/v1/memories"))
                .willReturn(aResponse().withStatus(500).withBody("down")));

        assertThatThrownBy(() -> collection(wm).upsert(new Note(M1, "new", "hr")).block())
                .isInstanceOf(GoodMemUpsertException.class)
                .hasMessageContaining("could NOT be restored")
                .satisfies(error ->
                        assertThat(((GoodMemUpsertException) error).getLostKey()).isEqualTo(M1));
    }

    @Test
    void aRecordThatDoesNotExistYetIsNeverDeleted(WireMockRuntimeInfo wm) {
        stubSpace();
        stubFor(post(urlPathEqualTo("/v1/memories:batchGet"))
                .willReturn(okJson("{\"results\":[]}")));
        stubFor(post(urlPathEqualTo("/v1/memories"))
                .willReturn(okJson("{\"memoryId\":\"" + M_NEW + "\"}")));

        String key = collection(wm).upsert(new Note(M_NEW, "hello", "ops")).block();

        assertThat(key).isEqualTo(M_NEW);
        verify(0, deleteRequestedFor(urlPathMatching("/v1/memories/.*")));
    }

    @Test
    void aRecordWithNoKeyIsCreatedWithoutAnyLookup(WireMockRuntimeInfo wm) {
        stubSpace();
        stubFor(post(urlPathEqualTo("/v1/memories"))
                .willReturn(okJson("{\"memoryId\":\"server-assigned\"}")));

        String key = collection(wm).upsert(new Note(null, "hello", "ops")).block();

        assertThat(key).isEqualTo("server-assigned");
        verify(0, postRequestedFor(urlPathEqualTo("/v1/memories:batchGet")));
    }
}
