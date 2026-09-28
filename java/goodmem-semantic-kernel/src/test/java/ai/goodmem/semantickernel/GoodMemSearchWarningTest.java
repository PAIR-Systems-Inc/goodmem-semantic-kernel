package ai.goodmem.semantickernel;

import com.github.tomakehurst.wiremock.junit5.WireMockRuntimeInfo;
import com.github.tomakehurst.wiremock.junit5.WireMockTest;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.logging.Level;
import java.util.logging.LogRecord;

import static ai.goodmem.semantickernel.RetrievalFixtures.*;
import static com.github.tomakehurst.wiremock.client.WireMock.*;
import static org.assertj.core.api.Assertions.*;

/**
 * {@link GoodMemCollection#search} returns a {@code Flux} of results, which has no slot for a
 * flag, so a problem the server reported has to show in the log. Before the fix every one of
 * these searches looked healthy: the status events were dropped and nothing was logged.
 *
 * <p>These drive the connector's real HTTP client against WireMock, with retrieval streams
 * captured from a live server (see {@link RetrievalFixtures}).
 */
@WireMockTest
class GoodMemSearchWarningTest {

    private static final String SPACE_ID = "01a0d44b-746f-775b-b91e-bc73d4058e27";

    public static class Note {
        @GoodMemKey public String id;
        @GoodMemData public String content;
        @GoodMemData("tenant") public String tenant;
    }

    private LogCapture logs;

    @BeforeEach
    void captureLogs() {
        logs = new LogCapture(GoodMemCollection.class.getName());
    }

    @AfterEach
    void releaseLogs() {
        logs.close();
    }

    private GoodMemCollection<Note> collection(WireMockRuntimeInfo wm, String stream) {
        stubFor(get(urlPathEqualTo("/v1/spaces"))
                .willReturn(okJson("{\"spaces\":[{\"spaceId\":\"" + SPACE_ID + "\",\"name\":\"notes\"}]}")));
        stubFor(post(urlPathEqualTo("/v1/memories:retrieve"))
                .willReturn(ok(stream).withHeader("Content-Type", "application/x-ndjson")));
        return GoodMemCollection.of("notes", Note.class, GoodMemOptions.builder()
                .baseUrl("http://localhost:" + wm.getHttpPort())
                .apiKey("test-key")
                .build());
    }

    private List<String> warnings() {
        return logs.records().stream()
                .filter(r -> r.getLevel() == Level.WARNING)
                .map(LogRecord::getMessage)
                .toList();
    }

    @Test
    void problemWithHits_emitsTheHits_andLogsAWarning(WireMockRuntimeInfo wm) {
        // Contract Q4a, on the captured stream of a search whose reranker does not exist.
        var hits = collection(wm, load("retrieve_degraded_hits")).search("fixture canary", 5).collectList().block();

        assertThat(hits).hasSize(1);
        assertThat(hits.get(0).record().content).isEqualTo(CANARY_TEXT);
        assertThat(hits.get(0).score()).isCloseTo(-DEGRADED_RAW_SCORE, within(1e-9));
        assertThat(warnings()).singleElement().asString()
                .contains("may be incomplete", "NOT_FOUND", "RERANKING_FAILED")
                .doesNotContain("FEATURE_DISABLED");
    }

    @Test
    void problemWithNoHits_emitsNothing_logsAWarning_andDoesNotError(WireMockRuntimeInfo wm) {
        // Contract Q4b: an empty result must not look like a search that found nothing.
        var hits = collection(wm, load("retrieve_degraded_empty")).search("fixture canary", 5).collectList().block();

        assertThat(hits).isEmpty();
        assertThat(warnings()).singleElement().asString()
                .contains("returned no results", "RERANKING_FAILED");
    }

    @Test
    void unknownCode_keepsTheHits_andLogsTheServersCode(WireMockRuntimeInfo wm) {
        // Contract Q3: never crash, never drop.
        var hits = collection(wm, okWith(statusLine("SOMETHING_NEW_IN_A_LATER_SERVER", "unrecognised", null)))
                .search("fixture canary", 5).collectList().block();

        assertThat(hits).hasSize(1);
        assertThat(warnings()).singleElement().asString()
                .contains("UNKNOWN (server code SOMETHING_NEW_IN_A_LATER_SERVER): unrecognised");
    }

    @Test
    void truncatedLastLine_keepsTheHits_andLogsAWarning(WireMockRuntimeInfo wm) {
        var hits = collection(wm, load("retrieve_ok") + "\n{\"retrievedItem\":{\"chunk\":{\"chu")
                .search("fixture canary", 5).collectList().block();

        assertThat(hits).extracting(h -> h.record().content).containsExactly(CANARY_TEXT);
        assertThat(warnings()).singleElement().asString().contains("MALFORMED_STREAM");
    }

    @Test
    void informationalNotices_logNothing(WireMockRuntimeInfo wm) {
        // Contract Q1: by code alone, whatever the details say.
        var hits = collection(wm, okWith(
                statusLine("FEATURE_DISABLED", "no LLM configured", "{\"feature\":\"summarization\"}"),
                statusLine("LLM_CAPABILITY_INFERRED", "capabilities inferred", "{\"reranker_id\":\"x\"}")))
                .search("fixture canary", 5).collectList().block();

        assertThat(hits).hasSize(1);
        assertThat(logs.records()).isEmpty();
    }

    @Test
    void healthySearch_logsNothing(WireMockRuntimeInfo wm) {
        var hits = collection(wm, load("retrieve_ok")).search("fixture canary", 5).collectList().block();

        assertThat(hits).hasSize(1);
        assertThat(logs.records()).isEmpty();
    }

    @Test
    void pluginRecall_logsAWarning_whenTheSearchReportedAProblemAndFoundNothing(WireMockRuntimeInfo wm) {
        var plugin = new GoodMemPlugin<>(collection(wm, load("retrieve_degraded_empty")), Note.class,
                (content, proto) -> new Note());

        String recalled = plugin.recall("fixture canary", 3);

        assertThat(recalled).isEqualTo("(no relevant memories found)");
        assertThat(warnings()).singleElement().asString().contains("returned no results", "RERANKING_FAILED");
    }
}
