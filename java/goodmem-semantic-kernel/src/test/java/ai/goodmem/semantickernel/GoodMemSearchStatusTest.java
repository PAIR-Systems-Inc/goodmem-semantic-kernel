package ai.goodmem.semantickernel;

import com.github.tomakehurst.wiremock.junit5.WireMockRuntimeInfo;
import com.github.tomakehurst.wiremock.junit5.WireMockTest;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.logging.Level;
import java.util.logging.LogRecord;

import static ai.goodmem.semantickernel.RetrievalFixtures.*;
import static com.github.tomakehurst.wiremock.client.WireMock.*;
import static org.assertj.core.api.Assertions.*;

/**
 * The retrieval status contract, through {@link GoodMemCollection#searchWithStatus}.
 *
 * <p>Q1: {@code FEATURE_DISABLED} and {@code LLM_CAPABILITY_INFERRED} are noise, by code
 * alone. Q3: an unrecognised code is surfaced as {@code UNKNOWN} and marks the result partial.
 * Q4a: a problem with hits returns the hits, partial, with the statuses. Q4b: a problem with no
 * hits returns empty and partial, logs a warning, and does not error.
 *
 * <p>These drive the connector's real HTTP client against WireMock, with retrieval streams
 * captured from a live server (see {@link RetrievalFixtures}).
 */
@WireMockTest
class GoodMemSearchStatusTest {

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

    private static GoodMemOptions options(WireMockRuntimeInfo wm) {
        return GoodMemOptions.builder()
                .baseUrl("http://localhost:" + wm.getHttpPort())
                .apiKey("test-key")
                .build();
    }

    private static void serve(String stream) {
        stubFor(get(urlPathEqualTo("/v1/spaces"))
                .willReturn(okJson("{\"spaces\":[{\"spaceId\":\"" + SPACE_ID + "\",\"name\":\"notes\"}]}")));
        stubFor(post(urlPathEqualTo("/v1/memories:retrieve"))
                .willReturn(ok(stream).withHeader("Content-Type", "application/x-ndjson")));
    }

    private static GoodMemCollection.SearchResults<Note> search(WireMockRuntimeInfo wm, String stream) {
        serve(stream);
        try (var collection = GoodMemCollection.of("notes", Note.class, options(wm))) {
            return collection.searchWithStatus("fixture canary", 5).block();
        }
    }

    @Test
    void healthySearch_isNotPartial(WireMockRuntimeInfo wm) {
        var search = search(wm, load("retrieve_ok"));

        assertThat(search.results()).extracting(r -> r.record().content).containsExactly(CANARY_TEXT);
        assertThat(search.partial()).isFalse();
        assertThat(search.statuses()).isEmpty();
        assertThat(logs.records()).isEmpty();
    }

    @Test
    void q1_informationalCodes_areNoise_byCodeAlone(WireMockRuntimeInfo wm) {
        // Details that name a reranker must not turn a notice into a problem.
        var search = search(wm, okWith(
                statusLine("FEATURE_DISABLED", "no LLM configured", "{\"reranker_id\":\"" + MISSING_RERANKER + "\"}"),
                statusLine("LLM_CAPABILITY_INFERRED", "capabilities inferred", null)));

        assertThat(search.results()).hasSize(1);
        assertThat(search.partial()).isFalse();
        assertThat(search.statuses()).isEmpty();
        assertThat(logs.records()).isEmpty();
    }

    @Test
    void q3_unknownCode_isSurfacedAsUnknown_withTheServersCode_andKeepsTheHits(WireMockRuntimeInfo wm) {
        var search = search(wm, okWith(
                statusLine("SOMETHING_NEW_IN_A_LATER_SERVER", "unrecognised", "{\"stage\":\"retrieve\"}")));

        assertThat(search.results()).hasSize(1);
        assertThat(search.partial()).isTrue();
        var status = search.statuses().get(0);
        assertThat(search.statuses()).hasSize(1);
        assertThat(status.code()).isEqualTo("UNKNOWN");
        assertThat(status.originalCode()).isEqualTo("SOMETHING_NEW_IN_A_LATER_SERVER");
        assertThat(status.unrecognized()).isTrue();
        assertThat(status.message()).isEqualTo("unrecognised");
        assertThat(status.details()).containsEntry("stage", "retrieve");
    }

    @Test
    void q3_statusWithNoCode_isUnknown(WireMockRuntimeInfo wm) {
        var search = search(wm, okWith(statusLine(null, "a status with no code", null)));

        assertThat(search.results()).hasSize(1);
        assertThat(search.partial()).isTrue();
        assertThat(search.statuses()).singleElement().satisfies(s -> {
            assertThat(s.code()).isEqualTo("UNKNOWN");
            assertThat(s.originalCode()).isNull();
            assertThat(s.unrecognized()).isTrue();
        });
    }

    @Test
    void q4a_capturedDegradedStream_returnsTheHit_partial_withTheStatuses(WireMockRuntimeInfo wm) {
        var search = search(wm, load("retrieve_degraded_hits"));

        assertThat(search.results()).hasSize(1);
        assertThat(search.results().get(0).record().content).isEqualTo(CANARY_TEXT);
        // The reranker failed, so this is the vector score, negated into higher-is-better.
        assertThat(search.results().get(0).score()).isCloseTo(-DEGRADED_RAW_SCORE, within(1e-9));
        assertThat(search.partial()).isTrue();
        assertThat(search.statuses()).extracting(GoodMemRetrievalStatus::code)
                .containsExactly("NOT_FOUND", "RERANKING_FAILED");
        assertThat(search.statuses()).allSatisfy(s -> {
            assertThat(s.details()).containsEntry("reranker_id", MISSING_RERANKER);
            assertThat(s.unrecognized()).isFalse();
        });
        // The flag is in the result, so there is nothing to log.
        assertThat(logs.records()).isEmpty();
    }

    @Test
    void q4b_capturedDegradedEmptyStream_returnsEmpty_partial_logsAWarning_andDoesNotError(WireMockRuntimeInfo wm) {
        var search = search(wm, load("retrieve_degraded_empty"));

        assertThat(search.results()).isEmpty();
        assertThat(search.partial()).isTrue();
        assertThat(search.statuses()).extracting(GoodMemRetrievalStatus::code)
                .containsExactly("NOT_FOUND", "RERANKING_FAILED");
        assertThat(logs.records()).singleElement().satisfies(r -> {
            assertThat(r.getLevel()).isEqualTo(Level.WARNING);
            assertThat(r.getLoggerName()).isEqualTo("ai.goodmem.semantickernel.GoodMemCollection");
            assertThat(r.getMessage()).contains(
                    "returned no results", "NOT_FOUND", "RERANKING_FAILED", MISSING_RERANKER);
        });
    }

    @Test
    void healthyEmptySearch_isNotPartial_andLogsNothing(WireMockRuntimeInfo wm) {
        var search = search(wm, "");

        assertThat(search.results()).isEmpty();
        assertThat(search.partial()).isFalse();
        assertThat(logs.records()).isEmpty();
    }

    @Test
    void truncatedLastLine_keepsTheHitsBeforeIt_andIsPartial(WireMockRuntimeInfo wm) {
        // Four whole lines, then a fifth cut off mid-object.
        var search = search(wm, load("retrieve_ok").stripTrailing() + "\n{\"retrievedItem\":{\"chunk\":{\"chu");

        assertThat(search.results()).extracting(r -> r.record().content).containsExactly(CANARY_TEXT);
        assertThat(search.partial()).isTrue();
        assertThat(search.statuses()).singleElement().satisfies(s -> {
            assertThat(s.code()).isEqualTo("MALFORMED_STREAM");
            assertThat(s.message()).startsWith("Line 5 of the retrieval stream could not be parsed");
            assertThat(s.unrecognized()).isFalse();
        });
    }

    @Test
    void lineThatIsNotAnObject_isReported_andTheLinesAfterItAreStillRead(WireMockRuntimeInfo wm) {
        var search = search(wm, "[1,2]\nnull\n" + load("retrieve_ok"));

        assertThat(search.results()).hasSize(1);
        assertThat(search.statuses()).extracting(GoodMemRetrievalStatus::code)
                .containsExactly("MALFORMED_STREAM", "MALFORMED_STREAM");
    }

    @Test
    void search_logsAWarning_evenWhenTheProblemCameWithHits(WireMockRuntimeInfo wm) {
        // search() has no slot for the flag, so the log is the only place it can show.
        serve(load("retrieve_degraded_hits"));
        try (var collection = GoodMemCollection.of("notes", Note.class, options(wm))) {
            var hits = collection.search("fixture canary", 5).collectList().block();

            assertThat(hits).hasSize(1);
        }
        assertThat(logs.records()).singleElement().satisfies(r -> {
            assertThat(r.getLevel()).isEqualTo(Level.WARNING);
            assertThat(r.getMessage()).contains("the 1 results it returned may be incomplete", "RERANKING_FAILED");
        });
    }

    @Test
    void loggedStatuses_stayOnOneLine(WireMockRuntimeInfo wm) {
        search(wm, statusLine("VECTOR_SEARCH_FAILED", "first\\nWARNING: forged line", null));

        assertThat(logs.records()).singleElement().extracting(LogRecord::getMessage).asString()
                .doesNotContain("\n")
                .contains("VECTOR_SEARCH_FAILED: first WARNING: forged line");
    }

    @Test
    void vectorStoreCollections_reportStatusesToo(WireMockRuntimeInfo wm) {
        serve(load("retrieve_degraded_empty"));
        try (var store = new GoodMemVectorStore(options(wm))) {
            var search = store.getCollection("notes", Note.class).searchWithStatus("fixture canary", 5).block();

            assertThat(search.partial()).isTrue();
            assertThat(search.statuses()).extracting(GoodMemRetrievalStatus::code)
                    .containsExactly("NOT_FOUND", "RERANKING_FAILED");
        }
    }
}
