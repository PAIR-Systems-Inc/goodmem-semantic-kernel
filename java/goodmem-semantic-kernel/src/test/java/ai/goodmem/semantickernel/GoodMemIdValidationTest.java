package ai.goodmem.semantickernel;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.io.IOException;
import java.util.List;
import java.util.Locale;
import java.util.stream.Stream;

import static org.assertj.core.api.Assertions.*;

/**
 * Ids that can reach a URL path must be UUIDs, checked before any request.
 *
 * <p>Nothing here is mocked: each test drives the public connector API through
 * its own {@link java.net.http.HttpClient} over TCP to {@link RecordingHttpServer},
 * which records every request line it receives.
 */
class GoodMemIdValidationTest {

    static final String U = "0198c3a2-7f4e-7c1a-9b2d-5e6f7a8b9c0d";
    static final String EMBEDDER_ID = "019cfd1c-c033-7517-b7de-f73941a0464b";
    static final String REFUSAL = "must be a GoodMem UUID";

    private static final ObjectMapper MAPPER = new ObjectMapper();

    static Stream<String> payloads() {
        return Stream.of(
                "../spaces/" + U,
                "a/../../spaces/" + U,
                "%2e%2e/spaces/" + U,
                "..%2Fspaces%2F" + U,
                U + "/../../spaces/" + U,
                "",
                " " + U,
                U + "?x=1",
                U + "#frag",
                U + "\n",
                "..");
    }

    /**
     * A blank setting means "not configured", as it does for any GOODMEM_* env var;
     * it is never sent anywhere. Every other payload is refused.
     */
    static Stream<String> settingPayloads() {
        return payloads().filter(p -> !p.isEmpty());
    }

    public static class Note {
        @GoodMemKey
        public String id;

        @GoodMemData
        public String content;

        public Note() {}

        Note(String id, String content) {
            this.id = id;
            this.content = content;
        }
    }

    private RecordingHttpServer server;

    @BeforeEach
    void start() throws IOException {
        server = new RecordingHttpServer();
    }

    @AfterEach
    void stop() {
        server.close();
    }

    private GoodMemOptions options(String embedderId) {
        return GoodMemOptions.builder()
                .baseUrl(server.url())
                .apiKey("test-key")
                .embedderId(embedderId)
                .build();
    }

    private GoodMemCollection<Note> collection() {
        return GoodMemCollection.of("notes", Note.class, options(EMBEDDER_ID));
    }

    private static Throwable outcome(Runnable call) {
        try {
            call.run();
            return null;
        } catch (Throwable error) {
            return error;
        }
    }

    private void assertRefused(Throwable error) {
        // Requests first: against the unfixed code this is the assertion that
        // fails, and its message shows the path the server actually received.
        assertThat(server.requests()).as("the server received %s", server.requests()).isEmpty();
        assertThat(error).isInstanceOf(IllegalArgumentException.class).hasMessageContaining(REFUSAL);
    }

    // ── delete: the key is the path segment of DELETE /v1/memories/{key} ──

    @ParameterizedTest
    @MethodSource("payloads")
    void deleteRefusesANonUuidKey(String payload) {
        assertRefused(outcome(() -> collection().delete(payload).block()));
    }

    @ParameterizedTest
    @MethodSource("payloads")
    void deleteAllRefusesTheWholeBatchBeforeDeletingAny(String payload) {
        assertRefused(outcome(() -> collection().deleteAll(List.of(U, payload)).block()));
    }

    @Test
    void deleteWithAUuidReachesExactlyThatMemory() {
        collection().delete(U).block();

        assertThat(server.requests()).map(Object::toString).containsExactly("DELETE /v1/memories/" + U);
    }

    @Test
    void anUppercaseUuidIsNormalised() {
        collection().delete(U.toUpperCase(Locale.ROOT)).block();

        assertThat(server.requests()).map(Object::toString).containsExactly("DELETE /v1/memories/" + U);
    }

    // ── upsert: the key reaches DELETE /v1/memories/{key} on replace ──

    @ParameterizedTest
    @MethodSource("payloads")
    void upsertRefusesANonUuidKey(String payload) {
        assertRefused(outcome(() -> collection().upsert(new Note(payload, "x")).block()));
    }

    @ParameterizedTest
    @MethodSource("payloads")
    void upsertAllRefusesTheWholeBatchBeforeWritingAny(String payload) {
        var batch = List.of(new Note(U, "first"), new Note(payload, "second"));
        assertRefused(outcome(() -> collection().upsertAll(batch).collectList().block()));
    }

    @Test
    void upsertWithAUuidKeySendsExactlyThatId() throws IOException {
        String key = collection().upsert(new Note(U, "x")).block();

        assertThat(key).isEqualTo(U);
        var requests = server.requests().stream()
                .filter(r -> !r.target().startsWith("/v1/spaces")).toList();
        assertThat(requests).map(Object::toString)
                .containsExactly("POST /v1/memories:batchGet", "POST /v1/memories");
        assertThat(MAPPER.readTree(requests.get(0).body()).path("memoryIds").get(0).asText()).isEqualTo(U);
        assertThat(MAPPER.readTree(requests.get(1).body()).path("memoryId").asText()).isEqualTo(U);
    }

    @Test
    void upsertWithoutAKeyLetsTheServerAssignOne() throws IOException {
        var note = new Note(null, "x");

        collection().upsert(note).block();

        assertThat(note.id).isEqualTo(RecordingHttpServer.ASSIGNED_MEMORY_ID);
        var last = server.requests().get(server.requests().size() - 1);
        assertThat(MAPPER.readTree(last.body()).has("memoryId")).isFalse();
    }

    // ── get: ids travel in the batchGet body, and are checked the same way ──

    @ParameterizedTest
    @MethodSource("payloads")
    void getRefusesANonUuidKey(String payload) {
        assertRefused(outcome(() -> collection().get(payload).block()));
    }

    @ParameterizedTest
    @MethodSource("payloads")
    void getAllRefusesANonUuidKey(String payload) {
        assertRefused(outcome(() -> collection().getAll(List.of(U, payload)).collectList().block()));
    }

    @Test
    void getWithAUuidAsksForExactlyThatMemory() throws IOException {
        collection().get(U.toUpperCase(Locale.ROOT)).block();

        var request = server.requests().get(0);
        assertThat(server.requests()).hasSize(1);
        assertThat(request.toString()).isEqualTo("POST /v1/memories:batchGet");
        assertThat(MAPPER.readTree(request.body()).path("memoryIds").get(0).asText()).isEqualTo(U);
    }

    // ── Ids from configuration ──

    @ParameterizedTest
    @MethodSource("settingPayloads")
    void aNonUuidEmbedderSettingIsRefused(String payload) {
        var collection = GoodMemCollection.of("notes", Note.class, options(payload));

        Throwable error = outcome(() -> collection.ensureCollectionExists().block());

        assertRefused(error);
        assertThat(error).hasMessageStartingWith("embedderId");
    }

    @Test
    void aUuidEmbedderSettingIsUsed() throws IOException {
        var collection = GoodMemCollection.of("other", Note.class, options(EMBEDDER_ID));

        collection.ensureCollectionExists().block();

        var create = server.requests().stream()
                .filter(r -> r.toString().equals("POST /v1/spaces")).findFirst().orElseThrow();
        assertThat(MAPPER.readTree(create.body()).path("spaceEmbedders").get(0).path("embedderId").asText())
                .isEqualTo(EMBEDDER_ID);
    }

    // ── Ids from the server: the space id in DELETE /v1/spaces/{id} ──

    @ParameterizedTest
    @MethodSource("settingPayloads")
    void aNonUuidSpaceIdFromTheServerIsNeverDeleted(String payload) {
        server.listedSpaceId(payload);

        Throwable error = outcome(() -> collection().ensureCollectionDeleted().block());

        assertThat(server.requests()).noneMatch(r -> r.method().equals("DELETE"));
        assertThat(error).isInstanceOf(IllegalArgumentException.class).hasMessageContaining(REFUSAL);
    }

    @ParameterizedTest
    @MethodSource("settingPayloads")
    void aNonUuidSpaceIdFromTheServerIsNeverDeletedByTheStore(String payload) {
        server.listedSpaceId(payload);

        try (var store = new GoodMemVectorStore(options(EMBEDDER_ID))) {
            Throwable error = outcome(() -> store.ensureCollectionDeleted("notes").block());

            assertThat(server.requests()).noneMatch(r -> r.method().equals("DELETE"));
            assertThat(error).isInstanceOf(IllegalArgumentException.class).hasMessageContaining(REFUSAL);
        }
    }

    @Test
    void ensureCollectionDeletedDeletesExactlyTheListedSpace() {
        collection().ensureCollectionDeleted().block();

        var requests = server.requests();
        assertThat(requests.get(requests.size() - 1).toString())
                .isEqualTo("DELETE /v1/spaces/" + RecordingHttpServer.DEFAULT_SPACE_ID);
    }
}
