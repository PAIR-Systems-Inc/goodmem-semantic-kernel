package ai.goodmem.semantickernel;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;

/**
 * A typed, reactive collection backed by a single GoodMem space.
 * <p>
 * Maps Semantic Kernel's vector-store collection concepts to GoodMem:
 * <ul>
 *   <li>Collection name → GoodMem space name</li>
 *   <li>Record key ({@link GoodMemKey}) → GoodMem {@code memoryId}</li>
 *   <li>Content field ({@link GoodMemData} named "content") → {@code originalContent}</li>
 *   <li>Remaining {@link GoodMemData} fields → GoodMem metadata</li>
 * </ul>
 *
 * <p>All public methods return Project Reactor types ({@link Mono}/{@link Flux})
 * and are safe to compose on any scheduler.
 *
 * <pre>{@code
 * var collection = GoodMemCollection.of("agent-memory", Note.class);
 * collection.ensureCollectionExists().block();
 * collection.upsert(new Note("Paris is in France", "geo")).block();
 * collection.search("european capitals", 3)
 *     .doOnNext(r -> System.out.println(r.score() + " " + r.record().content))
 *     .blockLast();
 * }</pre>
 *
 * @param <T> the record type; must have exactly one {@link GoodMemKey} field and at least one {@link GoodMemData} field
 */
public final class GoodMemCollection<T> implements AutoCloseable {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final TypeReference<Map<String, Object>> MAP_TYPE = new TypeReference<>() {};

    /**
     * JDK platform logging: with no configuration, java.util.logging prints warnings to
     * standard error; an application on SLF4J or Log4j routes it with their System.Logger bridge.
     */
    private static final System.Logger LOG = System.getLogger(GoodMemCollection.class.getName());

    private final String name;
    private final GoodMemClient client;
    private final GoodMemSchema<T> schema;
    private final GoodMemOptions options;
    private final boolean ownsClient;

    /** Lazily resolved space UUID, cached after first resolution. */
    private final AtomicReference<String> spaceId = new AtomicReference<>();

    // ── Construction ──────────────────────────────────────────────────────────

    private GoodMemCollection(String name, Class<T> recordClass,
                              GoodMemClient client, GoodMemOptions options, boolean ownsClient) {
        this.name = name;
        this.schema = GoodMemSchema.build(recordClass);
        this.client = client;
        this.options = options;
        this.ownsClient = ownsClient;
    }

    /**
     * Creates a collection that manages its own {@link GoodMemClient}.
     * Configuration is read from {@code GOODMEM_*} environment variables.
     */
    public static <T> GoodMemCollection<T> of(String collectionName, Class<T> recordClass) {
        return of(collectionName, recordClass, GoodMemOptions.builder().build());
    }

    /**
     * Creates a collection with explicit configuration.
     */
    public static <T> GoodMemCollection<T> of(String collectionName, Class<T> recordClass, GoodMemOptions options) {
        return new GoodMemCollection<>(collectionName, recordClass, new GoodMemClient(options), options, true);
    }

    /** Package-private constructor used by {@link GoodMemVectorStore} to share a client. */
    static <T> GoodMemCollection<T> withSharedClient(String name, Class<T> recordClass,
                                                      GoodMemClient client, GoodMemOptions options) {
        return new GoodMemCollection<>(name, recordClass, client, options, false);
    }

    public String getName() { return name; }

    // ── Collection lifecycle ──────────────────────────────────────────────────

    /**
     * Returns {@code true} if the GoodMem space for this collection already exists.
     */
    public Mono<Boolean> collectionExists() {
        return client.listSpaces(name)
                .map(spaces -> spaces.stream()
                        .anyMatch(s -> name.equals(s.path("name").asText(null))));
    }

    /**
     * Ensures the GoodMem space exists, creating it if necessary.
     * Idempotent — safe to call multiple times.
     */
    public Mono<Void> ensureCollectionExists() {
        return resolveSpaceId().then();
    }

    /**
     * Deletes the underlying GoodMem space if it exists.
     * Idempotent — does nothing if the space is not found.
     */
    public Mono<Void> ensureCollectionDeleted() {
        return client.listSpaces(name)
                .flatMap(spaces -> {
                    for (ObjectNode space : spaces) {
                        String sName = space.path("name").asText(null);
                        String sId = space.path("spaceId").asText(null);
                        if (name.equals(sName) && sId != null) {
                            spaceId.set(null);
                            return client.deleteSpace(sId);
                        }
                    }
                    return Mono.empty();
                });
    }

    // ── CRUD ─────────────────────────────────────────────────────────────────

    /**
     * Inserts or replaces a record.
     *
     * <p>GoodMem memories cannot be updated in place, so replacing one means deleting
     * it and creating it again. Deleting first and then failing to create would destroy
     * the record, so the current version is read before the delete and written back if
     * the create fails; a failure then raises {@link GoodMemUpsertException} saying
     * whether the restore succeeded.
     *
     * <p>A {@code null} key lets the server assign one. Any other key must be a UUID;
     * anything else is refused with {@link IllegalArgumentException} before a request
     * is made.
     *
     * @return the server-assigned memory ID
     */
    public Mono<String> upsert(T record) {
        return Mono.fromCallable(() -> serialize(record))
                .flatMap(ser -> upsertSerialized(record, ser));
    }

    /** A null key lets the server assign one; any other key must be a UUID. */
    private GoodMemSchema.SerializedRecord serialize(T record) {
        var ser = schema.serialize(record);
        if (ser.memoryId() == null) return ser;
        return new GoodMemSchema.SerializedRecord(
                ser.content(), ser.metadata(), GoodMemIds.requireUuid(ser.memoryId(), "key"));
    }

    private Mono<String> upsertSerialized(T record, GoodMemSchema.SerializedRecord ser) {
        return resolveSpaceId().flatMap(sid -> {
            if (ser.memoryId() == null) {
                return create(sid, ser, record);
            }

            return client.batchGetMemories(List.of(ser.memoryId()))
                    .flatMap(existing -> {
                        if (existing.isEmpty()) {
                            // Nothing to replace; a plain create, with no delete.
                            return create(sid, ser, record);
                        }
                        ObjectNode previous = existing.get(0);
                        return client.deleteMemory(ser.memoryId())
                                .then(create(sid, ser, record))
                                .onErrorResume(error -> restore(sid, ser.memoryId(), previous)
                                        .map(restored -> {
                                            throw new GoodMemUpsertException(
                                                    "Updating record " + ser.memoryId() + " failed: "
                                                            + error.getMessage() + ". "
                                                            + (restored
                                                                    ? "The previous version was restored."
                                                                    : "The previous version could NOT be"
                                                                            + " restored and is lost."),
                                                    restored,
                                                    restored ? null : ser.memoryId(),
                                                    error);
                                        }));
                    });
        });
    }

    private Mono<String> create(String spaceId, GoodMemSchema.SerializedRecord ser, T record) {
        return client.createMemory(spaceId, ser.content(), null, ser.metadata(), ser.memoryId())
                .map(result -> {
                    String returnedId = result.path("memoryId").asText(null);
                    if (returnedId != null) schema.setKey(record, returnedId);
                    return returnedId != null ? returnedId : "";
                });
    }

    /** Writes the previous version back. Emits {@code true} when that succeeded. */
    private Mono<Boolean> restore(String spaceId, String memoryId, ObjectNode previous) {
        String encoded = previous.path("originalContent").asText(null);
        String content = encoded == null
                ? ""
                : new String(Base64.getDecoder().decode(encoded), StandardCharsets.UTF_8);

        Map<String, Object> metadata = null;
        if (previous.path("metadata").isObject()) {
            metadata = MAPPER.convertValue(previous.get("metadata"), MAP_TYPE);
        }
        String contentType = previous.path("contentType").asText("text/plain");

        return client.createMemory(spaceId, content, contentType, metadata, memoryId)
                .map(ignored -> true)
                .onErrorReturn(false);
    }

    /**
     * Upserts multiple records sequentially.
     *
     * <p>Every key is checked before anything is sent, so one bad key refuses the
     * whole batch rather than leaving it half written.
     *
     * @return a {@link Flux} emitting each server-assigned memory ID
     */
    public Flux<String> upsertAll(Iterable<T> records) {
        return Mono.fromCallable(() -> {
                    List<Pending<T>> batch = new ArrayList<>();
                    for (T record : records) batch.add(new Pending<>(record, serialize(record)));
                    return batch;
                })
                .flatMapMany(Flux::fromIterable)
                .concatMap(p -> upsertSerialized(p.record(), p.serialized()));
    }

    private record Pending<R>(R record, GoodMemSchema.SerializedRecord serialized) {}

    /**
     * Retrieves a single record by its memory ID.
     *
     * @return the record, or an empty {@link Mono} if not found
     */
    public Mono<T> get(String key) {
        return Mono.fromCallable(() -> GoodMemIds.requireUuid(key, "key"))
                .flatMap(id -> client.batchGetMemories(List.of(id)))
                .flatMap(mems -> mems.isEmpty()
                        ? Mono.empty()
                        : Mono.just(schema.deserialize(mems.get(0))));
    }

    /**
     * Retrieves multiple records by their memory IDs.
     *
     * @return a {@link Flux} emitting only the records that were found
     */
    public Flux<T> getAll(List<String> keys) {
        if (keys.isEmpty()) return Flux.empty();
        // Ids travel in the batchGet body, not the path; checked for consistency.
        return Mono.fromCallable(() -> requireUuids(keys))
                .flatMap(client::batchGetMemories)
                .flatMapMany(Flux::fromIterable)
                .map(schema::deserialize);
    }

    /**
     * Deletes a record by its memory ID. 404 is silently ignored.
     *
     * <p>The key is a URL path segment, so it must be a UUID: anything else is refused
     * with {@link IllegalArgumentException} before a request is made.
     */
    public Mono<Void> delete(String key) {
        return Mono.fromCallable(() -> GoodMemIds.requireUuid(key, "key"))
                .flatMap(client::deleteMemory);
    }

    /**
     * Deletes multiple records. 404s are silently ignored.
     *
     * <p>Every key is checked before the first delete, so one bad key refuses the
     * whole batch rather than leaving it half deleted.
     */
    public Mono<Void> deleteAll(Iterable<String> keys) {
        return Mono.fromCallable(() -> requireUuids(keys))
                .flatMapMany(Flux::fromIterable)
                .concatMap(client::deleteMemory)
                .then();
    }

    private static List<String> requireUuids(Iterable<String> keys) {
        List<String> ids = new ArrayList<>();
        for (String key : keys) ids.add(GoodMemIds.requireUuid(key, "key"));
        return ids;
    }

    // ── Search ────────────────────────────────────────────────────────────────

    /**
     * Searches this collection semantically. GoodMem embeds the query server-side.
     *
     * <p>When the server reports a problem, such as a reranker that failed, the results it did
     * return are still emitted and a warning naming the statuses is logged, since a
     * {@link Flux} of results has nowhere else to put them. It never errors for a reported
     * problem. Use {@link #searchWithStatus} to get the statuses and the partial flag.
     *
     * @param query natural-language query text
     * @param top   maximum number of results to return
     * @return a {@link Flux} of {@link SearchResult} ordered by relevance (highest score first)
     */
    public Flux<SearchResult<T>> search(String query, int top) {
        return searchCore(query, top)
                .doOnNext(search -> {
                    // Contract Q4a/Q4b: the results come back whatever happened, and with no
                    // slot for a flag on a Flux, the log line is where a reported problem shows.
                    if (search.partial()) logPartial(search);
                })
                .flatMapMany(search -> Flux.fromIterable(search.results()));
    }

    /**
     * Searches like {@link #search}, and also returns every problem the GoodMem server
     * reported while it ran.
     *
     * <p>This follows the retrieval status contract every GoodMem integration follows.
     * {@code FEATURE_DISABLED} and {@code LLM_CAPABILITY_INFERRED} are informational and are not
     * reported. Any other status sets {@link SearchResults#partial()} and is listed in
     * {@link SearchResults#statuses()}; a code this connector does not recognise is listed as
     * {@code UNKNOWN}, with the server's code in {@link GoodMemRetrievalStatus#originalCode()}.
     * The results the server returned are always kept. A reported problem with no results
     * emits an empty, partial result and logs a warning; it never errors.
     *
     * @param query natural-language query text
     * @param top   maximum number of results to return
     * @return the results, best first, with the statuses the server reported
     */
    public Mono<SearchResults<T>> searchWithStatus(String query, int top) {
        return searchCore(query, top)
                .doOnNext(search -> {
                    // Contract Q4b: empty plus a flag, never an error, and a warning as well.
                    if (search.partial() && search.results().isEmpty()) logPartial(search);
                });
    }

    private Mono<SearchResults<T>> searchCore(String query, int top) {
        return resolveSpaceId()
                .flatMap(sid -> client.retrieveMemories(query, List.of(sid), top))
                .map(response -> new SearchResults<>(
                        response.results().stream()
                                .map(r -> new SearchResult<>(
                                        schema.deserializeFromRetrieve(r.chunk(), r.memory()),
                                        r.score()))
                                .toList(),
                        response.statuses()));
    }

    private void logPartial(SearchResults<T> search) {
        String statuses = RetrievalStatuses.describe(search.statuses());
        LOG.log(System.Logger.Level.WARNING, search.results().isEmpty()
                ? "GoodMem search of " + name + " reported a problem and returned no results: " + statuses
                : "GoodMem search of " + name + " reported a problem; the " + search.results().size()
                        + " results it returned may be incomplete: " + statuses);
    }

    /** A single search result pairing a deserialized record with its relevance score. */
    public record SearchResult<R>(R record, double score) {}

    /**
     * The results of {@link #searchWithStatus}, with any problem the GoodMem server reported.
     *
     * @param results  every result the server returned, best first; a reported problem never
     *                 removes results
     * @param statuses the problems the server reported; informational notices
     *                 ({@code FEATURE_DISABLED}, {@code LLM_CAPABILITY_INFERRED}) are not included
     */
    public record SearchResults<R>(List<SearchResult<R>> results, List<GoodMemRetrievalStatus> statuses) {

        public SearchResults {
            results = List.copyOf(results);
            statuses = List.copyOf(statuses);
        }

        /**
         * {@code true} when the server reported a real problem during this search, so
         * {@link #results()} may be incomplete, or empty when they should not be. It does not
         * depend on whether any results came back.
         */
        public boolean partial() {
            return !statuses.isEmpty();
        }
    }

    // ── AutoCloseable ─────────────────────────────────────────────────────────

    @Override
    public void close() {
        if (ownsClient) client.close();
    }

    // ── Internal helpers ─────────────────────────────────────────────────────

    private Mono<String> resolveSpaceId() {
        return Mono.defer(() -> {
            String cached = spaceId.get();
            if (cached != null) return Mono.just(cached);

            // Checked before the first request, so a bad setting sends nothing.
            String configured = options.getEmbedderId();
            String embedderId = configured != null && !configured.isBlank()
                    ? GoodMemIds.requireUuid(configured, "embedderId")
                    : null;
            return resolveSpaceId(embedderId);
        });
    }

    private Mono<String> resolveSpaceId(String configuredEmbedderId) {
        return client.listSpaces(name).flatMap(spaces -> {
            for (ObjectNode space : spaces) {
                if (name.equals(space.path("name").asText(null))) {
                    String sid = space.path("spaceId").asText(null);
                    if (sid != null) {
                        spaceId.set(sid);
                        return Mono.just(sid);
                    }
                }
            }
            // Space doesn't exist — create it.
            return resolveEmbedderId(configuredEmbedderId).flatMap(eid ->
                    client.createSpace(name, eid, null)
            ).map(created -> {
                String sid = created.path("spaceId").asText(null);
                if (sid == null)
                    throw new GoodMemException("GoodMem did not return a spaceId after creating space '" + name + "'.");
                spaceId.set(sid);
                return sid;
            });
        });
    }

    private Mono<String> resolveEmbedderId(String configured) {
        if (configured != null) return Mono.just(configured);

        return client.listEmbedders().map(embedders -> {
            if (!embedders.isEmpty()) {
                String eid = embedders.get(0).path("embedderId").asText(null);
                if (eid != null) return eid;
            }
            throw new GoodMemException(
                    "No embedders configured in GoodMem. Create one via the API or set GOODMEM_EMBEDDER_ID.");
        });
    }
}
