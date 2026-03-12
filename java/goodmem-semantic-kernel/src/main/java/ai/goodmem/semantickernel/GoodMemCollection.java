package ai.goodmem.semantickernel;

import com.fasterxml.jackson.databind.node.ObjectNode;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.util.List;
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
     * Inserts or replaces a record. If the record has a non-null key, the existing
     * memory with that ID is deleted first (GoodMem is append-only; upsert is
     * implemented as delete-then-insert).
     *
     * @return the server-assigned memory ID
     */
    public Mono<String> upsert(T record) {
        return resolveSpaceId().flatMap(sid -> {
            var ser = schema.serialize(record);

            Mono<Void> deletePrior = ser.memoryId() != null
                    ? client.deleteMemory(ser.memoryId())
                    : Mono.empty();

            return deletePrior.then(
                    client.createMemory(sid, ser.content(), null, ser.metadata(), ser.memoryId())
            ).map(result -> {
                String returnedId = result.path("memoryId").asText(null);
                if (returnedId != null) schema.setKey(record, returnedId);
                return returnedId != null ? returnedId : "";
            });
        });
    }

    /**
     * Upserts multiple records sequentially.
     *
     * @return a {@link Flux} emitting each server-assigned memory ID
     */
    public Flux<String> upsertAll(Iterable<T> records) {
        return Flux.fromIterable(records).concatMap(this::upsert);
    }

    /**
     * Retrieves a single record by its memory ID.
     *
     * @return the record, or an empty {@link Mono} if not found
     */
    public Mono<T> get(String key) {
        return client.batchGetMemories(List.of(key))
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
        return client.batchGetMemories(keys)
                .flatMapMany(Flux::fromIterable)
                .map(schema::deserialize);
    }

    /**
     * Deletes a record by its memory ID. 404 is silently ignored.
     */
    public Mono<Void> delete(String key) {
        return client.deleteMemory(key);
    }

    /**
     * Deletes multiple records. 404s are silently ignored.
     */
    public Mono<Void> deleteAll(Iterable<String> keys) {
        return Flux.fromIterable(keys).concatMap(client::deleteMemory).then();
    }

    // ── Search ────────────────────────────────────────────────────────────────

    /**
     * Searches this collection semantically. GoodMem embeds the query server-side.
     *
     * @param query natural-language query text
     * @param top   maximum number of results to return
     * @return a {@link Flux} of {@link SearchResult} ordered by relevance (highest score first)
     */
    public Flux<SearchResult<T>> search(String query, int top) {
        return resolveSpaceId().flatMapMany(sid ->
                client.retrieveMemories(query, List.of(sid), top)
                        .flatMapMany(Flux::fromIterable)
                        .map(r -> new SearchResult<>(
                                schema.deserializeFromRetrieve(r.chunk(), r.memory()),
                                r.score()))
        );
    }

    /** A single search result pairing a deserialized record with its relevance score. */
    public record SearchResult<R>(R record, double score) {}

    // ── AutoCloseable ─────────────────────────────────────────────────────────

    @Override
    public void close() {
        if (ownsClient) client.close();
    }

    // ── Internal helpers ─────────────────────────────────────────────────────

    private Mono<String> resolveSpaceId() {
        String cached = spaceId.get();
        if (cached != null) return Mono.just(cached);

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
            return resolveEmbedderId().flatMap(eid ->
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

    private Mono<String> resolveEmbedderId() {
        String configured = options.getEmbedderId();
        if (configured != null && !configured.isBlank()) return Mono.just(configured);

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
