package ai.goodmem.semantickernel;

import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

/**
 * Factory for {@link GoodMemCollection} instances.
 * <p>
 * A single {@link GoodMemVectorStore} owns one shared HTTP client that is reused
 * across every collection it creates, reducing connection overhead when managing
 * multiple GoodMem spaces.
 *
 * <pre>{@code
 * try (var store = new GoodMemVectorStore()) {
 *     var notes = store.getCollection("notes", Note.class);
 *     var todos = store.getCollection("todos", Note.class);
 *
 *     notes.ensureCollectionExists().block();
 *     todos.ensureCollectionExists().block();
 *
 *     // list all spaces
 *     store.listCollectionNames().doOnNext(System.out::println).blockLast();
 * }
 * }</pre>
 */
public final class GoodMemVectorStore implements AutoCloseable {

    private final GoodMemClient client;
    private final GoodMemOptions options;

    /**
     * Creates a store that reads configuration from {@code GOODMEM_*} environment variables.
     */
    public GoodMemVectorStore() {
        this(GoodMemOptions.builder().build());
    }

    /**
     * Creates a store with explicit configuration.
     */
    public GoodMemVectorStore(GoodMemOptions options) {
        this.options = options;
        this.client = new GoodMemClient(options);
    }

    // ── Collection factory ────────────────────────────────────────────────────

    /**
     * Returns a {@link GoodMemCollection} for the given collection (space) name.
     * The returned collection shares this store's HTTP client.
     *
     * @param name        the GoodMem space name
     * @param recordClass the record type; must have {@link GoodMemKey} and {@link GoodMemData} fields
     */
    public <T> GoodMemCollection<T> getCollection(String name, Class<T> recordClass) {
        return GoodMemCollection.withSharedClient(name, recordClass, client, options);
    }

    // ── Store-level operations ────────────────────────────────────────────────

    /**
     * Lists the names of all GoodMem spaces visible to the configured API key.
     */
    public Flux<String> listCollectionNames() {
        return client.listSpaces(null)
                .flatMapMany(spaces -> Flux.fromIterable(spaces)
                        .map(s -> s.path("name").asText(""))
                        .filter(n -> !n.isBlank()));
    }

    /**
     * Returns {@code true} if a space with the given name exists.
     */
    public Mono<Boolean> collectionExists(String name) {
        return client.listSpaces(name)
                .map(spaces -> spaces.stream()
                        .anyMatch(s -> name.equals(s.path("name").asText(null))));
    }

    /**
     * Deletes the space with the given name if it exists.
     * Idempotent — does nothing if the space is not found.
     */
    public Mono<Void> ensureCollectionDeleted(String name) {
        return client.listSpaces(name)
                .flatMap(spaces -> {
                    for (var space : spaces) {
                        if (name.equals(space.path("name").asText(null))) {
                            String sid = space.path("spaceId").asText(null);
                            if (sid != null) return client.deleteSpace(sid);
                        }
                    }
                    return Mono.empty();
                });
    }

    // ── AutoCloseable ─────────────────────────────────────────────────────────

    @Override
    public void close() {
        client.close();
    }
}
