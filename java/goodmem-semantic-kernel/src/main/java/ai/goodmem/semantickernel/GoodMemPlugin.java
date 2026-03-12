package ai.goodmem.semantickernel;

import com.microsoft.semantickernel.semanticfunctions.annotations.DefineKernelFunction;
import com.microsoft.semantickernel.semanticfunctions.annotations.KernelFunctionParameter;

import java.util.stream.Collectors;

/**
 * Semantic Kernel plugin that exposes GoodMem memory operations as kernel functions.
 * <p>
 * Register this plugin with the kernel so that the LLM can decide when to
 * save facts and when to recall them:
 *
 * <pre>{@code
 * var collection = GoodMemCollection.of("agent-memory", Fact.class);
 * collection.ensureCollectionExists().block();
 *
 * KernelPlugin memoryPlugin = KernelPluginFactory.createFromObject(
 *         new GoodMemPlugin(collection), "memory");
 * kernel.getPlugins().add(memoryPlugin);
 * }</pre>
 *
 * <p>The record type {@code T} must have a {@link GoodMemData} field whose
 * storage name or Java field name is {@code "content"} — this is the field
 * that the {@code save} function writes into and the {@code recall} function
 * reads from when formatting the response string.
 *
 * @param <T> the record type managed by the underlying collection
 */
public final class GoodMemPlugin<T> {

    private final GoodMemCollection<T> collection;
    private final Class<T> recordClass;
    private final java.util.function.BiFunction<String, T, T> recordFactory;

    /**
     * Creates a plugin backed by the given collection.
     *
     * @param collection    the GoodMem collection to use for storage and retrieval
     * @param recordClass   the class of the record type
     * @param recordFactory a factory that creates a record instance from
     *                      ({@code content}, prototype) — used by {@link #save}.
     *                      The prototype is {@code null} on the first call; a
     *                      simple implementation just sets the content field.
     *
     * <pre>{@code
     * // Example factory for a Fact record with public fields:
     * new GoodMemPlugin<>(collection, Fact.class, (content, proto) -> {
     *     Fact f = new Fact();
     *     f.content = content;
     *     return f;
     * });
     * }</pre>
     */
    public GoodMemPlugin(GoodMemCollection<T> collection, Class<T> recordClass,
                         java.util.function.BiFunction<String, T, T> recordFactory) {
        this.collection = collection;
        this.recordClass = recordClass;
        this.recordFactory = recordFactory;
    }

    // ── Kernel functions ──────────────────────────────────────────────────────

    /**
     * Stores a new fact in long-term memory.
     *
     * @param content the text to remember
     * @return confirmation message
     */
    @DefineKernelFunction(
            name = "save",
            description = "Save a fact or piece of information to long-term memory.")
    public String save(
            @KernelFunctionParameter(name = "content", description = "The text to remember.")
            String content) {
        if (content == null || content.isBlank()) return "Nothing to save.";
        T record = recordFactory.apply(content, null);
        collection.upsert(record).block();
        return "Saved to memory.";
    }

    /**
     * Searches long-term memory and returns the most relevant results.
     *
     * @param query natural-language search query
     * @param top   maximum number of memories to return (default 3)
     * @return newline-separated list of relevant memory texts, or a fallback message
     */
    @DefineKernelFunction(
            name = "recall",
            description = "Search long-term memory for facts relevant to a query. "
                    + "Call this before answering factual questions.")
    public String recall(
            @KernelFunctionParameter(name = "query", description = "What to search for in memory.")
            String query,
            @KernelFunctionParameter(name = "top", description = "Number of memories to retrieve (default 3).",
                    defaultValue = "3")
            int top) {
        var results = collection.search(query, top)
                .collectList()
                .block();

        if (results == null || results.isEmpty()) return "(no relevant memories found)";

        return results.stream()
                .map(r -> extractContent(r.record()))
                .filter(s -> s != null && !s.isBlank())
                .collect(Collectors.joining("\n"));
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    /**
     * Extracts a displayable string from a record.
     * Tries to read the first {@link GoodMemData} field named "content";
     * falls back to {@link Object#toString()}.
     */
    private String extractContent(T record) {
        if (record == null) return null;
        for (var field : record.getClass().getDeclaredFields()) {
            if (field.isAnnotationPresent(GoodMemData.class)) {
                String sname = field.getAnnotation(GoodMemData.class).value();
                if (sname.isEmpty()) sname = field.getName();
                if ("content".equalsIgnoreCase(sname) || "content".equalsIgnoreCase(field.getName())) {
                    try {
                        field.setAccessible(true);
                        Object val = field.get(record);
                        return val != null ? val.toString() : null;
                    } catch (IllegalAccessException ignored) {}
                }
            }
        }
        return record.toString();
    }
}
