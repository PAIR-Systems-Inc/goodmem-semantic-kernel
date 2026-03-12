package ai.goodmem.samples;

// ExampleStore — GoodMemVectorStore managing multiple collections.
//
// A single store owns one shared HTTP connection used across all collections.
// This mirrors the pattern where you manage related spaces from one place.
//
// Required environment variables:
//   GOODMEM_BASE_URL   — GoodMem server URL  (default: http://localhost:8080)
//   GOODMEM_VERIFY_SSL — Set to 'false' for self-signed certs
//   GOODMEM_API_KEY    — GoodMem API key
//
// Run:
//   cd samples/java/ExampleStore
//   mvn install -f ../../../java/pom.xml -DskipTests   # build the library first
//   mvn exec:java

import ai.goodmem.semantickernel.GoodMemData;
import ai.goodmem.semantickernel.GoodMemKey;
import ai.goodmem.semantickernel.GoodMemVectorStore;
import reactor.core.scheduler.Schedulers;

import java.time.Duration;
import java.util.List;

public class ExampleStore {

    // ── Data model ────────────────────────────────────────────────────────────
    // @GoodMemKey  → maps to GoodMem memoryId (server-assigned UUID)
    // @GoodMemData named "content" → stored as originalContent
    // @GoodMemData with any other name → stored in metadata
    public static class Note {
        @GoodMemKey
        public String id;

        @GoodMemData
        public String content;

        @GoodMemData("source")
        public String source;
    }

    public static void main(String[] args) throws InterruptedException {
        for (var v : List.of("GOODMEM_API_KEY", "GOODMEM_BASE_URL")) {
            if (System.getenv(v) == null || System.getenv(v).isBlank())
                throw new IllegalStateException("Set " + v + " before running this example.");
        }

        try (var store = new GoodMemVectorStore()) {

            // 1. List all spaces currently visible to this API key.
            var names = store.listCollectionNames().collectList().block();
            System.out.println("Existing spaces: " + (names == null || names.isEmpty()
                    ? "(none)" : String.join(", ", names)));

            // 2. Get two collections from the same store (shared HTTP connection).
            var notes = store.getCollection("store-notes", Note.class);
            var todos = store.getCollection("store-todos", Note.class);

            // 3. Fresh slate for both.
            notes.ensureCollectionDeleted().block();
            notes.ensureCollectionExists().block();
            todos.ensureCollectionDeleted().block();
            todos.ensureCollectionExists().block();
            System.out.println("Both collections ready (fresh).");

            // 4. Write into each.
            notes.upsertAll(List.of(
                    note("The Eiffel Tower is in Paris", "facts"),
                    note("Mount Fuji is in Japan", "facts")
            )).blockLast();
            todos.upsertAll(List.of(
                    note("Buy groceries", "chat"),
                    note("Call the dentist", "chat")
            )).blockLast();
            System.out.println("Upserted into both collections.");

            // 5. Wait for server-side embeddings.
            System.out.println("Waiting for embeddings...");
            Thread.sleep(Duration.ofSeconds(3).toMillis());

            // 6. Search each independently.
            System.out.println("\n--- notes search: 'famous landmarks in europe' ---");
            notes.search("famous landmarks in europe", 3)
                    .doOnNext(r -> System.out.printf("  [%.3f] %s%n", r.score(), r.record().content))
                    .blockLast();

            System.out.println("\n--- todos search: 'health appointments' ---");
            todos.search("health appointments", 3)
                    .doOnNext(r -> System.out.printf("  [%.3f] %s%n", r.score(), r.record().content))
                    .blockLast();

            // Uncomment to clean up:
            // notes.ensureCollectionDeleted().block();
            // todos.ensureCollectionDeleted().block();
            // System.out.println("\nCollections deleted.");
        }

        Schedulers.shutdownNow();
    }

    private static Note note(String content, String source) {
        Note n = new Note();
        n.content = content;
        n.source = source;
        return n;
    }
}
