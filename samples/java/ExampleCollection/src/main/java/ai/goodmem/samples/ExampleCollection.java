package ai.goodmem.samples;

// ExampleCollection — single GoodMemCollection for basic CRUD + search.
//
// The simplest way to use the GoodMem SK connector: one collection, one model,
// direct upsert / get / search / delete operations.
//
// Required environment variables:
//   GOODMEM_BASE_URL   — GoodMem server URL  (default: http://localhost:8080)
//   GOODMEM_VERIFY_SSL — Set to 'false' for self-signed certs
//   GOODMEM_API_KEY    — GoodMem API key
//
// Run:
//   cd samples/java/ExampleCollection
//   mvn install -f ../../../java/pom.xml -DskipTests
//   mvn compile exec:java

import ai.goodmem.semantickernel.GoodMemCollection;
import ai.goodmem.semantickernel.GoodMemData;
import ai.goodmem.semantickernel.GoodMemKey;
import reactor.core.scheduler.Schedulers;

import java.util.List;

public class ExampleCollection {

    // ── Data model ────────────────────────────────────────────────────────────
    public static class Fact {
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

        // One collection = one GoodMem space.
        try (var facts = GoodMemCollection.of("example-facts", Fact.class)) {

            // 1. Fresh space.
            facts.ensureCollectionDeleted().block();
            facts.ensureCollectionExists().block();
            System.out.println("Collection 'example-facts' ready.");

            // 2. Upsert records.
            var ids = facts.upsertAll(List.of(
                    fact("The Great Wall of China is over 13,000 miles long.", "geography"),
                    fact("Water boils at 100 degrees Celsius at sea level.", "science"),
                    fact("Ada Lovelace is considered the first computer programmer.", "technology")
            )).collectList().block();
            System.out.println("Upserted 3 facts, IDs: " + ids);

            // 3. Wait for server-side embeddings.
            System.out.println("Waiting for embeddings...");
            Thread.sleep(3_000);

            // 4. Get by ID.
            if (ids != null && !ids.isEmpty()) {
                var first = facts.get(ids.get(0)).block();
                System.out.println("\nGet by ID '" + ids.get(0) + "':");
                if (first != null) System.out.println("  content: " + first.content);
            }

            // 5. Search.
            System.out.println("\n--- search: 'famous structures' ---");
            facts.search("famous structures", 3)
                    .doOnNext(r -> System.out.printf("  [%.3f] [%s] %s%n",
                            r.score(), r.record().source, r.record().content))
                    .blockLast();

            System.out.println("\n--- search: 'pioneers in computing' ---");
            facts.search("pioneers in computing", 3)
                    .doOnNext(r -> System.out.printf("  [%.3f] [%s] %s%n",
                            r.score(), r.record().source, r.record().content))
                    .blockLast();

            // 6. Delete one record.
            if (ids != null && !ids.isEmpty()) {
                facts.delete(ids.get(0)).block();
                System.out.println("\nDeleted first fact. Remaining count:");
                var remaining = facts.search("*", 10).collectList().block();
                System.out.println("  " + (remaining != null ? remaining.size() : 0) + " result(s)");
            }

            // Uncomment to clean up:
            // facts.ensureCollectionDeleted().block();
        }

        Schedulers.shutdownNow();
    }

    private static Fact fact(String content, String source) {
        Fact f = new Fact();
        f.content = content;
        f.source = source;
        return f;
    }
}
