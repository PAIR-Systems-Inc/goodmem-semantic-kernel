package ai.goodmem.semantickernel;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Schedulers;

import javax.net.ssl.SSLContext;
import javax.net.ssl.TrustManager;
import javax.net.ssl.X509TrustManager;
import java.net.URI;
import java.net.URLEncoder;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.security.SecureRandom;
import java.security.cert.X509Certificate;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Internal HTTP client wrapping the GoodMem REST API.
 *
 * <p>All methods are package-private. Consumers should use
 * {@link GoodMemCollection} or {@link GoodMemVectorStore} instead.
 *
 * <p>Each method wraps a blocking {@link HttpClient#send} call inside
 * {@code Mono.fromCallable(...).subscribeOn(Schedulers.boundedElastic())}
 * so it integrates cleanly with Project Reactor without tying up the
 * event-loop thread.
 */
final class GoodMemClient implements AutoCloseable {

    static final ObjectMapper MAPPER = new ObjectMapper();
    private static final Duration TIMEOUT = Duration.ofSeconds(30);

    private final HttpClient http;
    private final String baseUrl;
    private final String apiKey;

    GoodMemClient(GoodMemOptions options) {
        this.baseUrl = options.getBaseUrl().replaceAll("/+$", "");
        this.apiKey  = options.getApiKey().strip();
        this.http    = options.isVerifySsl() ? defaultClient() : insecureClient();
    }

    /** Package-private constructor for tests — injects a pre-built HttpClient. */
    GoodMemClient(HttpClient http, String baseUrl, String apiKey) {
        this.http    = http;
        this.baseUrl = baseUrl.replaceAll("/+$", "");
        this.apiKey  = apiKey;
    }

    @Override
    public void close() {
        // java.net.http.HttpClient does not implement Closeable before Java 21.
        // Nothing to release here; the JVM will GC the underlying connections.
    }

    // ── Spaces ────────────────────────────────────────────────────────────────

    Mono<List<ObjectNode>> listSpaces(String nameFilter) {
        return Mono.fromCallable(() -> {
            List<ObjectNode> all = new ArrayList<>();
            String nextToken = null;
            do {
                StringBuilder qs = new StringBuilder("?maxResults=1000");
                if (nextToken  != null) qs.append("&nextToken=").append(encode(nextToken));
                if (nameFilter != null) qs.append("&nameFilter=").append(encode(nameFilter));

                ObjectNode data = doGet("v1/spaces" + qs);
                for (JsonNode item : iterArray(data, "spaces"))
                    if (item.isObject()) all.add((ObjectNode) item);

                String tok = data.path("nextToken").asText(null);
                nextToken = (tok != null && !tok.isEmpty()) ? tok : null;
            } while (nextToken != null);
            return all;
        }).subscribeOn(Schedulers.boundedElastic());
    }

    /**
     * @param spaceId optional — when non-null it is sent as the requested UUID;
     *                the server may honour or ignore it.
     */
    Mono<ObjectNode> createSpace(String name, String embedderId, String spaceId) {
        return Mono.fromCallable(() -> {
            ObjectNode payload = MAPPER.createObjectNode().put("name", name);

            ArrayNode embedders = payload.putArray("spaceEmbedders");
            embedders.addObject()
                    .put("embedderId", embedderId)
                    .put("defaultRetrievalWeight", 1.0);

            payload.putObject("defaultChunkingConfig")
                    .putObject("recursive")
                    .put("chunkSize", 512)
                    .put("chunkOverlap", 64)
                    .put("keepStrategy", "KEEP_END")
                    .put("lengthMeasurement", "CHARACTER_COUNT");

            if (spaceId != null) payload.put("spaceId", spaceId);
            return doPost("v1/spaces", payload);
        }).subscribeOn(Schedulers.boundedElastic());
    }

    Mono<Void> deleteSpace(String spaceId) {
        return Mono.fromCallable(() -> {
            // The id is a path segment: only a UUID may reach it (see GoodMemIds).
            doDelete("v1/spaces/" + GoodMemIds.requireUuid(spaceId, "spaceId"), false);
            return (Void) null;
        }).subscribeOn(Schedulers.boundedElastic()).then();
    }

    // ── Embedders ─────────────────────────────────────────────────────────────

    Mono<List<ObjectNode>> listEmbedders() {
        return Mono.fromCallable(() -> {
            ObjectNode data = doGet("v1/embedders");
            List<ObjectNode> result = new ArrayList<>();
            for (JsonNode item : iterArray(data, "embedders"))
                if (item.isObject()) result.add((ObjectNode) item);
            return result;
        }).subscribeOn(Schedulers.boundedElastic());
    }

    // ── Memories ──────────────────────────────────────────────────────────────

    Mono<ObjectNode> createMemory(
            String spaceId,
            String content,
            String contentType,
            Map<String, Object> metadata,
            String memoryId) {
        return Mono.fromCallable(() -> {
            ObjectNode payload = MAPPER.createObjectNode()
                    .put("spaceId", spaceId)
                    .put("originalContent", content)
                    .put("contentType", contentType != null ? contentType : "text/plain");

            if (metadata != null && !metadata.isEmpty())
                payload.set("metadata", MAPPER.valueToTree(metadata));
            if (memoryId != null)
                payload.put("memoryId", memoryId);

            return doPost("v1/memories", payload);
        }).subscribeOn(Schedulers.boundedElastic());
    }

    Mono<List<ObjectNode>> batchGetMemories(List<String> memoryIds) {
        if (memoryIds.isEmpty()) return Mono.just(List.of());

        return Mono.fromCallable(() -> {
            ObjectNode payload = MAPPER.createObjectNode();
            ArrayNode ids = payload.putArray("memoryIds");
            memoryIds.forEach(ids::add);
            payload.put("includeContent", true);

            ObjectNode data = doPost("v1/memories:batchGet", payload);

            // Response: { "results": [ { "success": true, "memory": {...} }, ... ] }
            List<ObjectNode> result = new ArrayList<>();
            for (JsonNode item : iterArray(data, "results")) {
                if (item.path("success").asBoolean() && item.path("memory").isObject())
                    result.add((ObjectNode) item.get("memory"));
            }
            return result;
        }).subscribeOn(Schedulers.boundedElastic());
    }

    /** 404 responses are silently ignored (already deleted). */
    Mono<Void> deleteMemory(String memoryId) {
        return Mono.fromCallable(() -> {
            // The id is a path segment: only a UUID may reach it (see GoodMemIds).
            doDelete("v1/memories/" + GoodMemIds.requireUuid(memoryId, "memoryId"), true);
            return (Void) null;
        }).subscribeOn(Schedulers.boundedElastic()).then();
    }

    /**
     * Semantic search over one or more spaces.
     *
     * <p>GoodMem returns a streaming NDJSON body. Each line is a JSON event:
     * <ul>
     *   <li>{@code {"memoryDefinition": {...}}} — arrives indexed by position 0, 1, 2, …
     *   <li>{@code {"retrievedItem": {"chunk": {"memoryIndex": N, "relevanceScore": …}}}}
     *       — references a memory definition by index
     * </ul>
     *
     * <p>Scores are negated before returning: GoodMem uses lower-is-better (distance),
     * SK uses higher-is-better (similarity).
     */
    Mono<List<RetrieveResult>> retrieveMemories(String query, List<String> spaceIds, int top) {
        return Mono.fromCallable(() -> {
            ObjectNode payload = MAPPER.createObjectNode()
                    .put("message", query)
                    .put("requestedSize", top);

            ArrayNode keys = payload.putArray("spaceKeys");
            spaceIds.forEach(sid -> keys.addObject().put("spaceId", sid));

            String json = MAPPER.writeValueAsString(payload);
            HttpRequest request = baseRequest("v1/memories:retrieve")
                    .header("Accept", "application/x-ndjson")
                    .POST(HttpRequest.BodyPublishers.ofString(json))
                    .build();

            HttpResponse<String> response = http.send(request, HttpResponse.BodyHandlers.ofString());
            ensureSuccess(response);

            List<ObjectNode> memoryList = new ArrayList<>();
            List<ObjectNode> chunkRefs  = new ArrayList<>();

            for (String line : response.body().split("\n")) {
                String trimmed = line.strip();
                if (trimmed.isEmpty()) continue;

                JsonNode evt;
                try { evt = MAPPER.readTree(trimmed); } catch (Exception e) { continue; }

                JsonNode memDef = evt.path("memoryDefinition");
                if (memDef.isObject()) {
                    memoryList.add((ObjectNode) memDef);
                    continue;
                }

                JsonNode chunkRef = evt.path("retrievedItem").path("chunk");
                if (chunkRef.isObject()) chunkRefs.add((ObjectNode) chunkRef);
            }

            List<RetrieveResult> results = new ArrayList<>(chunkRefs.size());
            for (ObjectNode chunkRef : chunkRefs) {
                JsonNode chunkNode = chunkRef.path("chunk");
                ObjectNode chunk = chunkNode.isObject()
                        ? (ObjectNode) chunkNode
                        : MAPPER.createObjectNode();

                double rawScore  = chunkRef.path("relevanceScore").asDouble(0.0);
                int    memIndex  = chunkRef.path("memoryIndex").asInt(-1);
                ObjectNode mem   = (memIndex >= 0 && memIndex < memoryList.size())
                        ? memoryList.get(memIndex)
                        : MAPPER.createObjectNode();

                results.add(new RetrieveResult(chunk, mem, -rawScore)); // negate → higher-is-better
            }
            return results;
        }).subscribeOn(Schedulers.boundedElastic());
    }

    // ── Internal HTTP helpers ─────────────────────────────────────────────────

    private ObjectNode doGet(String path) throws Exception {
        HttpResponse<String> resp = http.send(
                baseRequest(path).GET().build(),
                HttpResponse.BodyHandlers.ofString());
        ensureSuccess(resp);
        return (ObjectNode) MAPPER.readTree(resp.body());
    }

    private ObjectNode doPost(String path, ObjectNode body) throws Exception {
        String json = MAPPER.writeValueAsString(body);
        HttpResponse<String> resp = http.send(
                baseRequest(path)
                        .POST(HttpRequest.BodyPublishers.ofString(json))
                        .build(),
                HttpResponse.BodyHandlers.ofString());
        ensureSuccess(resp);
        return (ObjectNode) MAPPER.readTree(resp.body());
    }

    private void doDelete(String path, boolean notFoundOk) throws Exception {
        HttpResponse<String> resp = http.send(
                baseRequest(path).DELETE().build(),
                HttpResponse.BodyHandlers.ofString());
        if (notFoundOk && resp.statusCode() == 404) return;
        ensureSuccess(resp);
    }

    private HttpRequest.Builder baseRequest(String path) {
        return HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + "/" + path))
                .header("x-api-key", apiKey)
                .header("Content-Type", "application/json")
                .header("Accept", "application/json")
                .timeout(TIMEOUT);
    }

    private static void ensureSuccess(HttpResponse<String> response) {
        if (response.statusCode() >= 400)
            throw new GoodMemException(
                    "GoodMem API error " + response.statusCode() + ": " + response.body());
    }

    private static String encode(String value) {
        return URLEncoder.encode(value, StandardCharsets.UTF_8);
    }

    /** Null-safe iteration over a named JSON array field. */
    private static Iterable<JsonNode> iterArray(JsonNode parent, String field) {
        JsonNode arr = parent.path(field);
        return arr.isArray() ? arr : List.of();
    }

    private static HttpClient defaultClient() {
        return HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(10))
                .build();
    }

    @SuppressWarnings("java:S4830") // intentional: SSL verification disabled by user request
    private static HttpClient insecureClient() {
        try {
            TrustManager[] trustAll = {new X509TrustManager() {
                public void checkClientTrusted(X509Certificate[] c, String a) {}
                public void checkServerTrusted(X509Certificate[] c, String a) {}
                public X509Certificate[] getAcceptedIssuers() { return new X509Certificate[0]; }
            }};
            SSLContext ctx = SSLContext.getInstance("TLS");
            ctx.init(null, trustAll, new SecureRandom());
            return HttpClient.newBuilder()
                    .sslContext(ctx)
                    .connectTimeout(Duration.ofSeconds(10))
                    .build();
        } catch (Exception e) {
            throw new GoodMemException("Failed to build insecure SSL context", e);
        }
    }

    /** A correlated retrieve result: chunk data, parent memory metadata, and similarity score. */
    record RetrieveResult(ObjectNode chunk, ObjectNode memory, double score) {}
}
