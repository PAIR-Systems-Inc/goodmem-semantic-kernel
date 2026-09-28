package ai.goodmem.semantickernel;

import com.fasterxml.jackson.databind.JsonNode;

import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Sorts the {@code status} events of a retrieval stream: the retrieval status contract every
 * GoodMem integration follows.
 */
final class RetrievalStatuses {

    static final String UNKNOWN = "UNKNOWN";
    static final String MALFORMED_STREAM = "MALFORMED_STREAM";

    /** Notices that carry no loss of results, by their code alone (contract Q1). */
    private static final Set<String> INFORMATIONAL = Set.of("LLM_CAPABILITY_INFERRED", "FEATURE_DISABLED");

    /** Every code GoodMem's API defines for a status ({@code GoodMemStatus.code}, server v1.0.320). */
    private static final Set<String> KNOWN = Set.of(
            "GOODMEM_STATUS_CODE_UNSPECIFIED", "INVALID_ARGUMENT", "NOT_FOUND", "PERMISSION_DENIED",
            "FAILED_PRECONDITION", "EMBEDDER_FAILED", "EMBEDDER_UNAVAILABLE", "EMBEDDER_TIMEOUT",
            "VECTOR_SEARCH_FAILED", "VECTOR_SEARCH_PARTIAL", "VECTOR_SEARCH_TIMEOUT", "SPACE_INACCESSIBLE",
            "SPACE_NOT_FOUND", "SPACE_NO_EMBEDDERS", "CHUNK_NOT_FOUND", "MEMORY_LOAD_FAILED",
            "MEMORY_CONTENT_UNAVAILABLE", "RERANKING_FAILED", "SUMMARIZATION_FAILED", "SUMMARIZATION_TIMEOUT",
            "RATE_LIMITED", "RESOURCE_EXHAUSTED", "CONFIGURATION_ERROR", "LLM_CAPABILITY_INFERRED",
            "FEATURE_DISABLED");

    private RetrievalStatuses() {}

    /**
     * Returns the status to report, or {@code null} for an informational notice.
     *
     * <p>Never throws: a code this connector does not know, or no code at all, is reported as
     * {@code UNKNOWN} (contract Q3), so a newer server cannot make a failure look like success.
     */
    static GoodMemRetrievalStatus classify(JsonNode status) {
        String code = text(status.path("code"));
        if (code != null && INFORMATIONAL.contains(code)) return null;

        String message = text(status.path("message"));
        Map<String, String> details = new LinkedHashMap<>();
        for (Iterator<Map.Entry<String, JsonNode>> it = status.path("details").fields(); it.hasNext(); ) {
            Map.Entry<String, JsonNode> entry = it.next();
            String value = text(entry.getValue());
            details.put(entry.getKey(), value == null ? "" : value);
        }

        return code != null && KNOWN.contains(code)
                ? new GoodMemRetrievalStatus(code, message, details, null, false)
                : new GoodMemRetrievalStatus(UNKNOWN, message, details, code, true);
    }

    /** A line of the stream that could not be read, reported like a server status. */
    static GoodMemRetrievalStatus malformed(String message) {
        return new GoodMemRetrievalStatus(MALFORMED_STREAM, message, Map.of(), null, false);
    }

    /**
     * One line naming every status, for a log message. Line breaks in the server's text are
     * flattened so a message cannot start a log line of its own.
     */
    static String describe(List<GoodMemRetrievalStatus> statuses) {
        return statuses.stream()
                .map(GoodMemRetrievalStatus::toString)
                .collect(Collectors.joining("; ", "[", "]"))
                .replace('\r', ' ')
                .replace('\n', ' ');
    }

    private static String text(JsonNode node) {
        if (node == null || node.isMissingNode() || node.isNull()) return null;
        return node.isTextual() ? node.asText() : node.toString();
    }
}
