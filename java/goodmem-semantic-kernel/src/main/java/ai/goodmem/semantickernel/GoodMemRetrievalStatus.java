package ai.goodmem.semantickernel;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Collectors;

/**
 * A problem the GoodMem server reported while it ran a search.
 *
 * <p>Only statuses that mean something may be missing from the results are reported.
 * {@code FEATURE_DISABLED} and {@code LLM_CAPABILITY_INFERRED} are notices about optional
 * features the search did not ask for, and are left out by their code alone. A code this
 * connector does not know is reported as {@code UNKNOWN}, with the server's own code kept in
 * {@link #originalCode()}. A line of the response stream that cannot be read is reported as
 * {@code MALFORMED_STREAM}.
 *
 * @param code         GoodMem's status code, such as {@code RERANKING_FAILED}; {@code UNKNOWN}
 *                     for a code this connector does not recognise; or {@code MALFORMED_STREAM}
 *                     for a line it could not read
 * @param message      the server's human-readable message
 * @param details      the server's details, such as {@code reranker_id}; empty when it sent none
 * @param originalCode the code exactly as the server sent it, when {@code code} is
 *                     {@code UNKNOWN}; {@code null} otherwise, and when the server sent no code
 * @param unrecognized {@code true} when the server's code is not one this connector recognises
 */
public record GoodMemRetrievalStatus(
        String code,
        String message,
        Map<String, String> details,
        String originalCode,
        boolean unrecognized) {

    public GoodMemRetrievalStatus {
        Objects.requireNonNull(code, "code");
        message = message == null ? "" : message;
        details = details == null
                ? Map.of()
                : Collections.unmodifiableMap(new LinkedHashMap<>(details));
    }

    @Override
    public String toString() {
        String named = originalCode == null ? code : code + " (server code " + originalCode + ")";
        String extra = details.isEmpty()
                ? ""
                : details.entrySet().stream()
                        .map(e -> e.getKey() + "=" + e.getValue())
                        .collect(Collectors.joining(", ", " {", "}"));
        return named + ": " + message + extra;
    }
}
