package ai.goodmem.semantickernel;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.logging.Handler;
import java.util.logging.Level;
import java.util.logging.LogRecord;
import java.util.logging.Logger;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * Retrieval streams captured from a live GoodMem server (v1.0.320), and variations of them.
 *
 * <p>{@code retrieve_ok} is a healthy search with one hit. {@code retrieve_degraded_hits} is
 * the same search with a reranker id that does not exist: the server sends {@code NOT_FOUND},
 * {@code FEATURE_DISABLED} and {@code RERANKING_FAILED}, then the vector hit.
 * {@code retrieve_degraded_empty} is that search on an empty space.
 */
final class RetrievalFixtures {

    /** The text of the one chunk in the captured streams. */
    static final String CANARY_TEXT = "The fixture canary is ORYX-2290. CAMEL toolkit audit.\n";

    /** Its raw vector score in the degraded stream (negative; lower is better). */
    static final double DEGRADED_RAW_SCORE = -0.5845972299575806;

    static final String MISSING_RERANKER = "00000000-0000-7000-8000-000000000000";

    private RetrievalFixtures() {}

    static String load(String name) {
        try (InputStream in = RetrievalFixtures.class.getResourceAsStream("/fixtures/" + name + ".ndjson")) {
            if (in == null) throw new IllegalStateException("no fixture " + name);
            return new String(in.readAllBytes(), StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /** A status event line, as the server writes one. */
    static String statusLine(String code, String message, String detailsJson) {
        String codePart = code == null ? "" : "\"code\":\"" + code + "\",";
        String detailsPart = detailsJson == null ? "" : ",\"details\":" + detailsJson;
        return "{\"status\":{" + codePart + "\"message\":\"" + message + "\"" + detailsPart + "}}";
    }

    /** The healthy captured stream with extra lines before it. */
    static String okWith(String... leadingLines) {
        return String.join("\n", leadingLines) + "\n" + load("retrieve_ok");
    }

    /** Records what one java.util.logging logger publishes, which is where System.Logger goes by default. */
    static final class LogCapture extends Handler implements AutoCloseable {
        private final Logger logger;
        private final List<LogRecord> records = new CopyOnWriteArrayList<>();

        LogCapture(String loggerName) {
            logger = Logger.getLogger(loggerName);
            setLevel(Level.ALL);
            logger.addHandler(this);
        }

        @Override public void publish(LogRecord record) { records.add(record); }
        @Override public void flush() {}
        @Override public void close() { logger.removeHandler(this); }

        List<LogRecord> records() { return List.copyOf(records); }
    }
}
