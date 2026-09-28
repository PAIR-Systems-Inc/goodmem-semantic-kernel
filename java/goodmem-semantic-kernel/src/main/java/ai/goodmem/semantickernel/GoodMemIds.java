package ai.goodmem.semantickernel;

import java.util.Locale;
import java.util.regex.Pattern;

/**
 * The one check every GoodMem id passes before the connector sends it.
 *
 * <p>Ids become URL path segments ({@code v1/memories/{id}}). Escaping them is not
 * enough: the GoodMem server decodes {@code %2e%2e} back into {@code ..}, so an id
 * such as {@code ../spaces/<uuid>} could turn a memory delete into a space delete.
 * GoodMem ids (memories, spaces, embedders) are all UUIDs, so anything that is not
 * one is refused before a request is made with it.
 */
final class GoodMemIds {

    // Matched with matches(), which must consume the whole input.
    private static final Pattern UUID = Pattern.compile(
            "[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}");

    private GoodMemIds() {}

    /**
     * Returns {@code value} as a lowercase canonical UUID.
     *
     * @param field the argument or setting name, used in the error
     * @throws IllegalArgumentException if {@code value} is not a UUID
     */
    static String requireUuid(String value, String field) {
        if (value != null && UUID.matcher(value).matches()) {
            return value.toLowerCase(Locale.ROOT);
        }
        String shown = value == null
                ? "null"
                : "'" + (value.length() > 64 ? value.substring(0, 61) + "..." : value) + "'";
        throw new IllegalArgumentException(field + " must be a GoodMem UUID, got " + shown
                + ". It was not sent: GoodMem ids are UUIDs, and any other value can change"
                + " which URL a request goes to.");
    }
}
