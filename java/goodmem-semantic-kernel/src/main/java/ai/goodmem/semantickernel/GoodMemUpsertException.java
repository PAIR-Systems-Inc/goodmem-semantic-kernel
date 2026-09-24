package ai.goodmem.semantickernel;

/**
 * Thrown when an upsert fails while replacing an existing record.
 *
 * <p>GoodMem has no update endpoint ({@code PUT} and {@code PATCH} return 404) and
 * rejects a create that reuses an existing ID with 409, so replacing a record means
 * deleting the old memory before writing the new one. When the write fails, the
 * connector puts the previous version back; {@link #isRestored()} says whether that
 * succeeded and {@link #getLostKey()} names the record if it did not.
 */
public class GoodMemUpsertException extends GoodMemException {

    private final boolean restored;
    private final String lostKey;

    public GoodMemUpsertException(String message, boolean restored, String lostKey, Throwable cause) {
        super(message, cause);
        this.restored = restored;
        this.lostKey = lostKey;
    }

    /** Whether the previous version was successfully restored. */
    public boolean isRestored() {
        return restored;
    }

    /** The key whose previous version could not be restored, or {@code null}. */
    public String getLostKey() {
        return lostKey;
    }
}
