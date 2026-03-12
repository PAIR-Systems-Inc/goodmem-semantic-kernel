package ai.goodmem.semantickernel;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Marks a field or method as a data property for GoodMem storage.
 * <p>
 * Resolution order for which field becomes the GoodMem {@code originalContent}:
 * <ol>
 *   <li>A field whose {@link #value()} storage name or whose Java field name is
 *       {@code "content"} (case-insensitive).</li>
 *   <li>The first {@code String}-typed {@code @GoodMemData} field.</li>
 *   <li>The first {@code @GoodMemData} field regardless of type.</li>
 * </ol>
 * All other {@code @GoodMemData} fields are stored as GoodMem metadata.
 *
 * <pre>{@code
 * public class Note {
 *     @GoodMemKey
 *     public String id;
 *
 *     @GoodMemData                   // becomes originalContent
 *     public String content;
 *
 *     @GoodMemData("source")         // stored as metadata["source"]
 *     public String source;
 * }
 * }</pre>
 */
@Target({ElementType.FIELD, ElementType.METHOD})
@Retention(RetentionPolicy.RUNTIME)
public @interface GoodMemData {
    /**
     * Override the storage name used as the metadata key.
     * Defaults to the Java field name when empty.
     */
    String value() default "";
}
