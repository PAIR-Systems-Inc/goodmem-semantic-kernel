package ai.goodmem.semantickernel;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Marks the field or method that holds the GoodMem memory ID (primary key).
 * <p>
 * The annotated field must be of type {@code String}. After a successful
 * {@link GoodMemCollection#upsert upsert}, GoodMem writes the server-assigned
 * ID back into this field (if it is writable).
 *
 * <pre>{@code
 * public class Note {
 *     @GoodMemKey
 *     public String id;
 *
 *     @GoodMemData
 *     public String content;
 * }
 * }</pre>
 */
@Target({ElementType.FIELD, ElementType.METHOD})
@Retention(RetentionPolicy.RUNTIME)
public @interface GoodMemKey {
}
