package ai.goodmem.semantickernel;

import com.fasterxml.jackson.databind.node.ObjectNode;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Reflection-driven schema for a GoodMem record type {@code T}.
 * <p>
 * Inspects public fields (and inherited public fields) annotated with
 * {@link GoodMemKey} and {@link GoodMemData} to determine how to map
 * between a Java object and the GoodMem REST payload.
 *
 * @param <T> the record type
 */
final class GoodMemSchema<T> {

    private final Class<T> type;
    private final Field keyField;
    private final Field contentField;
    private final List<FieldMapping> metaFields;

    record FieldMapping(Field field, String storageName) {}

    // ── Construction ──────────────────────────────────────────────────────────

    private GoodMemSchema(Class<T> type, Field keyField, Field contentField,
                          List<FieldMapping> metaFields) {
        this.type = type;
        this.keyField = keyField;
        this.contentField = contentField;
        this.metaFields = metaFields;
        keyField.setAccessible(true);
        contentField.setAccessible(true);
        metaFields.forEach(m -> m.field().setAccessible(true));
    }

    /**
     * Builds a schema for {@code recordClass} using {@link GoodMemKey} /
     * {@link GoodMemData} annotations found on its fields.
     */
    static <T> GoodMemSchema<T> build(Class<T> recordClass) {
        List<Field> allFields = new ArrayList<>();
        for (Class<?> c = recordClass; c != null && c != Object.class; c = c.getSuperclass()) {
            for (Field f : c.getDeclaredFields()) {
                allFields.add(f);
            }
        }

        Field keyField = null;
        List<FieldMapping> dataFields = new ArrayList<>();

        for (Field f : allFields) {
            if (f.isAnnotationPresent(GoodMemKey.class)) {
                keyField = f;
                continue;
            }
            if (f.isAnnotationPresent(GoodMemData.class)) {
                String storageName = f.getAnnotation(GoodMemData.class).value();
                if (storageName.isEmpty()) storageName = f.getName();
                dataFields.add(new FieldMapping(f, storageName));
            }
        }

        if (keyField == null)
            throw new IllegalArgumentException(
                    "Record type '" + recordClass.getSimpleName() + "' has no @GoodMemKey field. "
                    + "Add @GoodMemKey to the field that maps to the GoodMem memory ID.");

        if (dataFields.isEmpty())
            throw new IllegalArgumentException(
                    "Record type '" + recordClass.getSimpleName() + "' has no @GoodMemData fields. "
                    + "At least one @GoodMemData field is required.");

        Field contentField = resolveContentField(dataFields);
        List<FieldMapping> metaFields = dataFields.stream()
                .filter(m -> m.field() != contentField)
                .toList();

        return new GoodMemSchema<>(recordClass, keyField, contentField, metaFields);
    }

    /** Resolution order: name=="content" → first String field → first field. */
    private static Field resolveContentField(List<FieldMapping> dataFields) {
        for (var m : dataFields)
            if ("content".equalsIgnoreCase(m.storageName()) || "content".equalsIgnoreCase(m.field().getName()))
                return m.field();
        for (var m : dataFields)
            if (m.field().getType() == String.class)
                return m.field();
        return dataFields.get(0).field();
    }

    // ── Serialization (record → API payload) ─────────────────────────────────

    SerializedRecord serialize(T record) {
        String memoryId = readString(keyField, record);
        String content = readString(contentField, record);
        if (content == null) content = "";

        Map<String, Object> metadata = null;
        if (!metaFields.isEmpty()) {
            metadata = new java.util.LinkedHashMap<>();
            for (var m : metaFields) {
                Object val = readRaw(m.field(), record);
                if (val != null) metadata.put(m.storageName(), val);
            }
            if (metadata.isEmpty()) metadata = null;
        }
        return new SerializedRecord(content, metadata, memoryId);
    }

    record SerializedRecord(String content, Map<String, Object> metadata, String memoryId) {}

    void setKey(T record, String key) {
        try {
            keyField.set(record, key);
        } catch (IllegalAccessException | IllegalArgumentException ignored) {
            // field not writable — nothing to do
        }
    }

    // ── Deserialization (batchGet response → record) ──────────────────────────

    T deserialize(ObjectNode mem) {
        T instance = newInstance();

        String memoryId = mem.path("memoryId").asText(null);
        if (memoryId != null) setKey(instance, memoryId);

        String content = mem.path("originalContent").asText("");
        setField(contentField, instance, content);

        ObjectNode metadata = (ObjectNode) mem.get("metadata");
        if (metadata != null) applyMetadata(instance, metadata);

        return instance;
    }

    // ── Deserialization (retrieve/search response → record) ───────────────────

    T deserializeFromRetrieve(ObjectNode chunk, ObjectNode memory) {
        T instance = newInstance();

        String memoryId = chunk.path("memoryId").asText(null);
        if (memoryId == null) memoryId = memory.path("memoryId").asText(null);
        if (memoryId != null) setKey(instance, memoryId);

        // Content comes from chunk.chunkText in search results.
        String content = chunk.path("chunkText").asText("");
        setField(contentField, instance, content);

        ObjectNode metadata = (ObjectNode) memory.get("metadata");
        if (metadata != null) applyMetadata(instance, metadata);

        return instance;
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private void applyMetadata(T instance, ObjectNode metadata) {
        for (var m : metaFields) {
            var node = metadata.get(m.storageName());
            if (node == null || node.isNull()) continue;
            Class<?> ft = m.field().getType();
            try {
                if (ft == String.class)         m.field().set(instance, node.asText());
                else if (ft == int.class || ft == Integer.class)     m.field().set(instance, node.asInt());
                else if (ft == long.class || ft == Long.class)       m.field().set(instance, node.asLong());
                else if (ft == double.class || ft == Double.class)   m.field().set(instance, node.asDouble());
                else if (ft == float.class || ft == Float.class)     m.field().set(instance, (float) node.asDouble());
                else if (ft == boolean.class || ft == Boolean.class) m.field().set(instance, node.asBoolean());
            } catch (IllegalAccessException ignored) {}
        }
    }

    private static String readString(Field f, Object obj) {
        Object val = readRaw(f, obj);
        return val instanceof String s ? s : (val != null ? val.toString() : null);
    }

    private static Object readRaw(Field f, Object obj) {
        try { return f.get(obj); }
        catch (IllegalAccessException e) { return null; }
    }

    private void setField(Field f, T obj, Object value) {
        try { f.set(obj, value); }
        catch (IllegalAccessException | IllegalArgumentException ignored) {}
    }

    @SuppressWarnings("unchecked")
    private T newInstance() {
        try {
            var ctor = type.getDeclaredConstructor();
            ctor.setAccessible(true);
            return ctor.newInstance();
        } catch (Exception e) {
            throw new GoodMemException(
                    "Cannot instantiate '" + type.getSimpleName() + "': ensure it has a no-arg constructor. "
                    + e.getMessage(), e);
        }
    }
}
