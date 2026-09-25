package ai.goodmem.semantickernel;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.Locale;
import java.util.stream.Stream;

import static org.assertj.core.api.Assertions.*;

class GoodMemIdsTest {

    private static final String U = GoodMemIdValidationTest.U;

    @Test
    void normalisesToLowercase() {
        assertThat(GoodMemIds.requireUuid(U.toUpperCase(Locale.ROOT), "key")).isEqualTo(U);
    }

    static Stream<String> refused() {
        return Stream.concat(GoodMemIdValidationTest.payloads(), Stream.of(
                null,
                U + " ",
                U.replace("-", ""),
                "{" + U + "}",
                U.substring(0, 35) + "١")); // Arabic-Indic digit one
    }

    @ParameterizedTest
    @MethodSource("refused")
    void refusesEverythingElseNamingTheField(String payload) {
        assertThatThrownBy(() -> GoodMemIds.requireUuid(payload, "key"))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageStartingWith("key must be a GoodMem UUID");
    }
}
