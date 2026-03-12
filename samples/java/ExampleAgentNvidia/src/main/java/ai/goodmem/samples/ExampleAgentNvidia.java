package ai.goodmem.samples;

// ExampleAgentNvidia — GoodMem collection wired into a Semantic Kernel agent (NVIDIA NIM).
//
// The agent has a memory search tool backed by GoodMem. When the user asks a
// question, the LLM decides whether to call the tool to look up relevant
// memories before composing its answer.
//
// Note: This sample uses NVIDIA's OpenAI-compatible NIM endpoint via the
// SK OpenAI connector with a custom endpoint URI. The model must support
// tool/function calling — check https://build.nvidia.com for model capability
// details and to obtain an API key.
//
// Required environment variables:
//   GOODMEM_BASE_URL   — GoodMem server URL  (default: http://localhost:8080)
//   GOODMEM_VERIFY_SSL — Set to 'false' for self-signed certs
//   GOODMEM_API_KEY    — GoodMem API key
//   NVIDIA_API_KEY     — NVIDIA API key (https://build.nvidia.com)
//
// Optional environment variables:
//   NVIDIA_MODEL       — NVIDIA NIM model ID (default: meta/llama-3.1-70b-instruct)
//                        Must support OpenAI tool_calls format; 8b models do not.
//
// Run:
//   cd samples/java/ExampleAgentNvidia
//   mvn install -f ../../../java/pom.xml -DskipTests
//   mvn compile exec:java

import ai.goodmem.semantickernel.GoodMemCollection;
import ai.goodmem.semantickernel.GoodMemData;
import ai.goodmem.semantickernel.GoodMemKey;
import ai.goodmem.semantickernel.GoodMemPlugin;
import reactor.core.scheduler.Schedulers;
import reactor.netty.http.HttpResources;
import com.microsoft.semantickernel.Kernel;
import com.microsoft.semantickernel.aiservices.openai.chatcompletion.OpenAIChatCompletion;
import com.microsoft.semantickernel.orchestration.InvocationContext;
import com.microsoft.semantickernel.orchestration.ToolCallBehavior;
import com.microsoft.semantickernel.plugin.KernelPluginFactory;
import com.microsoft.semantickernel.services.chatcompletion.ChatCompletionService;
import com.microsoft.semantickernel.services.chatcompletion.ChatHistory;
import com.azure.ai.openai.OpenAIAsyncClient;
import com.azure.ai.openai.OpenAIClientBuilder;
import com.azure.core.credential.KeyCredential;
import com.azure.core.http.HttpPipelineCallContext;
import com.azure.core.http.HttpPipelineNextPolicy;
import com.azure.core.http.HttpResponse;
import com.azure.core.http.policy.HttpPipelinePolicy;
import reactor.core.publisher.Mono;

import java.util.List;
import java.util.Scanner;

public class ExampleAgentNvidia {

    // NVIDIA NIM base URL — requests are redirected here from the OpenAI base URL
    // via an HTTP pipeline policy (Azure SDK forces Azure-mode paths when .endpoint() is set,
    // so we stay in non-Azure mode and rewrite at the HTTP layer instead).
    private static final String OPENAI_BASE  = "https://api.openai.com/v1";
    private static final String NVIDIA_BASE  = "https://integrate.api.nvidia.com/v1";
    // llama-3.1-70b supports OpenAI-compatible tool_calls; the 8b model does not.
    private static final String DEFAULT_NVIDIA_MODEL = "meta/llama-3.1-70b-instruct";

    // ── Data model ────────────────────────────────────────────────────────────
    public static class Memory {
        @GoodMemKey
        public String id;

        @GoodMemData                   // becomes originalContent
        public String content;

        @GoodMemData("source")         // stored as metadata["source"]
        public String source;
    }

    public static void main(String[] args) throws InterruptedException {
        for (var v : List.of("GOODMEM_API_KEY", "GOODMEM_BASE_URL", "NVIDIA_API_KEY")) {
            if (System.getenv(v) == null || System.getenv(v).isBlank())
                throw new IllegalStateException("Set " + v + " before running this example.");
        }

        var model = System.getenv("NVIDIA_MODEL");
        if (model == null || model.isBlank()) model = DEFAULT_NVIDIA_MODEL;

        // ── 1. Set up GoodMem collection with seed data ───────────────────────
        var collection = GoodMemCollection.of("agent-memory", Memory.class);

        collection.ensureCollectionDeleted().block();
        collection.ensureCollectionExists().block();
        collection.upsertAll(List.of(
                memory("The Pacific Ocean is the largest ocean on Earth.", "geography"),
                memory("Python was created by Guido van Rossum and first released in 1991.", "technology"),
                memory("The speed of light is approximately 299,792 km/s.", "science"),
                memory("Shakespeare wrote Hamlet, Macbeth, and Romeo and Juliet.", "literature"),
                memory("Semantic Kernel is a Microsoft SDK for building AI agents.", "technology")
        )).blockLast();
        System.out.println("Seeded 5 memories into the 'agent-memory' GoodMem space.");

        // GoodMem's embedding pipeline is asynchronous — wait before searching.
        System.out.println("Waiting for embeddings...");
        Thread.sleep(3_000);

        // ── 2. Build the Semantic Kernel using NVIDIA's OpenAI-compatible NIM endpoint ──
        // Do NOT call .endpoint() — that forces Azure-mode URL patterns.
        // Instead, stay in non-Azure mode and rewrite api.openai.com → integrate.api.nvidia.com
        // via an HTTP pipeline policy injected into the Azure SDK client.
        OpenAIAsyncClient nvidiaClient = new OpenAIClientBuilder()
                .credential(new KeyCredential(System.getenv("NVIDIA_API_KEY")))
                .addPolicy(new NvidiaUrlRewritePolicy())
                .buildAsyncClient();

        ChatCompletionService chatService = OpenAIChatCompletion.builder()
                .withOpenAIAsyncClient(nvidiaClient)
                .withModelId(model)
                .build();

        // ── 3. Register GoodMemPlugin so the LLM can call memory.recall ───────
        var memoryPlugin = new GoodMemPlugin<>(collection, Memory.class, (content, proto) -> {
            Memory m = new Memory();
            m.content = content;
            m.source = "agent";
            return m;
        });

        var kernel = Kernel.builder()
                .withAIService(ChatCompletionService.class, chatService)
                .withPlugin(KernelPluginFactory.createFromObject(memoryPlugin, "memory"))
                .build();

        // ── 4. Interactive chat loop ───────────────────────────────────────────
        System.out.println("\nMemory agent ready (model: " + model + "). Type 'exit' to quit.\n");

        var history = new ChatHistory(
                "You are a helpful assistant with access to a long-term memory store. "
                + "Always search memory before answering factual questions. "
                + "Cite what you found in memory when it is relevant.");

        var invocationContext = InvocationContext.builder()
                .withToolCallBehavior(ToolCallBehavior.allowAllKernelFunctions(true))
                .build();

        var scanner = new Scanner(System.in);
        while (true) {
            System.out.print("You: ");
            var input = scanner.nextLine().trim();
            if (input.isEmpty() || "exit".equalsIgnoreCase(input)) break;

            history.addUserMessage(input);

            var response = chatService
                    .getChatMessageContentsAsync(history, kernel, invocationContext)
                    .block();

            if (response != null && !response.isEmpty()) {
                var reply = response.get(response.size() - 1).getContent();
                System.out.println("Agent: " + reply);
                history.addAssistantMessage(reply);
            }
        }

        collection.close();
        HttpResources.disposeLoopsAndConnections();
        Schedulers.shutdownNow();
    }

    private static Memory memory(String content, String source) {
        Memory m = new Memory();
        m.content = content;
        m.source = source;
        return m;
    }

    /** Rewrites api.openai.com requests to the NVIDIA NIM endpoint at the HTTP layer. */
    static class NvidiaUrlRewritePolicy implements HttpPipelinePolicy {
        @Override
        public Mono<HttpResponse> process(HttpPipelineCallContext context, HttpPipelineNextPolicy next) {
            var url = context.getHttpRequest().getUrl().toString();
            if (url.startsWith(OPENAI_BASE)) {
                context.getHttpRequest().setUrl(NVIDIA_BASE + url.substring(OPENAI_BASE.length()));
            }
            return next.process();
        }
    }
}
