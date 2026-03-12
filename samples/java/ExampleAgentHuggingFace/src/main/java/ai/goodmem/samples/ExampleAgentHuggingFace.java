package ai.goodmem.samples;

// ExampleAgentHuggingFace — GoodMem collection wired into a Semantic Kernel agent (Hugging Face).
//
// The agent has a memory search tool backed by GoodMem. When the user asks a
// question, the LLM decides whether to call the tool to look up relevant
// memories before composing its answer.
//
// Note: SK's built-in HuggingFace connector does not support chat completion
// with function calling. This sample uses HuggingFace's OpenAI-compatible
// Inference API endpoint instead, via the SK OpenAI connector with a URL-rewrite
// policy. Not all HuggingFace models support tool calling — instruction-tuned
// models work best (e.g. Llama-3.1-8B-Instruct).
//
// Required environment variables:
//   GOODMEM_BASE_URL   — GoodMem server URL  (default: http://localhost:8080)
//   GOODMEM_VERIFY_SSL — Set to 'false' for self-signed certs
//   GOODMEM_API_KEY    — GoodMem API key
//   HF_TOKEN           — Hugging Face API token (https://huggingface.co/settings/tokens)
//
// Optional environment variables:
//   HF_MODEL           — Hugging Face model ID (default: meta-llama/Llama-3.1-8B-Instruct)
//
// Run:
//   cd samples/java/ExampleAgentHuggingFace
//   mvn install -f ../../../java/pom.xml -DskipTests
//   mvn compile exec:java

import ai.goodmem.semantickernel.GoodMemCollection;
import ai.goodmem.semantickernel.GoodMemData;
import ai.goodmem.semantickernel.GoodMemKey;
import ai.goodmem.semantickernel.GoodMemPlugin;
import reactor.core.publisher.Mono;
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

import java.util.List;
import java.util.Scanner;

public class ExampleAgentHuggingFace {

    // HuggingFace Inference API uses an OpenAI-compatible endpoint.
    // Azure SDK forces Azure-mode paths when .endpoint() is set, so we stay in
    // non-Azure mode and rewrite api.openai.com → router.huggingface.co at the HTTP layer.
    private static final String OPENAI_BASE = "https://api.openai.com/v1";
    private static final String HF_BASE     = "https://router.huggingface.co/v1";
    private static final String DEFAULT_HF_MODEL = "meta-llama/Llama-3.1-8B-Instruct";

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
        for (var v : List.of("GOODMEM_API_KEY", "GOODMEM_BASE_URL", "HF_TOKEN")) {
            if (System.getenv(v) == null || System.getenv(v).isBlank())
                throw new IllegalStateException("Set " + v + " before running this example.");
        }

        var model = System.getenv("HF_MODEL");
        if (model == null || model.isBlank()) model = DEFAULT_HF_MODEL;

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

        System.out.println("Waiting for embeddings...");
        Thread.sleep(3_000);

        // ── 2. Build the Semantic Kernel using HF's OpenAI-compatible Inference API ──
        OpenAIAsyncClient hfClient = new OpenAIClientBuilder()
                .credential(new KeyCredential(System.getenv("HF_TOKEN")))
                .addPolicy(new HuggingFaceUrlRewritePolicy())
                .buildAsyncClient();

        ChatCompletionService chatService = OpenAIChatCompletion.builder()
                .withOpenAIAsyncClient(hfClient)
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

    /** Rewrites api.openai.com requests to the HuggingFace Inference API at the HTTP layer. */
    static class HuggingFaceUrlRewritePolicy implements HttpPipelinePolicy {
        @Override
        public Mono<HttpResponse> process(HttpPipelineCallContext context, HttpPipelineNextPolicy next) {
            var url = context.getHttpRequest().getUrl().toString();
            if (url.startsWith(OPENAI_BASE)) {
                context.getHttpRequest().setUrl(HF_BASE + url.substring(OPENAI_BASE.length()));
            }
            return next.process();
        }
    }
}
