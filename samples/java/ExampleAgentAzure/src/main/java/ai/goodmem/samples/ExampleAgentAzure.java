package ai.goodmem.samples;

// ExampleAgentAzure — GoodMem collection wired into a Semantic Kernel agent (Azure OpenAI).
//
// Functionally identical to ExampleAgent, but uses Azure OpenAI as the LLM provider.
// The Azure SDK's OpenAIClientBuilder natively supports Azure endpoints when built
// with AzureKeyCredential — no URL-rewrite policy needed.
//
// Required environment variables:
//   GOODMEM_BASE_URL                   — GoodMem server URL  (default: http://localhost:8080)
//   GOODMEM_VERIFY_SSL                 — Set to 'false' for self-signed certs
//   GOODMEM_API_KEY                    — GoodMem API key
//   AZURE_OPENAI_API_KEY               — Azure OpenAI API key
//   AZURE_OPENAI_ENDPOINT              — Azure OpenAI endpoint (e.g. https://my-resource.openai.azure.com)
//   AZURE_OPENAI_CHAT_DEPLOYMENT_NAME  — Chat model deployment name (e.g. gpt-4o-mini)
//
// Run:
//   cd samples/java/ExampleAgentAzure
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
import com.azure.core.credential.AzureKeyCredential;

import java.util.List;
import java.util.Scanner;

public class ExampleAgentAzure {

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
        for (var v : List.of(
                "GOODMEM_API_KEY", "GOODMEM_BASE_URL",
                "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")) {
            if (System.getenv(v) == null || System.getenv(v).isBlank())
                throw new IllegalStateException("Set " + v + " before running this example.");
        }

        var deployment = System.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME");
        var endpoint   = System.getenv("AZURE_OPENAI_ENDPOINT");
        var apiKey     = System.getenv("AZURE_OPENAI_API_KEY");

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

        // ── 2. Build the Semantic Kernel with Azure OpenAI ────────────────────
        // AzureKeyCredential puts the SDK into Azure mode: it constructs
        // /openai/deployments/{deployment}/chat/completions?api-version=... paths natively.
        // AzureKeyCredential + .endpoint() puts the SDK into Azure mode natively.
        // API version is managed by the SDK default; override via the Azure portal if needed.
        OpenAIAsyncClient azureClient = new OpenAIClientBuilder()
                .endpoint(endpoint)
                .credential(new AzureKeyCredential(apiKey))
                .buildAsyncClient();

        ChatCompletionService chatService = OpenAIChatCompletion.builder()
                .withOpenAIAsyncClient(azureClient)
                .withModelId(deployment)
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
        System.out.println("\nMemory agent ready. Type 'exit' to quit.\n");

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
}
