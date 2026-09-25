package ai.goodmem.semantickernel;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * A local HTTP server on a real socket. It records the request line of every
 * request exactly as it arrived on the wire, and answers like a minimal GoodMem
 * server with one space named "notes".
 *
 * <p>A raw socket rather than WireMock or the JDK server, so nothing between the
 * connector and the recording normalises or rejects the path.
 */
final class RecordingHttpServer implements AutoCloseable {

    static final String DEFAULT_SPACE_ID = "01a0d16b-bbcd-701c-bfb4-fa306021e078";
    static final String ASSIGNED_MEMORY_ID = "019cfd1d-5a1e-7a4b-9c3e-2f6a1b0c0e09";

    private static final ObjectMapper MAPPER = new ObjectMapper();

    record Recorded(String method, String target, String body) {
        @Override
        public String toString() {
            return method + " " + target;
        }
    }

    private final ServerSocket socket;
    private final List<Recorded> requests = new ArrayList<>();
    private volatile String listedSpaceId = DEFAULT_SPACE_ID;

    RecordingHttpServer() throws IOException {
        socket = new ServerSocket(0, 50, InetAddress.getLoopbackAddress());
        Thread accept = new Thread(this::acceptLoop, "recording-http-server");
        accept.setDaemon(true);
        accept.start();
    }

    String url() {
        return "http://127.0.0.1:" + socket.getLocalPort();
    }

    /** The space id the space listing reports for "notes". */
    void listedSpaceId(String id) {
        listedSpaceId = id;
    }

    synchronized List<Recorded> requests() {
        return List.copyOf(requests);
    }

    @Override
    public void close() {
        try { socket.close(); } catch (IOException ignored) {}
    }

    private void acceptLoop() {
        while (!socket.isClosed()) {
            try {
                Socket client = socket.accept();
                Thread handler = new Thread(() -> handle(client));
                handler.setDaemon(true);
                handler.start();
            } catch (IOException closed) {
                return;
            }
        }
    }

    private void handle(Socket client) {
        try (client) {
            InputStream in = client.getInputStream();
            List<String> head = new ArrayList<>();
            for (String line = readLine(in); !line.isEmpty(); line = readLine(in)) head.add(line);
            if (head.isEmpty()) return;

            String[] requestLine = head.get(0).split(" ");
            String length = header(head, "content-length");
            byte[] body = new byte[0];
            if ("chunked".equalsIgnoreCase(header(head, "transfer-encoding"))) {
                ByteArrayOutputStream chunks = new ByteArrayOutputStream();
                for (int size = Integer.parseInt(readLine(in), 16); size > 0;
                     size = Integer.parseInt(readLine(in), 16)) {
                    chunks.write(in.readNBytes(size));
                    readLine(in);
                }
                readLine(in);
                body = chunks.toByteArray();
            } else if (!length.isEmpty()) {
                body = in.readNBytes(Integer.parseInt(length));
            }

            Recorded request = new Recorded(
                    requestLine[0], requestLine[1], new String(body, StandardCharsets.UTF_8));
            synchronized (this) { requests.add(request); }

            Map.Entry<Integer, String> response = respond(request);
            byte[] payload = response.getValue().getBytes(StandardCharsets.UTF_8);
            OutputStream out = client.getOutputStream();
            out.write(("HTTP/1.1 " + response.getKey() + " X\r\n"
                    + "Content-Type: application/json\r\n"
                    + "Content-Length: " + payload.length + "\r\n"
                    + "Connection: close\r\n\r\n").getBytes(StandardCharsets.US_ASCII));
            out.write(payload);
            out.flush();
        } catch (IOException ignored) {
            // The client went away; nothing to record.
        }
    }

    private Map.Entry<Integer, String> respond(Recorded request) throws IOException {
        String route = request.target().split("\\?", 2)[0];
        String method = request.method();

        if (method.equals("GET") && route.equals("/v1/spaces")) {
            var listing = MAPPER.createObjectNode();
            listing.putArray("spaces").addObject().put("name", "notes").put("spaceId", listedSpaceId);
            return Map.entry(200, MAPPER.writeValueAsString(listing));
        }
        if (method.equals("POST") && route.equals("/v1/spaces"))
            return Map.entry(200, "{\"spaceId\":\"" + DEFAULT_SPACE_ID + "\",\"name\":\"notes\"}");
        if (method.equals("POST") && route.equals("/v1/memories:batchGet"))
            return Map.entry(200, "{\"results\":[]}");
        if (method.equals("POST") && route.equals("/v1/memories:retrieve"))
            return Map.entry(200, "");
        if (method.equals("POST") && route.equals("/v1/memories")) {
            JsonNode body = MAPPER.readTree(request.body());
            String id = body.path("memoryId").asText(ASSIGNED_MEMORY_ID);
            return Map.entry(200, MAPPER.writeValueAsString(MAPPER.createObjectNode().put("memoryId", id)));
        }
        if (method.equals("DELETE"))
            return Map.entry(204, "");
        return Map.entry(404, "{\"error\":\"no route\"}");
    }

    private static String header(List<String> head, String name) {
        for (String line : head) {
            int colon = line.indexOf(':');
            if (colon > 0 && line.substring(0, colon).trim().equalsIgnoreCase(name))
                return line.substring(colon + 1).trim();
        }
        return "";
    }

    private static String readLine(InputStream in) throws IOException {
        ByteArrayOutputStream line = new ByteArrayOutputStream();
        for (int b = in.read(); b != -1 && b != '\n'; b = in.read()) {
            if (b != '\r') line.write(b);
        }
        return line.toString(StandardCharsets.ISO_8859_1);
    }
}
