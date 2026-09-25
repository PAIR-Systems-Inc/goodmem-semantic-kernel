using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Text.Json.Nodes;

namespace GoodMem.SemanticKernel.Tests;

/// <summary>
/// A local HTTP server on a real socket. It records the request line of every
/// request exactly as it arrived on the wire, and answers like a minimal
/// GoodMem server with one space named "notes".
/// </summary>
/// <remarks>
/// A raw socket rather than HttpListener or Kestrel, so nothing between the
/// connector and the recording normalises the path.
/// </remarks>
internal sealed class RecordingHttpServer : IDisposable
{
    internal const string DefaultSpaceId = "01a0d16b-bbcd-701c-bfb4-fa306021e078";
    internal const string AssignedMemoryId = "019cfd1d-5a1e-7a4b-9c3e-2f6a1b0c0e09";

    private readonly TcpListener _listener = new(IPAddress.Loopback, 0);
    private readonly CancellationTokenSource _stop = new();
    private readonly object _lock = new();
    private readonly List<RecordedRequest> _requests = [];

    internal sealed record RecordedRequest(string Method, string Target, string Body)
    {
        public override string ToString() => $"{Method} {Target}";
    }

    public RecordingHttpServer()
    {
        _listener.Start();
        _ = Task.Run(AcceptLoopAsync);
    }

    /// <summary>The space id the space listing reports for "notes".</summary>
    public string ListedSpaceId { get; set; } = DefaultSpaceId;

    public string Url => $"http://127.0.0.1:{((IPEndPoint)_listener.LocalEndpoint).Port}";

    public IReadOnlyList<RecordedRequest> Requests
    {
        get { lock (_lock) return _requests.ToList(); }
    }

    public string Describe() => "[" + string.Join(", ", Requests) + "]";

    public void Dispose()
    {
        _stop.Cancel();
        _listener.Stop();
    }

    private async Task AcceptLoopAsync()
    {
        while (!_stop.IsCancellationRequested)
        {
            TcpClient client;
            try { client = await _listener.AcceptTcpClientAsync(_stop.Token).ConfigureAwait(false); }
            catch { return; }
            _ = Task.Run(() => HandleAsync(client));
        }
    }

    private async Task HandleAsync(TcpClient client)
    {
        using var _ = client;
        var stream = client.GetStream();

        // Headers end at the first blank line.
        var lines = new List<string>();
        for (var line = await ReadLineAsync(stream); line is { Length: > 0 }; line = await ReadLineAsync(stream))
            lines.Add(line);
        if (lines.Count == 0) return;

        var requestLine = lines[0].Split(' ');
        string Header(string name) => lines
            .Where(l => l.StartsWith(name + ":", StringComparison.OrdinalIgnoreCase))
            .Select(l => l[(name.Length + 1)..].Trim())
            .FirstOrDefault() ?? "";

        // HttpClient sends JSON content chunked, with no Content-Length.
        var body = new List<byte>();
        if (Header("transfer-encoding").Equals("chunked", StringComparison.OrdinalIgnoreCase))
        {
            for (var size = Convert.ToInt32(await ReadLineAsync(stream), 16); size > 0;
                 size = Convert.ToInt32(await ReadLineAsync(stream), 16))
            {
                body.AddRange(await ReadExactlyAsync(stream, size));
                await ReadLineAsync(stream);
            }
            await ReadLineAsync(stream);
        }
        else if (int.TryParse(Header("content-length"), out var length))
        {
            body.AddRange(await ReadExactlyAsync(stream, length));
        }

        var request = new RecordedRequest(requestLine[0], requestLine[1], Encoding.UTF8.GetString(body.ToArray()));
        lock (_lock) _requests.Add(request);

        var (status, json, contentType) = Respond(request);
        var payload = Encoding.UTF8.GetBytes(json);
        var response =
            $"HTTP/1.1 {status} {(status < 300 ? "OK" : "Not Found")}\r\n" +
            $"Content-Type: {contentType}\r\n" +
            $"Content-Length: {payload.Length}\r\n" +
            "Connection: close\r\n\r\n";
        await stream.WriteAsync(Encoding.ASCII.GetBytes(response)).ConfigureAwait(false);
        await stream.WriteAsync(payload).ConfigureAwait(false);
    }

    private (int Status, string Body, string ContentType) Respond(RecordedRequest request)
    {
        var route = request.Target.Split('?')[0];
        const string Json = "application/json";

        if (request.Method == "GET" && route == "/v1/spaces")
        {
            var space = new JsonObject { ["name"] = "notes", ["spaceId"] = ListedSpaceId };
            return (200, new JsonObject { ["spaces"] = new JsonArray(space) }.ToJsonString(), Json);
        }
        if (request.Method == "POST" && route == "/v1/spaces")
            return (200, new JsonObject { ["spaceId"] = DefaultSpaceId, ["name"] = "notes" }.ToJsonString(), Json);
        if (request.Method == "POST" && route == "/v1/memories:batchGet")
            return (200, """{"results":[]}""", Json);
        if (request.Method == "POST" && route == "/v1/memories:retrieve")
            return (200, "", "application/x-ndjson");
        if (request.Method == "POST" && route == "/v1/memories")
        {
            var memoryId = JsonNode.Parse(request.Body)?["memoryId"]?.GetValue<string>() ?? AssignedMemoryId;
            return (200, new JsonObject { ["memoryId"] = memoryId }.ToJsonString(), Json);
        }
        if (request.Method == "DELETE")
            return (204, "", Json);
        return (404, """{"error":"no route"}""", Json);
    }

    private static async Task<string> ReadLineAsync(NetworkStream stream)
    {
        var bytes = new List<byte>();
        var one = new byte[1];
        while (await stream.ReadAsync(one).ConfigureAwait(false) == 1)
        {
            if (one[0] == '\n') break;
            if (one[0] != '\r') bytes.Add(one[0]);
        }
        return Encoding.Latin1.GetString(bytes.ToArray());
    }

    private static async Task<byte[]> ReadExactlyAsync(NetworkStream stream, int count)
    {
        var buffer = new byte[count];
        await stream.ReadExactlyAsync(buffer).ConfigureAwait(false);
        return buffer;
    }
}
