using Microsoft.Extensions.Logging;

namespace GoodMem.SemanticKernel.Tests;

/// <summary>Records every log call made through the loggers it creates.</summary>
internal sealed class CapturingLoggerFactory : ILoggerFactory
{
    internal sealed record Entry(LogLevel Level, string Category, string Message);

    private readonly List<Entry> _entries = [];

    public IReadOnlyList<Entry> Entries
    {
        get { lock (_entries) return _entries.ToList(); }
    }

    public ILogger CreateLogger(string categoryName) => new Logger(this, categoryName);

    public void AddProvider(ILoggerProvider provider) { }

    public void Dispose() { }

    private sealed class Logger(CapturingLoggerFactory owner, string category) : ILogger
    {
        public IDisposable? BeginScope<TState>(TState state) where TState : notnull => null;

        public bool IsEnabled(LogLevel logLevel) => true;

        public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception? exception,
            Func<TState, Exception?, string> formatter)
        {
            lock (owner._entries)
                owner._entries.Add(new Entry(logLevel, category, formatter(state, exception)));
        }
    }
}
