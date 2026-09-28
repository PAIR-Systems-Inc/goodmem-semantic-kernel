using Microsoft.Extensions.Logging;

namespace GoodMem.SemanticKernel;

/// <summary>
/// The logger used when <see cref="GoodMemOptions.LoggerFactory"/> is not set: warnings and
/// errors go to standard error, so a search the server reported a problem with is visible
/// without any logging set up. Anything below <see cref="LogLevel.Warning"/> is dropped.
/// </summary>
internal sealed class GoodMemStandardErrorLogger : ILogger
{
    private readonly string _category;
    private readonly Func<TextWriter> _writer;

    internal GoodMemStandardErrorLogger(string category, Func<TextWriter>? writer = null)
    {
        _category = category;
        _writer = writer ?? (() => Console.Error);
    }

    public IDisposable? BeginScope<TState>(TState state) where TState : notnull => null;

    public bool IsEnabled(LogLevel logLevel) => logLevel >= LogLevel.Warning && logLevel != LogLevel.None;

    public void Log<TState>(
        LogLevel logLevel,
        EventId eventId,
        TState state,
        Exception? exception,
        Func<TState, Exception?, string> formatter)
    {
        if (!IsEnabled(logLevel)) return;
        var label = logLevel switch
        {
            LogLevel.Warning => "warn",
            LogLevel.Error => "fail",
            _ => "crit",
        };
        _writer().WriteLine($"{label}: {_category}: {formatter(state, exception)}");
    }
}
