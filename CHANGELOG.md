# Changelog

## 0.3.0

Audit release. Every defect below was reproduced against the `main` tree at
`108b8f3` on a live GoodMem server (v1.0.320) before it was fixed.

### Fixed — Python

- **Reads never returned anything.** `get()` parsed `memories` from the batch
  response, but the endpoint answers with `{"results": [{"success", "memory"}]}`.
  Records that existed on the server came back as `None`. It also now asks for
  the content, which the endpoint omits by default, and returns records in the
  order they were requested.
- **A failed update destroyed the record.** GoodMem has no update endpoint, so
  upsert deletes and re-creates. The delete went first unconditionally, so a
  create that failed afterwards left nothing behind — reproduced with a record
  whose content field was empty, which the server rejects with HTTP 400. The
  current version is now read before the delete and written back if the create
  fails; `GoodMemUpsertError` reports whether the restore succeeded.
- **Filtering was unavailable.** `search(filter=...)` raised
  `VectorStoreOperationNotSupportedException`. Filters are now translated into
  GoodMem filter expressions and evaluated server-side, with the quoting and
  casts the server actually accepts (verified live), so a value containing an
  apostrophe is a value and an injection payload matches only its own row.
- **Retrieval statuses were discarded.** A problem the server reported was
  dropped, so a half-failed search was indistinguishable from a complete one.
  Statuses now reach `KernelSearchResults.metadata` as `goodmem_partial` and
  `goodmem_statuses`, an unrecognised code surfaces as `UNKNOWN` rather than
  vanishing, and a problem with no results warns instead of looking successful.
- **A truncated response threw away the whole search.** Events received before
  the break are kept and the break is reported as `MALFORMED_STREAM`.
- **Spaces were reused without checking the embedder,** and an unconfigured
  embedder was silently resolved to whichever one the server listed first. The
  embedder decides how everything in a space is indexed and cannot be changed
  afterwards, so both are now errors that say what to set.
- **Chunks were joined to memories by arrival position.** Now joined by UUID,
  which cannot attach one memory's metadata to another's chunk.
- **Server error messages were lost** behind `raise_for_status()`.

### Changed — Python

- Built on the official `goodmem` SDK instead of a hand-written HTTP client.
- `upsert` waits for indexing by default, so a search straight after a write
  can find what was written. Turn it off with `GOODMEM_WAIT_FOR_INDEXING=false`.
- New settings: `reranker_id`, `timeout`, `wait_for_indexing`,
  `indexing_timeout`. A reranker score is passed through without negation.
- Tests: 36 offline (driving the real SDK over a mock transport, with event
  shapes captured from a live server) and 13 live. `ruff`, `mypy` and CI added
  — the repository had no CI at all.

### Fixed — .NET and Java

- The same delete-then-create data loss, with the same fix: the previous
  version is read first and restored if the write fails, and
  `GoodMemUpsertException` says whether the restore succeeded.
- .NET tests: 47 (was 44). Java tests: 23 (was 18).

### Not changed

- .NET and Java keep their own HTTP clients; only the upsert path was touched.
- Both already parsed the batch envelope correctly — Python was the outlier.
