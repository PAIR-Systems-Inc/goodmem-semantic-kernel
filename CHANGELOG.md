# Changelog

## 0.3.1

Security release. Python goes to 0.3.1 and Java to `0.1.1-SNAPSHOT`. The .NET
project file declares no version, so it has nothing to bump.

### Fixed — Python, .NET and Java

- **A record key could make a request hit a different resource.** Keys are
  GoodMem memory ids, and the connector puts them in URL paths. In Python the
  SDK builds `f"/v1/memories/{id}"` and httpx resolves dot segments, so
  `collection.delete("../spaces/<uuid>")` sent `DELETE /v1/spaces/<uuid>`,
  which deletes a whole space, and then returned normally. Upserting a record
  keyed `"../spaces/<uuid>"` sent `GET /v1/spaces/<uuid>`. .NET and Java
  percent-encode the key, but a bare `..` still got through: .NET sent
  `DELETE /v1/`, Java sent `DELETE /v1/memories/..`, and Python sent
  `DELETE /v1`. Encoding is not a guard in any case, because the GoodMem server
  has been seen decoding `%2e%2e` back into `..`. Every one of these was
  recorded by a local HTTP server before the fix.
- Every id that can reach a URL path now goes through one validator per
  implementation before any request is made: `_ids.require_uuid` (Python),
  `GoodMemIds.RequireUuid` (.NET) and `GoodMemIds.requireUuid` (Java). The id
  must be a canonical 8-4-4-4-12 hex UUID and is lowercased. Anything else is
  refused with an error that names the field. That covers `""`, surrounding
  whitespace, a `?` or `#` suffix and a trailing newline. Python raises
  `ValueError`, which Semantic Kernel re-raises as
  `VectorStoreOperationException`. .NET raises `ArgumentException` and Java
  raises `IllegalArgumentException`.
- What is covered: delete (single and batch), upsert (single and batch),
  get (the ids go in the request body, but are checked the same way), the
  configured embedder id (and in Python the reranker id), and the space id
  that `ensure_collection_deleted` reads from the server's space listing
  before deleting that space. Upsert checks the key before it looks up the
  space. A batch is checked in full before the first request, so one bad key
  no longer leaves a batch half deleted or half written. In 0.3.0 all three
  deleted the valid key before they reached the bad one.

### Changed

- An empty-string key is now refused. Before this release, Python treated it
  as "no key" and let the server assign one, while .NET and Java sent it as
  `memoryId: ""`. To let the server assign a key, pass `None` (Python) or
  `null` (.NET, Java).
- A configured embedder id that is not a UUID now fails before any request.
  In Python this is a `VectorStoreInitializationException` that names
  `GOODMEM_EMBEDDER_ID`. An empty setting still means "not configured".

### Tests

- The new regression tests drive the real stack over TCP to a local server
  that records every request line. For every entry point listed above, they
  check that each of 11 malicious payloads is refused with zero requests
  recorded, and that a valid UUID reaches exactly the intended path. Against
  0.3.0 they fail: 89 of 94 in Python, 98 of 103 in .NET and 98 of 103 in
  Java. The ones that pass are the valid-UUID controls.
- Existing tests that used ids like `"m-1"` now use real UUIDs.
- Python: 130 offline (was 36) and 13 live. .NET: 167 (was 47). Java: 143
  (was 23).

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
