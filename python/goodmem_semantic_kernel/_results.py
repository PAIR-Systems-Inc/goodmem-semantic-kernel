"""Retrieval event handling: statuses, chunk/memory joining, and scores.

The rules here are the retrieval status contract every GoodMem integration
follows; see ``goodmem-integration-fix-patterns.md``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from goodmem.models.good_mem_status import GoodMemStatus
from goodmem.models.retrieve_memory_event import RetrieveMemoryEvent

# Notices that carry no loss of results, by their code alone (contract Q1).
_INFORMATIONAL_CODES = frozenset({"LLM_CAPABILITY_INFERRED", "FEATURE_DISABLED"})


def is_informational(status: GoodMemStatus) -> bool:
    """True for notices that carry no loss of results.

    An unrecognized code is deliberately not informational: the SDK decodes
    codes it does not know as ``None``, and a future server status must be
    surfaced rather than assumed harmless.
    """
    return status.code is not None and status.code in _INFORMATIONAL_CODES


def classify(
    events: Sequence[RetrieveMemoryEvent],
) -> tuple[list[dict[str, Any]], bool]:
    """Split statuses into what to report and whether results are incomplete.

    Returns ``(surfaced, degraded)``. Unknown codes surface as ``UNKNOWN`` and
    mark the result degraded, but never discard chunks and never raise
    (contract Q3).
    """
    surfaced: list[dict[str, Any]] = []
    degraded = False
    for event in events:
        status = event.status
        if status is None or is_informational(status):
            continue
        entry = status.model_dump(exclude_none=True)
        if status.code is None:
            entry["code"] = "UNKNOWN"
            entry["unrecognized"] = True
        surfaced.append(entry)
        degraded = True
    return surfaced, degraded


def hits_from_events(
    events: Iterable[RetrieveMemoryEvent],
    *,
    reranked: bool,
) -> list[dict[str, Any]]:
    """Join chunks to their memory definitions **by UUID**, ignoring event order.

    The previous implementation joined on ``memoryIndex``, the chunk's
    positional index into the memory definitions in arrival order. That is
    correct only while the server emits definitions in exactly that order;
    a UUID join cannot be wrong.

    Deduplicates by ``chunk_id``: two chunks of one memory are two results.
    """
    events = list(events)
    memories = {
        event.memory_definition.memory_id: event.memory_definition
        for event in events
        if event.memory_definition is not None
    }

    hits: list[dict[str, Any]] = []
    seen: set[str] = set()
    for event in events:
        item = event.retrieved_item
        if item is None or item.chunk is None:
            continue
        reference = item.chunk
        chunk = reference.chunk
        if chunk is None or not chunk.chunk_text:
            continue
        if chunk.chunk_id in seen:
            continue
        seen.add(chunk.chunk_id)

        memory = memories.get(chunk.memory_id) or item.memory
        hits.append(
            {
                "chunk_id": chunk.chunk_id,
                "chunk_text": chunk.chunk_text,
                "memory_id": chunk.memory_id,
                "space_id": getattr(memory, "space_id", None),
                "metadata": dict(getattr(memory, "metadata", None) or {}),
                "raw_score": reference.relevance_score,
                "score_kind": "reranker" if reranked else "vector",
            }
        )
    return hits


def score_for_host(hit: dict[str, Any]) -> float | None:
    """Return the score under Semantic Kernel's higher-is-better convention.

    GoodMem vector scores are pgvector negative inner products: the best match
    is the lowest number, so they are negated. Reranker scores are already
    higher-is-better and are passed through untouched — negating one would
    reverse the ranking. ``raw_score`` keeps the server's value either way.
    """
    raw = hit.get("raw_score")
    if raw is None:
        return None
    return float(raw) if hit.get("score_kind") == "reranker" else -float(raw)
