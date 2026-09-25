"""The one check every GoodMem id passes before the connector sends it.

The SDK puts an id straight into the request path (``f"/v1/memories/{id}"``)
and httpx resolves dot segments before sending, so a key of
``"../spaces/<uuid>"`` turns a memory delete into ``DELETE /v1/spaces/<uuid>``.
Escaping cannot be relied on either: the server decodes ``%2e%2e`` back into
``..``. GoodMem ids (memories, spaces, embedders, rerankers) are all UUIDs, so
anything that is not one is refused before a request is made with it.
"""

from __future__ import annotations

import re
import uuid

# fullmatch, not ^...$: in Python, $ also matches before a trailing newline.
_UUID = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")


def require_uuid(value: object, field: str) -> str:
    """Return ``value`` as a lowercase canonical UUID, or raise ``ValueError``.

    ``field`` names the argument or setting in the error, so the caller can
    tell which id was wrong.
    """
    text = str(value) if isinstance(value, uuid.UUID) else value
    if isinstance(text, str) and _UUID.fullmatch(text):
        return text.lower()
    shown = repr(value)
    if len(shown) > 80:
        shown = shown[:77] + "..."
    raise ValueError(
        f"{field} must be a GoodMem UUID, got {shown}. It was not sent: GoodMem "
        "ids are UUIDs, and any other value can change which URL a request goes to."
    )
