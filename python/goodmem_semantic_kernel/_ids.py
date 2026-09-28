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
from typing import cast

# fullmatch, not ^...$: in Python, $ also matches before a trailing newline.
_UUID = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")


def canonical_uuid(value: object) -> str | None:
    """Return ``value`` as a plain lowercase UUID string, or ``None``.

    What comes back is always a new, exact ``str``, never ``value`` itself. A
    ``str`` subclass can override ``lower()``, ``__str__`` or ``__format__``
    and hand the SDK a different string from the one that was checked, so no
    method of ``value`` is called after the check.
    """
    # type(), not isinstance(): isinstance() believes an object that fakes
    # its __class__.
    if issubclass(type(value), uuid.UUID):
        try:
            value = str(value)
        except Exception:
            return None
    if not issubclass(type(value), str):
        return None
    text = cast(str, value)
    if _UUID.fullmatch(text) is None:
        return None
    # str.lower, not text.lower(): the base method, which returns an exact str.
    return str.lower(text)


def describe(value: object) -> str:
    """Show an id in an error message, shortened so a huge one stays readable."""
    if issubclass(type(value), str):
        shown = repr(str.__str__(cast(str, value)))
    else:
        try:
            shown = str.__str__(repr(value))
        except Exception:
            shown = f"a {type(value).__name__}"
    return shown if len(shown) <= 80 else shown[:77] + "..."


def require_uuid(value: object, field: str) -> str:
    """Return ``value`` as a lowercase canonical UUID, or raise ``ValueError``.

    ``field`` names the argument or setting in the error, so the caller can
    tell which id was wrong.
    """
    canonical = canonical_uuid(value)
    if canonical is not None:
        return canonical
    raise ValueError(
        f"{field} must be a GoodMem UUID, got {describe(value)}. It was not sent: GoodMem "
        "ids are UUIDs, and any other value can change which URL a request goes to."
    )
