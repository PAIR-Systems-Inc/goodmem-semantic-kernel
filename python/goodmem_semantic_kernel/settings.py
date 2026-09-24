"""GoodMem settings for the Semantic Kernel connector."""

from typing import ClassVar

from pydantic import SecretStr
from semantic_kernel.kernel_pydantic import KernelBaseSettings


class GoodMemSettings(KernelBaseSettings):
    """Settings for the GoodMem Semantic Kernel connector.

    All fields can be configured via environment variables with the
    ``GOODMEM_`` prefix. For example, ``GOODMEM_BASE_URL`` sets ``base_url``.

    Priority (highest to lowest):
    1. Constructor keyword arguments
    2. Environment variables (``GOODMEM_*``)
    3. ``.env`` file
    4. Field defaults

    Attributes:
        base_url: Base URL of the GoodMem server (without trailing slash).
        api_key: API key for authentication (sent as ``x-api-key`` header).
        embedder_id: Embedder UUID used when creating a space. When unset, the
            connector will not invent one: creating a collection raises and
            names this setting, because the embedder decides how every memory
            in the space is indexed and is not changeable afterwards.
        reranker_id: Optional reranker UUID applied to searches.
        verify_ssl: Whether to verify TLS certificates (default ``True``).
            Set to ``False`` for local servers with self-signed certificates
            (``GOODMEM_VERIFY_SSL=false``).
        timeout: Per-request timeout in seconds.
        wait_for_indexing: Whether ``upsert`` waits for each written memory to
            finish indexing before returning. GoodMem indexes asynchronously,
            so without this a search immediately after a write can miss it.
        indexing_timeout: How long ``upsert`` waits when
            ``wait_for_indexing`` is set.
    """

    env_prefix: ClassVar[str] = "GOODMEM_"

    base_url: str = "http://localhost:8080"
    api_key: SecretStr
    embedder_id: str | None = None
    reranker_id: str | None = None
    verify_ssl: bool = True
    timeout: float = 30.0
    wait_for_indexing: bool = True
    indexing_timeout: float = 60.0
