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
        embedder_id: Optional embedder UUID to use when creating spaces.
            If None, the first available embedder is used automatically.
        verify_ssl: Whether to verify TLS certificates (default ``True``).
            Set to ``False`` for local servers with self-signed certificates
            (``GOODMEM_VERIFY_SSL=false``).
    """

    env_prefix: ClassVar[str] = "GOODMEM_"

    base_url: str = "http://localhost:8080"
    api_key: SecretStr
    embedder_id: str | None = None
    verify_ssl: bool = True
