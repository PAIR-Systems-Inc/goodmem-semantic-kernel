"""SDK client ownership for the Semantic Kernel connector.

The connector talks to GoodMem through the official ``goodmem`` SDK's async
client. A caller may inject their own ``AsyncGoodmem``; an injected client
keeps its own server, credentials and TLS settings, and is never closed here.
"""

from __future__ import annotations

from typing import Any, cast

from goodmem import AsyncGoodmem

from goodmem_semantic_kernel._typing import AsyncGoodmemClient
from goodmem_semantic_kernel.settings import GoodMemSettings


class GoodMemConnection:
    """Owns an ``AsyncGoodmem`` client, or borrows a caller-supplied one."""

    def __init__(
        self,
        settings: GoodMemSettings | None = None,
        client: AsyncGoodmem | None = None,
    ) -> None:
        self._settings = settings or GoodMemSettings()  # type: ignore[call-arg]
        self._injected = client
        self._client: AsyncGoodmem | None = client

    @property
    def owns_client(self) -> bool:
        """True when this object created the client and must close it."""
        return self._injected is None

    @property
    def settings(self) -> GoodMemSettings:
        return self._settings

    @property
    def client(self) -> AsyncGoodmemClient:
        """Return the SDK client, creating one on first use."""
        if self._client is None:
            self._client = AsyncGoodmem(
                base_url=self._settings.base_url,
                api_key=self._settings.api_key.get_secret_value(),
                verify=self._settings.verify_ssl,
                timeout=self._settings.timeout,
            )
        return cast(AsyncGoodmemClient, self._client)

    async def aclose(self) -> None:
        """Close the client only if we created it."""
        if self._injected is None and self._client is not None:
            await self._client.close()
            self._client = None

    async def __aenter__(self) -> GoodMemConnection:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.aclose()
