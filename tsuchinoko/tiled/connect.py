"""Tiled connection helper with API-key auth.

Lightfall Auth v2 mints session-lifetime API keys at user login and forwards
the secret to tsuchinoko via the NATS bind_run payload. The executor
uses the secret directly (no refresh, no Keycloak dependency); if the
key expires mid-job the executor fails the job and Lightfall re-mints on
next login.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

import httpx
from loguru import logger
from tiled.client import from_uri


class ApiKeyAuth(httpx.Auth):
    """httpx.Auth that sends a static Tiled API key on every request."""

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def sync_auth_flow(
        self, request: httpx.Request,
    ) -> Generator[httpx.Request, httpx.Response, None]:
        request.headers["Authorization"] = f"Apikey {self._api_key}"
        yield request


def connect_tiled(
    url: str,
    api_key: str | None = None,
    proxy_url: str | None = None,
) -> Any:
    """Connect to a Tiled server, optionally with API-key auth and proxy.

    Args:
        url: Tiled server URL.
        api_key: Optional Tiled API key (from Lightfall's session-key cache).
        proxy_url: Optional proxy URL (e.g. ``socks5://localhost:1080``).

    Returns:
        Tiled client instance.
    """
    kwargs: dict[str, Any] = {}
    if api_key:
        kwargs["auth"] = ApiKeyAuth(api_key)

    if not proxy_url:
        logger.info(f"Connecting to Tiled at {url}")
        return from_uri(url, **kwargs)

    # Route through proxy (same monkey-patch pattern as Lightfall exporter)
    import tiled.client.context as context_mod
    from tiled.client.transport import Transport as OriginalTransport

    proxy_transport = httpx.HTTPTransport(proxy=proxy_url)

    original_httpx_get = context_mod.httpx.get

    def proxy_httpx_get(u, **kw):
        with httpx.Client(proxy=proxy_url, timeout=15.0) as client:
            return client.get(u, **kw)

    context_mod.httpx.get = proxy_httpx_get

    class ProxyTransport(OriginalTransport):
        def __init__(self, *, transport=None, **kw):
            super().__init__(transport=proxy_transport, **kw)

    original_transport_cls = context_mod.Transport
    context_mod.Transport = ProxyTransport

    try:
        logger.info(f"Connecting to Tiled at {url} via proxy {proxy_url}")
        return from_uri(url, **kwargs)
    finally:
        context_mod.Transport = original_transport_cls
        context_mod.httpx.get = original_httpx_get
