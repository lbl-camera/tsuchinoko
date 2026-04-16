"""Tiled connection helper with Bearer token auth.

Mirrors the pattern from LUCID's exporter (lucid.exporter.tiled_utils):
LUCID passes its Keycloak token via NATS, and Tsuchinoko uses it to
connect to Tiled without independent authentication.
"""

from __future__ import annotations

from typing import Any

import httpx
from loguru import logger
from tiled.client import from_uri


class BearerAuth(httpx.Auth):
    """httpx.Auth that adds a static Bearer token header."""

    def __init__(self, token: str) -> None:
        self._token = token

    def sync_auth_flow(self, request: httpx.Request):
        request.headers["Authorization"] = f"Bearer {self._token}"
        yield request


def connect_tiled(
    url: str,
    token: str | None = None,
    proxy_url: str | None = None,
) -> Any:
    """Connect to a Tiled server, optionally with auth and proxy.

    Args:
        url: Tiled server URL.
        token: Optional Bearer token (from LUCID's Keycloak session).
        proxy_url: Optional proxy URL (e.g. ``socks5://localhost:1080``).

    Returns:
        Tiled client instance.
    """
    kwargs: dict[str, Any] = {}
    if token:
        kwargs["auth"] = BearerAuth(token)

    if not proxy_url:
        logger.info(f"Connecting to Tiled at {url}")
        return from_uri(url, **kwargs)

    # Route through proxy (same monkey-patch pattern as LUCID exporter)
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
