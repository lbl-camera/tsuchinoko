"""Tiled connection helper with Bearer token auth and NATS-based refresh.

Mirrors the pattern from LUCID's exporter (lucid.exporter.tiled_utils):
LUCID passes its Keycloak token via NATS, and Tsuchinoko uses it to
connect to Tiled without independent authentication.

When the token expires (HTTP 401), the auth handler requests a fresh
token from LUCID over NATS rather than touching Keycloak directly.
This avoids race conditions from multiple clients refreshing the same
session token.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import TYPE_CHECKING, Any

import httpx
from loguru import logger
from tiled.client import from_uri

if TYPE_CHECKING:
    from tsuchinoko.nats.client import NATSClient


class BearerAuth(httpx.Auth):
    """httpx.Auth with automatic token refresh via NATS.

    On a 401 response, requests a fresh token from LUCID at
    ``{lucid_prefix}.auth.token`` instead of retrying with the
    stale token.
    """

    def __init__(
        self,
        token: str,
        nats_client: NATSClient | None = None,
        lucid_prefix: str = "",
    ) -> None:
        self._token = token
        self._nats_client = nats_client
        self._lucid_prefix = lucid_prefix

    def sync_auth_flow(
        self, request: httpx.Request,
    ) -> Generator[httpx.Request, httpx.Response, None]:
        request.headers["Authorization"] = f"Bearer {self._token}"
        response = yield request

        if response.status_code == 401 and self._nats_client:
            new_token = self._refresh_token()
            if new_token:
                self._token = new_token
                request.headers["Authorization"] = f"Bearer {self._token}"
                yield request

    def _refresh_token(self) -> str | None:
        """Ask LUCID for a fresh Keycloak token via NATS."""
        subject = f"{self._lucid_prefix}.auth.token"
        try:
            reply = self._nats_client.request_threadsafe(subject, {}, timeout=5.0)
            token = reply.get("token")
            if token:
                logger.info("Refreshed Tiled auth token from LUCID")
                return token
            logger.warning(f"Token refresh reply missing 'token': {reply}")
        except Exception as e:
            logger.warning(f"Token refresh failed: {e}")
        return None


def connect_tiled(
    url: str,
    token: str | None = None,
    proxy_url: str | None = None,
    nats_client: NATSClient | None = None,
    lucid_prefix: str = "",
) -> Any:
    """Connect to a Tiled server, optionally with auth and proxy.

    Args:
        url: Tiled server URL.
        token: Optional Bearer token (from LUCID's Keycloak session).
        proxy_url: Optional proxy URL (e.g. ``socks5://localhost:1080``).
        nats_client: Optional NATSClient for automatic token refresh.
        lucid_prefix: LUCID's NATS prefix (needed for token refresh subject).

    Returns:
        Tiled client instance.
    """
    kwargs: dict[str, Any] = {}
    if token:
        kwargs["auth"] = BearerAuth(token, nats_client, lucid_prefix)

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
