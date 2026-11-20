"""Tiled connection helper with API-key auth.

Lightfall Auth v2 mints session-lifetime API keys at user login and forwards
the secret to tsuchinoko via the NATS bind_run payload. The executor
uses the secret directly (no refresh, no Keycloak dependency); if the
key expires mid-job the executor fails the job and Lightfall re-mints on
next login.

httpx flavour
-------------
tiled>=0.2.21 is built on ``httpx2`` (``tiled.client.context`` does
``import httpx2 as httpx``); older releases use ``httpx``. Tiled's
``Context`` rejects an ``auth`` that is not an instance of *its* ``Auth``
base, and anything injected into its client (transports, the module-level
``get`` it uses to probe the server) must be the same flavour. Everything
handed to tiled here is therefore built from :func:`tiled_httpx`, never from
a top-level ``import httpx``.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

from loguru import logger
from tiled.client import from_uri

# Keyword arguments of the module-level ``httpx.get`` that configure the
# *client* rather than the request; ``Client.get`` rejects them (tiled>=0.2.21
# probes with ``httpx.get(uri, timeout=..., trust_env=...)``).
_CLIENT_KWARGS = ("trust_env", "verify", "cert", "timeout", "follow_redirects")
_PROBE_TIMEOUT = 15.0


def tiled_httpx() -> Any:
    """The httpx module the installed tiled client uses (``httpx2`` or ``httpx``)."""
    import tiled.client.context as context_mod

    return context_mod.httpx


class _ApiKeyAuthFlow:
    """Adds ``Authorization: Apikey <key>`` to every request; no refresh."""

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def __repr__(self) -> str:  # never expose the key in logs or tracebacks
        return f"{type(self).__name__}(api_key=<redacted>)"

    def auth_flow(self, request: Any) -> Generator[Any, Any, None]:
        request.headers["Authorization"] = f"Apikey {self._api_key}"
        yield request


_AUTH_CLASSES: dict[type, type] = {}


def api_key_auth_class() -> type:
    """``ApiKeyAuth`` built on the ``Auth`` base of tiled's httpx flavour."""
    base = tiled_httpx().Auth
    cls = _AUTH_CLASSES.get(base)
    if cls is None:
        cls = _AUTH_CLASSES[base] = type(
            "ApiKeyAuth",
            (_ApiKeyAuthFlow, base),
            {"__doc__": _ApiKeyAuthFlow.__doc__, "__module__": __name__},
        )
    return cls


def ApiKeyAuth(api_key: str) -> Any:  # noqa: N802 - kept callable under its old name
    """A static Tiled API-key auth that the installed tiled accepts."""
    return api_key_auth_class()(api_key)


def make_proxy_transport(proxy_url: str, **kwargs: Any) -> Any:
    """A transport of tiled's httpx flavour that routes through ``proxy_url``.

    ``socks5://`` needs ``socksio`` (``httpx[socks]`` / ``httpx2[socks]``).
    """
    return tiled_httpx().HTTPTransport(proxy=proxy_url, **kwargs)


def _install_proxy(proxy_url: str) -> dict[str, Any]:
    """Route tiled's server probe and client Transport through ``proxy_url``."""
    import tiled.client.context as context_mod
    from tiled.client.transport import Transport as OriginalTransport

    httpx_mod = context_mod.httpx
    proxy_transport = make_proxy_transport(proxy_url)
    original_get = httpx_mod.get

    def proxy_get(u: Any, **kw: Any) -> Any:
        client_kw = {k: kw.pop(k) for k in _CLIENT_KWARGS if k in kw}
        client_kw.setdefault("timeout", _PROBE_TIMEOUT)
        transport = make_proxy_transport(proxy_url, verify=client_kw.get("verify", True))
        with httpx_mod.Client(transport=transport, **client_kw) as client:
            return client.get(u, **kw)

    class ProxyTransport(OriginalTransport):
        def __init__(self, *, transport: Any = None, **kw: Any) -> None:
            super().__init__(transport=proxy_transport, **kw)

    state = {
        "context_mod": context_mod,
        "httpx_mod": httpx_mod,
        "original_get": original_get,
        "original_transport_cls": context_mod.Transport,
    }
    httpx_mod.get = proxy_get
    context_mod.Transport = ProxyTransport
    return state


def _restore_proxy(state: dict[str, Any]) -> None:
    state["context_mod"].Transport = state["original_transport_cls"]
    state["httpx_mod"].get = state["original_get"]


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

    # Route through proxy (same pattern as Lightfall's services.tiled_proxy)
    state = _install_proxy(proxy_url)
    try:
        logger.info(f"Connecting to Tiled at {url} via proxy {proxy_url}")
        return from_uri(url, **kwargs)
    finally:
        _restore_proxy(state)
