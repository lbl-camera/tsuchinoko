"""Tests for tsuchinoko.tiled.connect — ApiKeyAuth flow."""
from __future__ import annotations

from unittest.mock import MagicMock

import httpx

from tsuchinoko.tiled.connect import ApiKeyAuth


def test_apikey_auth_sets_authorization_header():
    auth = ApiKeyAuth("secret-key-here")
    request = httpx.Request("GET", "https://example.test/")
    # Drive the auth flow generator manually.
    flow = auth.sync_auth_flow(request)
    next(flow)
    assert request.headers["Authorization"] == "Apikey secret-key-here"


def test_apikey_auth_does_not_retry_on_response():
    """Static API keys have no refresh path; the flow yields once and ends."""
    auth = ApiKeyAuth("secret")
    request = httpx.Request("GET", "https://example.test/")
    flow = auth.sync_auth_flow(request)
    next(flow)
    # Sending any response should terminate the generator.
    fake_response = MagicMock(status_code=401)
    try:
        flow.send(fake_response)
        raise AssertionError("Expected StopIteration")
    except StopIteration:
        pass


# --- tiled httpx flavour (httpx2 on tiled>=0.2.21) ---------------------------

import pytest  # noqa: E402

from tsuchinoko.tiled import connect as connect_mod  # noqa: E402


def test_apikey_auth_is_tileds_auth_flavour():
    """Tiled's Context only accepts an instance of *its* httpx flavour's Auth."""
    auth = ApiKeyAuth("s3cr3t-value")
    assert isinstance(auth, connect_mod.tiled_httpx().Auth)
    assert "s3cr3t-value" not in repr(auth)  # never leak the key into logs


def test_apikey_header_sent_by_tileds_client_flavour():
    httpx_mod = connect_mod.tiled_httpx()
    seen = []

    def handler(request):
        seen.append(request.headers.get("Authorization"))
        return httpx_mod.Response(200, json={})

    with httpx_mod.Client(transport=httpx_mod.MockTransport(handler),
                          auth=ApiKeyAuth("secret")) as client:
        client.get("http://tiled.test/api/v1/")
    assert seen == ["Apikey secret"]


@pytest.fixture
def mock_proxy(monkeypatch):
    """Replace the proxy transport with a recording MockTransport of tiled's flavour."""
    httpx_mod = connect_mod.tiled_httpx()
    seen = []

    def handler(request):
        seen.append((str(request.url), request.headers.get("Authorization")))
        # Not a real Tiled server: refuse, so from_uri stops after its first calls.
        return httpx_mod.Response(401, json={"detail": "mock"})

    proxies = []

    def fake_make(proxy_url, **kwargs):
        proxies.append(proxy_url)
        return httpx_mod.MockTransport(handler)

    monkeypatch.setattr(connect_mod, "make_proxy_transport", fake_make)
    return seen, proxies


def test_connect_via_proxy_routes_through_proxy_with_apikey(mock_proxy):
    import tiled.client.context as context_mod

    seen, proxies = mock_proxy
    original_get = context_mod.httpx.get
    original_transport = context_mod.Transport
    with pytest.raises(Exception) as excinfo:
        connect_mod.connect_tiled(
            "http://tiled.test/api/v1", api_key="secret",
            proxy_url="socks5://localhost:1080",
        )
    # Tiled accepted the auth object (no "httpx.Auth or None allowed" TypeError)
    # and only failed on the mock server's 401.
    assert "httpx.Auth or None allowed" not in str(excinfo.value)
    assert not isinstance(excinfo.value, TypeError)
    assert seen, "no request went through the proxy transport"
    assert set(proxies) == {"socks5://localhost:1080"}
    # (The key itself is only attached after the server's root response, which
    # this mock refuses; test_apikey_header_sent_by_tileds_client_flavour
    # covers the header.)
    # Patches are undone.
    assert context_mod.httpx.get is original_get
    assert context_mod.Transport is original_transport


def test_proxy_probe_accepts_client_kwargs(mock_proxy):
    """tiled>=0.2.21 probes with httpx.get(uri, timeout=..., trust_env=...)."""
    import tiled.client.context as context_mod

    seen, _ = mock_proxy
    state = connect_mod._install_proxy("socks5://localhost:1080")
    try:
        response = context_mod.httpx.get(
            "http://tiled.test/api/v1/", timeout=5.0, trust_env=True,
        )
    finally:
        connect_mod._restore_proxy(state)
    assert response.status_code == 401
    assert seen[0][0] == "http://tiled.test/api/v1/"
