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
