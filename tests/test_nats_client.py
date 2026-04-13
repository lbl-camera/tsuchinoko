"""Unit tests for NATSClient using a mock nats connection."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tsuchinoko.nats.client import NATSClient
from tsuchinoko.nats.config import NATSConfig


@pytest.fixture
def mock_nc():
    nc = AsyncMock()
    nc.is_connected = True
    nc.subscribe = AsyncMock()
    nc.publish = AsyncMock()
    nc.drain = AsyncMock()
    return nc


@pytest.fixture
def config():
    return NATSConfig(url="nats://localhost:4222", lucid_prefix="test.lucid")


class TestConnection:
    async def test_connect(self, config, mock_nc):
        client = NATSClient()
        with patch("tsuchinoko.nats.client.nats.connect", return_value=mock_nc):
            await client.connect(config)
        assert client.is_connected

    async def test_close(self, config, mock_nc):
        client = NATSClient()
        with patch("tsuchinoko.nats.client.nats.connect", return_value=mock_nc):
            await client.connect(config)
            await client.close()
        mock_nc.drain.assert_awaited_once()
        assert not client.is_connected

    async def test_connect_failure_raises(self, config):
        client = NATSClient()
        with patch("tsuchinoko.nats.client.nats.connect", side_effect=Exception("refused")):
            with pytest.raises(Exception, match="refused"):
                await client.connect(config)
        assert not client.is_connected


class TestAuth:
    async def test_auth_approved(self, config, mock_nc):
        mock_msg = MagicMock()
        mock_msg.data = json.dumps({
            "status": "approved", "tiled_token": "tok123", "tiled_url": "https://tiled.example.com",
        }).encode()
        mock_nc.request = AsyncMock(return_value=mock_msg)

        client = NATSClient()
        with patch("tsuchinoko.nats.client.nats.connect", return_value=mock_nc):
            await client.connect(config)
            creds = await client.authenticate("test.lucid", "tsuchinoko", "1.0", 10.0)

        assert creds["tiled_token"] == "tok123"
        call_args = mock_nc.request.call_args
        assert call_args[0][0] == "test.lucid.auth.request"

    async def test_auth_denied(self, config, mock_nc):
        mock_msg = MagicMock()
        mock_msg.data = json.dumps({"status": "denied", "reason": "not trusted"}).encode()
        mock_nc.request = AsyncMock(return_value=mock_msg)

        client = NATSClient()
        with patch("tsuchinoko.nats.client.nats.connect", return_value=mock_nc):
            await client.connect(config)
            with pytest.raises(PermissionError, match="not trusted"):
                await client.authenticate("test.lucid", "tsuchinoko", "1.0", 10.0)

    async def test_auth_cached(self, config, mock_nc):
        mock_msg = MagicMock()
        mock_msg.data = json.dumps({
            "status": "approved", "tiled_token": "tok", "tiled_url": "https://t",
        }).encode()
        mock_nc.request = AsyncMock(return_value=mock_msg)

        client = NATSClient()
        with patch("tsuchinoko.nats.client.nats.connect", return_value=mock_nc):
            await client.connect(config)
            await client.authenticate("test.lucid", "tsuchinoko", "1.0", 10.0)
            creds = await client.authenticate("test.lucid", "tsuchinoko", "1.0", 10.0)

        assert mock_nc.request.await_count == 1
        assert creds["tiled_token"] == "tok"


class TestPublish:
    async def test_publish(self, config, mock_nc):
        client = NATSClient()
        with patch("tsuchinoko.nats.client.nats.connect", return_value=mock_nc):
            await client.connect(config)
            await client.publish("tsuchinoko.state", {"state": "running"})

        mock_nc.publish.assert_awaited_once()
        subject, data = mock_nc.publish.call_args[0]
        assert subject == "tsuchinoko.state"
        assert json.loads(data) == {"state": "running"}

    async def test_publish_when_disconnected(self):
        client = NATSClient()
        await client.publish("tsuchinoko.state", {"state": "running"})  # should not raise


class TestSubscribe:
    async def test_subscribe(self, config, mock_nc):
        cb = AsyncMock()
        mock_sub = MagicMock()
        mock_nc.subscribe = AsyncMock(return_value=mock_sub)

        client = NATSClient()
        with patch("tsuchinoko.nats.client.nats.connect", return_value=mock_nc):
            await client.connect(config)
            sub = await client.subscribe("tsuchinoko.status", cb)

        assert sub is mock_sub
        mock_nc.subscribe.assert_awaited_once_with("tsuchinoko.status", cb=cb)


class TestRequest:
    async def test_request(self, config, mock_nc):
        mock_msg = MagicMock()
        mock_msg.data = json.dumps({"result": "ok"}).encode()
        mock_nc.request = AsyncMock(return_value=mock_msg)

        client = NATSClient()
        with patch("tsuchinoko.nats.client.nats.connect", return_value=mock_nc):
            await client.connect(config)
            result = await client.request("tsuchinoko.status", {}, timeout=5.0)

        assert result == {"result": "ok"}
