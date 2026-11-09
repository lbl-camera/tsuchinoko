"""Integration tests requiring a real NATS broker at localhost:4222."""

import asyncio
import json
import time
import uuid
from threading import Thread

import numpy as np
import pytest
import pytest_asyncio

import nats as nats_lib

from tsuchinoko.adaptive.random_in_process import RandomInProcess
from tsuchinoko.core import Core, CoreState
from tsuchinoko.execution.simple import SimpleEngine
from tsuchinoko.nats.client import NATSClient
from tsuchinoko.nats.config import NATSConfig

NATS_URL = "nats://localhost:4222"


@pytest.fixture(scope="session")
def nats_url():
    return NATS_URL


@pytest_asyncio.fixture
async def nats_available(nats_url):
    try:
        nc = await nats_lib.connect(nats_url)
        await nc.drain()
    except Exception:
        pytest.skip("NATS server not available at localhost:4222")


@pytest_asyncio.fixture
async def mock_lightfall(nats_url, nats_available):
    nc = await nats_lib.connect(nats_url)
    prefix = f"test.lightfall.{uuid.uuid4().hex[:8]}"

    async def handle_auth(msg):
        reply = json.dumps({
            "status": "approved",
            "tiled_token": "test-token-123",
            "tiled_url": "https://tiled.test.example.com",
        }).encode()
        if msg.reply:
            await nc.publish(msg.reply, reply)

    sub = await nc.subscribe(f"{prefix}.auth.request", cb=handle_auth)
    yield {"prefix": prefix, "nc": nc}
    await sub.unsubscribe()
    await nc.drain()


def _measure(pos):
    x, y = pos
    return pos, np.sin(x / 30), 0.1, {}


def _slow_measure(pos):
    """Measure function with a delay so the core stays alive long enough to query."""
    time.sleep(0.3)
    x, y = pos
    return pos, np.sin(x / 30), 0.1, {}


def _make_core(nats_url, measure_func=_slow_measure, exit_at=None, pause_at=None):
    """Helper to build a Core with NATS that stays alive for queries."""
    engine = RandomInProcess(dimensionality=2, parameter_bounds=[(0, 100), (0, 100)])
    execution = SimpleEngine(measure_func=measure_func)
    config = NATSConfig(url=nats_url, lightfall_prefix="")
    core = Core(execution_engine=execution, adaptive_engine=engine, compute_metrics=False, nats_config=config)
    if exit_at is not None:
        core.exit_at = exit_at
    if pause_at is not None:
        core.pause_at = pause_at
    return core


@pytest.mark.asyncio
class TestNATSClientIntegration:
    async def test_connect_and_close(self, nats_url, nats_available):
        client = NATSClient()
        config = NATSConfig(url=nats_url)
        await client.connect(config)
        assert client.is_connected
        await client.close()
        assert not client.is_connected

    async def test_auth_handshake(self, nats_url, mock_lightfall):
        client = NATSClient()
        config = NATSConfig(url=nats_url, lightfall_prefix=mock_lightfall["prefix"])
        await client.connect(config)
        creds = await client.authenticate(mock_lightfall["prefix"], "tsuchinoko", "test", timeout=5.0)
        assert creds["tiled_token"] == "test-token-123"
        await client.close()

    async def test_publish_subscribe(self, nats_url, nats_available):
        client = NATSClient()
        config = NATSConfig(url=nats_url)
        await client.connect(config)

        received = []

        async def handler(msg):
            received.append(json.loads(msg.data))

        await client.subscribe("test.tsuchinoko.ping", handler)
        await client.publish("test.tsuchinoko.ping", {"hello": "world"})
        await asyncio.sleep(0.5)

        assert len(received) == 1
        assert received[0] == {"hello": "world"}
        await client.close()


@pytest.mark.asyncio
class TestNATSServiceIntegration:
    async def test_action_round_trip(self, nats_url, nats_available):
        core = _make_core(nats_url, exit_at=[20])

        thread = Thread(target=core.main, daemon=True)
        thread.start()
        core.state = CoreState.Starting
        await asyncio.sleep(1.5)

        nc = await nats_lib.connect(nats_url)
        try:
            msg = await nc.request("tsuchinoko.status", b"{}", timeout=5.0)
            reply = json.loads(msg.data)
            assert reply["status"] == "ok"
            assert reply["state"] in ["Running", "Inactive", "Exiting"]
        finally:
            await nc.drain()

        thread.join(timeout=30)
        assert not thread.is_alive()

    async def test_discovery(self, nats_url, nats_available):
        core = _make_core(nats_url, exit_at=[20])

        thread = Thread(target=core.main, daemon=True)
        thread.start()
        core.state = CoreState.Starting
        await asyncio.sleep(1.5)

        nc = await nats_lib.connect(nats_url)
        try:
            msg = await nc.request("_tsuchinoko.discover", b"{}", timeout=5.0)
            reply = json.loads(msg.data)
            assert reply["app_name"] == "tsuchinoko"
            assert reply["prefix"] == "tsuchinoko"
            assert reply["actions_count"] == 8
            assert reply["events_count"] == 4
        finally:
            await nc.drain()

        thread.join(timeout=30)

    async def test_meta_actions(self, nats_url, nats_available):
        core = _make_core(nats_url, exit_at=[20])

        thread = Thread(target=core.main, daemon=True)
        thread.start()
        core.state = CoreState.Starting
        await asyncio.sleep(1.5)

        nc = await nats_lib.connect(nats_url)
        try:
            msg = await nc.request("tsuchinoko.meta.actions", b"{}", timeout=5.0)
            reply = json.loads(msg.data)
            suffixes = [a["suffix"] for a in reply["actions"]]
            assert "experiment.start" in suffixes
            assert "status" in suffixes
        finally:
            await nc.drain()

        thread.join(timeout=30)
