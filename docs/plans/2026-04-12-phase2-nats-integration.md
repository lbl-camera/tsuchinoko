# Phase 2: NATS Integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a NATS client to Tsuchinoko so it can be remotely controlled over the NATS bus, with self-describing discovery for the future generic NATS-MCP bridge.

**Architecture:** Three new modules — `NATSClient` (connection/auth/pub-sub), `NATSService` (action handlers/discovery), `NATSConfig` (Pydantic settings). Core's `_main()` is rewritten as a proper async loop that owns the NATS lifecycle. The experiment thread publishes events via an `asyncio.Queue`. NATS is optional — omitting the URL preserves Phase 1 behavior exactly.

**Tech Stack:** nats-py >=2.0, Pydantic >=2.0, asyncio, pytest / pytest-asyncio

**Design spec:** `docs/design/2026-04-12-phase2-nats-integration.md`

---

## File Structure

### New files
| File | Responsibility |
|------|----------------|
| `tsuchinoko/nats/__init__.py` | Package exports: NATSClient, NATSService, NATSConfig |
| `tsuchinoko/nats/client.py` | Connection lifecycle, auth handshake, publish/subscribe |
| `tsuchinoko/nats/service.py` | Action handlers, discovery, meta endpoints, event publishing |
| `tsuchinoko/nats/config.py` | NATSConfig Pydantic model |
| `tests/test_nats_client.py` | Unit tests for NATSClient (mock nats connection) |
| `tests/test_nats_service.py` | Unit tests for NATSService (mock client + core) |
| `tests/test_core_rewrite.py` | Tests for rewritten _main() (no broker required) |
| `tests/test_nats_integration.py` | Integration tests (real NATS broker, skipped if unavailable) |

### Modified files
| File | Changes |
|------|---------|
| `tsuchinoko/core/__init__.py` | Rewrite `_main()`, add event queue, NATS lifecycle, accept `nats_config` |
| `tsuchinoko/config.py` | Add NATSConfig to AppConfig |
| `tsuchinoko/cli.py` | Real `run` command with --nats-url, --lucid-prefix |
| `pyproject.toml` | Add `nats-py>=2.0` to core dependencies |

---

## Task 1: NATSConfig and dependency

**Files:**
- Create: `tsuchinoko/nats/__init__.py`
- Create: `tsuchinoko/nats/config.py`
- Modify: `tsuchinoko/config.py`
- Modify: `pyproject.toml`
- Test: `tests/test_nats_config.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_nats_config.py`:

```python
"""Tests for NATS configuration."""

import pytest
from tsuchinoko.nats.config import NATSConfig
from tsuchinoko.config import AppConfig


class TestNATSConfig:
    def test_defaults(self):
        cfg = NATSConfig()
        assert cfg.url == ""
        assert cfg.lucid_prefix == "als.7011"
        assert cfg.app_name == "tsuchinoko"
        assert cfg.app_version == ""
        assert cfg.auth_timeout == 70.0
        assert cfg.connect_timeout == 5.0
        assert cfg.reconnect is True

    def test_custom_values(self):
        cfg = NATSConfig(url="nats://broker:4222", lucid_prefix="als.1234")
        assert cfg.url == "nats://broker:4222"
        assert cfg.lucid_prefix == "als.1234"

    def test_nats_disabled_by_default(self):
        cfg = NATSConfig()
        assert not cfg.url  # empty string is falsy

    def test_app_config_includes_nats(self):
        cfg = AppConfig()
        assert hasattr(cfg, 'nats')
        assert isinstance(cfg.nats, NATSConfig)
        assert cfg.nats.url == ""

    def test_app_config_from_dict(self):
        cfg = AppConfig(**{"nats": {"url": "nats://localhost:4222", "lucid_prefix": "test.prefix"}})
        assert cfg.nats.url == "nats://localhost:4222"
        assert cfg.nats.lucid_prefix == "test.prefix"
```

- [ ] **Step 2: Run test — verify it fails**

Run: `cd /c/Users/rp/PycharmProjects/tsuchinoko-phase1 && .venv/Scripts/python -m pytest tests/test_nats_config.py -v --noconftest`

Expected: `ModuleNotFoundError: No module named 'tsuchinoko.nats'`

- [ ] **Step 3: Create NATSConfig**

Create `tsuchinoko/nats/__init__.py`:

```python
"""NATS integration for Tsuchinoko."""

from .config import NATSConfig
```

Create `tsuchinoko/nats/config.py`:

```python
"""NATS configuration model."""

from pydantic import BaseModel, Field


class NATSConfig(BaseModel):
    """Configuration for the NATS connection."""
    url: str = ""
    lucid_prefix: str = "als.7011"
    app_name: str = "tsuchinoko"
    app_version: str = ""
    auth_timeout: float = Field(70.0, description="Auth handshake timeout (>60s for LUCID trust dialog)")
    connect_timeout: float = Field(5.0, description="Initial connection timeout")
    reconnect: bool = True
```

- [ ] **Step 4: Add NATSConfig to AppConfig**

In `tsuchinoko/config.py`, add import and field:

```python
from tsuchinoko.nats.config import NATSConfig
```

Add to `AppConfig`:

```python
class AppConfig(BaseModel):
    """Application-wide configuration."""
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    core: CoreConfig = Field(default_factory=CoreConfig)
    nats: NATSConfig = Field(default_factory=NATSConfig)
```

- [ ] **Step 5: Add nats-py dependency**

In `pyproject.toml`, add to core dependencies:

```toml
dependencies = [
  "gpCAM~=8.2.3",
  "fvgp~=4.6.5",
  "numpy",
  "scipy",
  "scikit-learn",
  "click",
  "loguru",
  "appdirs",
  "transitions",
  "pydantic>=2.0",
  "pyyaml",
  "nats-py>=2.0",
]
```

Install it: `cd /c/Users/rp/PycharmProjects/tsuchinoko-phase1 && .venv/Scripts/pip install nats-py -q`

- [ ] **Step 6: Run tests — verify pass**

Run: `cd /c/Users/rp/PycharmProjects/tsuchinoko-phase1 && .venv/Scripts/python -m pytest tests/test_nats_config.py -v --noconftest`

Expected: All 5 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add tsuchinoko/nats/ tsuchinoko/config.py pyproject.toml tests/test_nats_config.py
git commit -m "feat: add NATSConfig and nats-py dependency"
```

---

## Task 2: NATSClient

**Files:**
- Create: `tsuchinoko/nats/client.py`
- Create: `tests/test_nats_client.py`
- Modify: `tsuchinoko/nats/__init__.py`

The NATSClient owns the NATS connection, auth handshake, and publish/subscribe. All async. Thread-safe `publish_threadsafe()` for the experiment thread.

- [ ] **Step 1: Write failing tests**

Create `tests/test_nats_client.py`:

```python
"""Unit tests for NATSClient using a mock nats connection."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tsuchinoko.nats.client import NATSClient
from tsuchinoko.nats.config import NATSConfig


@pytest.fixture
def mock_nc():
    """Mock nats.NATS connection."""
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
            "status": "approved",
            "tiled_token": "tok123",
            "tiled_url": "https://tiled.example.com",
        }).encode()
        mock_nc.request = AsyncMock(return_value=mock_msg)

        client = NATSClient()
        with patch("tsuchinoko.nats.client.nats.connect", return_value=mock_nc):
            await client.connect(config)
            creds = await client.authenticate(
                lucid_prefix="test.lucid",
                app_name="tsuchinoko",
                app_version="1.0",
                timeout=10.0,
            )

        assert creds["tiled_token"] == "tok123"
        assert creds["tiled_url"] == "https://tiled.example.com"
        mock_nc.request.assert_awaited_once()
        # Subject should be {prefix}.auth.request
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
            # Second call should not hit NATS
            creds = await client.authenticate("test.lucid", "tsuchinoko", "1.0", 10.0)

        assert mock_nc.request.await_count == 1  # only called once
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

    async def test_publish_when_disconnected(self, config):
        client = NATSClient()
        # Should not raise, just silently skip
        await client.publish("tsuchinoko.state", {"state": "running"})


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
```

- [ ] **Step 2: Run tests — verify fail**

Run: `cd /c/Users/rp/PycharmProjects/tsuchinoko-phase1 && .venv/Scripts/python -m pytest tests/test_nats_client.py -v --noconftest`

Expected: `ModuleNotFoundError: No module named 'tsuchinoko.nats.client'`

- [ ] **Step 3: Implement NATSClient**

Create `tsuchinoko/nats/client.py`:

```python
"""NATS client for Tsuchinoko — connection, auth, publish/subscribe."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable, Optional

import nats
import nats.errors
from loguru import logger

from .config import NATSConfig


class NATSClient:
    """Shared NATS connection owner.

    Handles connection lifecycle, LUCID auth handshake, and
    thread-safe publish for the experiment thread.
    """

    def __init__(self) -> None:
        self._nc: Optional[nats.NATS] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._auth_state: dict[str, str] = {}
        self._tiled_credentials: dict[str, dict] = {}
        self.is_connected: bool = False

    async def connect(self, config: NATSConfig) -> None:
        """Connect to the NATS broker."""
        self._nc = await nats.connect(
            config.url,
            connect_timeout=int(config.connect_timeout),
            allow_reconnect=config.reconnect,
            error_cb=self._on_error,
            disconnected_cb=self._on_disconnect,
            reconnected_cb=self._on_reconnect,
        )
        self._loop = asyncio.get_running_loop()
        self.is_connected = True
        logger.info(f"Connected to NATS at {config.url}")

    async def close(self) -> None:
        """Drain and close the NATS connection."""
        if self._nc:
            await self._nc.drain()
        self.is_connected = False
        self._nc = None
        logger.info("NATS connection closed")

    async def authenticate(
        self,
        lucid_prefix: str,
        app_name: str,
        app_version: str,
        timeout: float,
    ) -> dict:
        """Perform auth handshake with LUCID. Returns Tiled credentials.

        Caches auth state per prefix — second call returns cached creds.
        Raises PermissionError if LUCID denies access.
        """
        cached = self._auth_state.get(lucid_prefix)
        if cached == "approved":
            return self._tiled_credentials.get(lucid_prefix, {})
        if cached == "denied":
            raise PermissionError(f"LUCID at '{lucid_prefix}' previously denied access")

        subject = f"{lucid_prefix}.auth.request"
        payload = json.dumps({
            "app_name": app_name,
            "app_version": app_version,
        }).encode()

        msg = await self._nc.request(subject, payload, timeout=timeout)
        data = json.loads(msg.data)

        if data.get("status") == "approved":
            self._auth_state[lucid_prefix] = "approved"
            self._tiled_credentials[lucid_prefix] = {
                "tiled_token": data.get("tiled_token"),
                "tiled_url": data.get("tiled_url"),
            }
            logger.info(f"Authenticated with LUCID at '{lucid_prefix}'")
            return self._tiled_credentials[lucid_prefix]
        else:
            reason = data.get("reason", "denied by operator")
            self._auth_state[lucid_prefix] = "denied"
            raise PermissionError(f"LUCID denied access: {reason}")

    async def publish(self, subject: str, payload: dict) -> None:
        """Publish a JSON message. No-op if disconnected."""
        if self._nc and self.is_connected:
            await self._nc.publish(subject, json.dumps(payload).encode())

    async def subscribe(self, subject: str, callback: Callable) -> Any:
        """Subscribe to a subject with an async callback."""
        return await self._nc.subscribe(subject, cb=callback)

    async def request(self, subject: str, payload: dict, timeout: float = 5.0) -> dict:
        """Send a request and return the parsed JSON response."""
        msg = await self._nc.request(subject, json.dumps(payload).encode(), timeout=timeout)
        return json.loads(msg.data)

    def publish_threadsafe(self, subject: str, payload: dict) -> None:
        """Publish from any thread (fire-and-forget)."""
        if self._nc and self.is_connected and self._loop:
            asyncio.run_coroutine_threadsafe(
                self.publish(subject, payload), self._loop
            )

    async def _on_error(self, exc: Exception) -> None:
        logger.error(f"NATS error: {exc}")

    async def _on_disconnect(self) -> None:
        self.is_connected = False
        logger.warning("NATS disconnected")

    async def _on_reconnect(self) -> None:
        self.is_connected = True
        logger.info("NATS reconnected")
```

- [ ] **Step 4: Update nats __init__.py**

```python
"""NATS integration for Tsuchinoko."""

from .config import NATSConfig
from .client import NATSClient
```

- [ ] **Step 5: Run tests — verify pass**

Run: `cd /c/Users/rp/PycharmProjects/tsuchinoko-phase1 && .venv/Scripts/python -m pytest tests/test_nats_client.py -v --noconftest`

Expected: All 9 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add tsuchinoko/nats/client.py tsuchinoko/nats/__init__.py tests/test_nats_client.py
git commit -m "feat: add NATSClient with connection, auth, and publish/subscribe"
```

---

## Task 3: NATSService

**Files:**
- Create: `tsuchinoko/nats/service.py`
- Create: `tests/test_nats_service.py`
- Modify: `tsuchinoko/nats/__init__.py`

The NATSService registers action handlers on `tsuchinoko.*`, serves discovery
and meta endpoints, and provides an `emit_event()` method for the experiment thread.

- [ ] **Step 1: Write failing tests**

Create `tests/test_nats_service.py`:

```python
"""Unit tests for NATSService using mock NATSClient and Core."""

import asyncio
import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from tsuchinoko.nats.service import NATSService


def _make_mock_core():
    """Create a mock Core with state machine behavior."""
    from tsuchinoko.core import CoreState

    core = MagicMock()
    core.state = CoreState.Inactive
    core.data = MagicMock()
    core.data._completed_iterations = 0
    core.data.__len__ = MagicMock(return_value=0)
    core.adaptive_engine = MagicMock()
    core.adaptive_engine.parameters = MagicMock()
    core.adaptive_engine.parameters.saveState.return_value = {"name": "top", "children": []}
    core.adaptive_engine.dimensionality = 2
    return core


def _make_mock_client():
    """Create a mock NATSClient."""
    client = AsyncMock()
    client._nc = AsyncMock()
    mock_sub = MagicMock()
    mock_sub.unsubscribe = AsyncMock()
    client.subscribe = AsyncMock(return_value=mock_sub)
    client.publish = AsyncMock()
    return client


def _make_nats_msg(data: dict, reply: str = "_INBOX.test"):
    """Create a mock NATS message with reply subject."""
    msg = MagicMock()
    msg.data = json.dumps(data).encode()
    msg.reply = reply
    msg.respond = AsyncMock()
    return msg


class TestServiceLifecycle:
    async def test_start_registers_subscriptions(self):
        core = _make_mock_core()
        client = _make_mock_client()
        service = NATSService(core, client)
        await service.start()

        # Should have registered: 8 actions + discover + meta.actions + meta.events = 11
        assert client.subscribe.await_count == 11

    async def test_stop_unsubscribes(self):
        core = _make_mock_core()
        client = _make_mock_client()
        service = NATSService(core, client)
        await service.start()
        await service.stop()

        # All subscriptions should be unsubscribed
        for call in client.subscribe.return_value.unsubscribe.call_args_list:
            pass  # just verifying it was called
        assert client.subscribe.return_value.unsubscribe.await_count == 11


class TestActionHandlers:
    async def test_handle_start(self):
        from tsuchinoko.core import CoreState
        core = _make_mock_core()
        client = _make_mock_client()
        service = NATSService(core, client)

        msg = _make_nats_msg({})
        await service._handle_start(msg)

        assert core.state == CoreState.Starting
        msg.respond.assert_awaited_once()
        reply = json.loads(msg.respond.call_args[0][0])
        assert reply["status"] == "ok"

    async def test_handle_pause(self):
        from tsuchinoko.core import CoreState
        core = _make_mock_core()
        core.state = CoreState.Running
        client = _make_mock_client()
        service = NATSService(core, client)

        msg = _make_nats_msg({})
        await service._handle_pause(msg)

        assert core.state == CoreState.Pausing
        msg.respond.assert_awaited_once()

    async def test_handle_resume(self):
        from tsuchinoko.core import CoreState
        core = _make_mock_core()
        core.state = CoreState.Paused
        client = _make_mock_client()
        service = NATSService(core, client)

        msg = _make_nats_msg({})
        await service._handle_resume(msg)

        assert core.state == CoreState.Resuming

    async def test_handle_stop(self):
        from tsuchinoko.core import CoreState
        core = _make_mock_core()
        core.state = CoreState.Running
        client = _make_mock_client()
        service = NATSService(core, client)

        msg = _make_nats_msg({})
        await service._handle_stop(msg)

        assert core.state == CoreState.Stopping

    async def test_handle_status(self):
        from tsuchinoko.core import CoreState
        core = _make_mock_core()
        core.state = CoreState.Running
        core.data._completed_iterations = 42
        core.data.__len__ = MagicMock(return_value=100)
        client = _make_mock_client()
        service = NATSService(core, client)

        msg = _make_nats_msg({})
        await service._handle_status(msg)

        reply = json.loads(msg.respond.call_args[0][0])
        assert reply["state"] == "Running"
        assert reply["iteration"] == 42
        assert reply["data_count"] == 100

    async def test_handle_get_parameters(self):
        core = _make_mock_core()
        client = _make_mock_client()
        service = NATSService(core, client)

        msg = _make_nats_msg({})
        await service._handle_get_parameters(msg)

        reply = json.loads(msg.respond.call_args[0][0])
        assert "parameters" in reply
        core.adaptive_engine.parameters.saveState.assert_called_once()

    async def test_handle_set_parameter(self):
        core = _make_mock_core()
        client = _make_mock_client()
        service = NATSService(core, client)

        msg = _make_nats_msg({"path": ["bounds", "axis_0_min"], "value": 5.0})
        await service._handle_set_parameter(msg)

        core.adaptive_engine.parameters.child.assert_called_once_with("bounds", "axis_0_min")
        core.adaptive_engine.parameters.child().setValue.assert_called_once_with(5.0)

    async def test_handler_error_returns_error_reply(self):
        core = _make_mock_core()
        # Make state setter raise
        type(core).state = PropertyMock(side_effect=Exception("boom"))
        client = _make_mock_client()
        service = NATSService(core, client)

        msg = _make_nats_msg({})
        await service._handle_start(msg)

        reply = json.loads(msg.respond.call_args[0][0])
        assert reply["status"] == "error"
        assert "boom" in reply["message"]


class TestDiscovery:
    async def test_discover_response(self):
        core = _make_mock_core()
        client = _make_mock_client()
        service = NATSService(core, client)

        msg = _make_nats_msg({})
        await service._handle_discover(msg)

        reply = json.loads(msg.respond.call_args[0][0])
        assert reply["app_name"] == "tsuchinoko"
        assert reply["prefix"] == "tsuchinoko"
        assert reply["actions_count"] == 8
        assert reply["events_count"] == 4
        assert "instance_id" in reply

    async def test_meta_actions(self):
        core = _make_mock_core()
        client = _make_mock_client()
        service = NATSService(core, client)

        msg = _make_nats_msg({})
        await service._handle_meta_actions(msg)

        reply = json.loads(msg.respond.call_args[0][0])
        actions = reply["actions"]
        suffixes = [a["suffix"] for a in actions]
        assert "experiment.start" in suffixes
        assert "experiment.stop" in suffixes
        assert "engine.get_parameters" in suffixes
        assert "status" in suffixes
        assert len(actions) == 8

    async def test_meta_events(self):
        core = _make_mock_core()
        client = _make_mock_client()
        service = NATSService(core, client)

        msg = _make_nats_msg({})
        await service._handle_meta_events(msg)

        reply = json.loads(msg.respond.call_args[0][0])
        events = reply["events"]
        suffixes = [e["suffix"] for e in events]
        assert "state" in suffixes
        assert "targets" in suffixes
        assert "gp.updated" in suffixes
        assert "error" in suffixes
        assert len(events) == 4
```

- [ ] **Step 2: Run tests — verify fail**

Run: `cd /c/Users/rp/PycharmProjects/tsuchinoko-phase1 && .venv/Scripts/python -m pytest tests/test_nats_service.py -v --noconftest`

Expected: `ModuleNotFoundError: No module named 'tsuchinoko.nats.service'`

- [ ] **Step 3: Implement NATSService**

Create `tsuchinoko/nats/service.py`:

```python
"""Tsuchinoko's service identity on the NATS bus.

Registers action handlers, serves discovery and meta endpoints,
and provides event publishing for the experiment thread.
"""

from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from tsuchinoko.core import Core
    from .client import NATSClient

# Action catalog
ACTIONS = [
    {"suffix": "experiment.configure", "description": "Set up experiment parameters"},
    {"suffix": "experiment.start", "description": "Begin the adaptive loop"},
    {"suffix": "experiment.pause", "description": "Pause the loop"},
    {"suffix": "experiment.resume", "description": "Resume from pause"},
    {"suffix": "experiment.stop", "description": "Stop and finalize"},
    {"suffix": "engine.set_parameter", "description": "Update a single engine parameter"},
    {"suffix": "engine.get_parameters", "description": "Retrieve current engine parameters"},
    {"suffix": "status", "description": "Query current state and progress"},
]

# Event catalog
EVENTS = [
    {"suffix": "state", "description": "State machine transitions"},
    {"suffix": "targets", "description": "New targets computed"},
    {"suffix": "gp.updated", "description": "GP outputs updated"},
    {"suffix": "error", "description": "Error in adaptive loop"},
]


class NATSService:
    """Tsuchinoko's NATS service — actions, discovery, and events."""

    def __init__(self, core: Core, client: NATSClient) -> None:
        self._core = core
        self._client = client
        self._subscriptions = []
        self._instance_id = str(uuid.uuid4())

    async def start(self) -> None:
        """Register all action handlers and meta endpoints."""
        handler_map = {
            "experiment.configure": self._handle_configure,
            "experiment.start": self._handle_start,
            "experiment.pause": self._handle_pause,
            "experiment.resume": self._handle_resume,
            "experiment.stop": self._handle_stop,
            "engine.set_parameter": self._handle_set_parameter,
            "engine.get_parameters": self._handle_get_parameters,
            "status": self._handle_status,
        }

        for suffix, handler in handler_map.items():
            sub = await self._client.subscribe(f"tsuchinoko.{suffix}", handler)
            self._subscriptions.append(sub)

        # Discovery and meta
        sub = await self._client.subscribe("_tsuchinoko.discover", self._handle_discover)
        self._subscriptions.append(sub)
        sub = await self._client.subscribe("tsuchinoko.meta.actions", self._handle_meta_actions)
        self._subscriptions.append(sub)
        sub = await self._client.subscribe("tsuchinoko.meta.events", self._handle_meta_events)
        self._subscriptions.append(sub)

        logger.info(f"NATSService started: {len(handler_map)} actions, instance={self._instance_id[:8]}")

    async def stop(self) -> None:
        """Unsubscribe all handlers."""
        for sub in self._subscriptions:
            await sub.unsubscribe()
        self._subscriptions.clear()
        logger.info("NATSService stopped")

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    async def _reply(self, msg, data: dict) -> None:
        """Send a JSON reply to a NATS request message."""
        await msg.respond(json.dumps(data).encode())

    async def _handle_configure(self, msg) -> None:
        try:
            data = json.loads(msg.data)
            # Apply configuration to adaptive engine parameters
            if "parameter_bounds" in data:
                bounds = data["parameter_bounds"]
                for i, (lo, hi) in enumerate(bounds):
                    self._core.adaptive_engine.parameters[("bounds", f"axis_{i}_min")] = lo
                    self._core.adaptive_engine.parameters[("bounds", f"axis_{i}_max")] = hi
            await self._reply(msg, {"status": "ok"})
        except Exception as e:
            logger.exception(e)
            await self._reply(msg, {"status": "error", "message": str(e)})

    async def _handle_start(self, msg) -> None:
        try:
            from tsuchinoko.core import CoreState
            self._core.state = CoreState.Starting
            await self._reply(msg, {"status": "ok", "state": self._core.state.name})
        except Exception as e:
            logger.exception(e)
            await self._reply(msg, {"status": "error", "message": str(e)})

    async def _handle_pause(self, msg) -> None:
        try:
            from tsuchinoko.core import CoreState
            self._core.state = CoreState.Pausing
            await self._reply(msg, {"status": "ok", "state": self._core.state.name})
        except Exception as e:
            logger.exception(e)
            await self._reply(msg, {"status": "error", "message": str(e)})

    async def _handle_resume(self, msg) -> None:
        try:
            from tsuchinoko.core import CoreState
            self._core.state = CoreState.Resuming
            await self._reply(msg, {"status": "ok", "state": self._core.state.name})
        except Exception as e:
            logger.exception(e)
            await self._reply(msg, {"status": "error", "message": str(e)})

    async def _handle_stop(self, msg) -> None:
        try:
            from tsuchinoko.core import CoreState
            self._core.state = CoreState.Stopping
            await self._reply(msg, {"status": "ok", "state": self._core.state.name})
        except Exception as e:
            logger.exception(e)
            await self._reply(msg, {"status": "error", "message": str(e)})

    async def _handle_set_parameter(self, msg) -> None:
        try:
            data = json.loads(msg.data)
            path = data["path"]
            value = data["value"]
            self._core.adaptive_engine.parameters.child(*path).setValue(value)
            await self._reply(msg, {"status": "ok"})
        except Exception as e:
            logger.exception(e)
            await self._reply(msg, {"status": "error", "message": str(e)})

    async def _handle_get_parameters(self, msg) -> None:
        try:
            state = self._core.adaptive_engine.parameters.saveState()
            await self._reply(msg, {"status": "ok", "parameters": state})
        except Exception as e:
            logger.exception(e)
            await self._reply(msg, {"status": "error", "message": str(e)})

    async def _handle_status(self, msg) -> None:
        try:
            await self._reply(msg, {
                "status": "ok",
                "state": self._core.state.name,
                "iteration": self._core.data._completed_iterations,
                "data_count": len(self._core.data),
            })
        except Exception as e:
            logger.exception(e)
            await self._reply(msg, {"status": "error", "message": str(e)})

    # ------------------------------------------------------------------
    # Discovery and meta
    # ------------------------------------------------------------------

    async def _handle_discover(self, msg) -> None:
        await self._reply(msg, {
            "instance_id": self._instance_id,
            "app_name": "tsuchinoko",
            "app_version": "",
            "prefix": "tsuchinoko",
            "actions_count": len(ACTIONS),
            "events_count": len(EVENTS),
            "state": self._core.state.name,
        })

    async def _handle_meta_actions(self, msg) -> None:
        await self._reply(msg, {"actions": ACTIONS})

    async def _handle_meta_events(self, msg) -> None:
        await self._reply(msg, {"events": EVENTS})
```

- [ ] **Step 4: Update nats __init__.py**

```python
"""NATS integration for Tsuchinoko."""

from .config import NATSConfig
from .client import NATSClient
from .service import NATSService
```

- [ ] **Step 5: Run tests — verify pass**

Run: `cd /c/Users/rp/PycharmProjects/tsuchinoko-phase1 && .venv/Scripts/python -m pytest tests/test_nats_service.py -v --noconftest`

Expected: All 14 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add tsuchinoko/nats/service.py tsuchinoko/nats/__init__.py tests/test_nats_service.py
git commit -m "feat: add NATSService with action handlers and discovery"
```

---

## Task 4: Core `_main()` rewrite

**Files:**
- Modify: `tsuchinoko/core/__init__.py`
- Create: `tests/test_core_rewrite.py`

Rewrite `_main()` to own the NATS lifecycle and drain an event queue. Add
`nats_config` parameter to Core.__init__. NATS is optional — no config means
the same behavior as before.

- [ ] **Step 1: Write failing tests**

Create `tests/test_core_rewrite.py`:

```python
"""Tests for the rewritten Core._main() with optional NATS."""

import asyncio
import time
from threading import Thread
from unittest.mock import AsyncMock, patch, MagicMock

import numpy as np
import pytest

from tsuchinoko.adaptive.random_in_process import RandomInProcess
from tsuchinoko.core import Core, CoreState
from tsuchinoko.execution.simple import SimpleEngine
from tsuchinoko.nats.config import NATSConfig


def _measure(pos):
    x, y = pos
    return pos, np.sin(x / 30), 0.1, {}


class TestCoreWithoutNATS:
    """Core with no NATS config should work exactly as before."""

    def test_no_nats_by_default(self):
        core = Core()
        assert core._nats_config is None
        assert core._nats_client is None

    def test_experiment_runs_without_nats(self):
        engine = RandomInProcess(dimensionality=2, parameter_bounds=[(0, 100), (0, 100)])
        execution = SimpleEngine(measure_func=_measure)
        core = Core(execution_engine=execution, adaptive_engine=engine, compute_metrics=False)
        core.exit_at = [5]

        thread = Thread(target=core.main)
        thread.start()
        core.state = CoreState.Starting
        thread.join(timeout=30)

        assert not thread.is_alive()
        assert len(core.data) >= 5


class TestCoreWithNATSConfig:
    """Core with NATS config should connect and register."""

    def test_accepts_nats_config(self):
        config = NATSConfig(url="nats://localhost:4222")
        core = Core(nats_config=config)
        assert core._nats_config is config

    def test_event_queue_exists(self):
        core = Core()
        assert hasattr(core, '_event_queue')


class TestEventQueue:
    """Test the event queue mechanism."""

    def test_emit_event_queues(self):
        core = Core()
        core.emit_event("tsuchinoko.state", {"state": "running"})
        assert not core._event_queue.empty()
        subject, payload = core._event_queue.get_nowait()
        assert subject == "tsuchinoko.state"
        assert payload == {"state": "running"}

    def test_emit_event_from_experiment_thread(self):
        engine = RandomInProcess(dimensionality=2, parameter_bounds=[(0, 100), (0, 100)])
        execution = SimpleEngine(measure_func=_measure)
        core = Core(execution_engine=execution, adaptive_engine=engine, compute_metrics=False)
        core.exit_at = [3]

        thread = Thread(target=core.main)
        thread.start()
        core.state = CoreState.Starting
        thread.join(timeout=30)

        # After running, the event queue should have state events
        # (at minimum the exit transition)
        assert not thread.is_alive()
        assert len(core.data) >= 3
```

- [ ] **Step 2: Run tests — verify fail**

Run: `cd /c/Users/rp/PycharmProjects/tsuchinoko-phase1 && .venv/Scripts/python -m pytest tests/test_core_rewrite.py -v --noconftest`

Expected: `TypeError: Core.__init__() got an unexpected keyword argument 'nats_config'`

- [ ] **Step 3: Rewrite Core**

In `tsuchinoko/core/__init__.py`, make these changes:

**Add import at top:**
```python
from queue import Queue, Empty
```

(Replace the existing `from queue import Queue`.)

**Modify `__init__`** — add `nats_config` parameter and event queue:

```python
    def __init__(self,
                 execution_engine: ExecutionEngine = None,
                 adaptive_engine: AdaptiveEngine = None,
                 compute_metrics: bool = True,
                 nats_config=None):
        self.execution_engine = execution_engine
        self.adaptive_engine = adaptive_engine

        self.iteration = 0

        # Initialize state machine
        self._state_machine = self._get_state_machine_class()()
        self._exception_queue = Queue()
        self._forced_position_queue = Queue()
        self._forced_measurement_queue = Queue()
        self._has_fresh_data = True
        self.compute_metrics = compute_metrics
        self.checkpoint_template = 'checkpoint_{n}.yml'
        self.checkpoint_at = []
        self.pause_at = []
        self.stop_at = []
        self.exit_at = []
        self.compute_metrics_at = []

        self.data = Data()

        self.experiment_thread = None

        # NATS (optional)
        self._nats_config = nats_config
        self._nats_client = None
        self._nats_service = None
        self._event_queue = asyncio.Queue()
```

**Add `emit_event` method:**

```python
    def emit_event(self, subject: str, payload: dict) -> None:
        """Queue an event for async publishing. Thread-safe."""
        try:
            self._event_queue.put_nowait((subject, payload))
        except Exception:
            pass  # Queue full or loop not running — drop silently
```

**Replace `_main()`:**

```python
    async def _main(self) -> None:
        # Connect to NATS if configured
        if self._nats_config and self._nats_config.url:
            from tsuchinoko.nats.client import NATSClient
            from tsuchinoko.nats.service import NATSService

            self._nats_client = NATSClient()
            try:
                await self._nats_client.connect(self._nats_config)
                if self._nats_config.lucid_prefix:
                    try:
                        await self._nats_client.authenticate(
                            self._nats_config.lucid_prefix,
                            self._nats_config.app_name,
                            self._nats_config.app_version,
                            self._nats_config.auth_timeout,
                        )
                    except Exception as e:
                        logger.warning(f"LUCID auth failed (continuing without): {e}")
                self._nats_service = NATSService(self, self._nats_client)
                await self._nats_service.start()
            except Exception as e:
                logger.warning(f"NATS connection failed (continuing without): {e}")
                self._nats_client = None

        try:
            while self.state != CoreState.Exiting:
                # Drain outbound events
                await self._drain_events()

                # State transitions
                if self.state == CoreState.Starting:
                    if not len(self.data):
                        self.data = Data(dimensionality=self.adaptive_engine.dimensionality)
                    self.adaptive_engine.reset()
                    self.experiment_thread = threading.Thread(
                        target=self.experiment_loop, daemon=True
                    )
                    self.experiment_thread.start()
                    self.state = CoreState.Running

                elif self.state == CoreState.Pausing:
                    self.state = CoreState.Paused

                elif self.state == CoreState.Resuming:
                    self.state = CoreState.Running

                elif self.state == CoreState.Stopping:
                    self.state = CoreState.Inactive
                    self.data = Data()

                await asyncio.sleep(0.05)
        finally:
            if self._nats_service:
                await self._nats_service.stop()
            if self._nats_client:
                await self._nats_client.close()

    async def _drain_events(self) -> None:
        """Drain the event queue and publish via NATS."""
        while not self._event_queue.empty():
            try:
                subject, payload = self._event_queue.get_nowait()
                if self._nats_client and self._nats_client.is_connected:
                    await self._nats_client.publish(subject, payload)
            except Exception:
                break
```

**Remove `notify_clients`** — it's replaced by `_drain_events`.

- [ ] **Step 4: Run tests — verify pass**

Run: `cd /c/Users/rp/PycharmProjects/tsuchinoko-phase1 && .venv/Scripts/python -m pytest tests/test_core_rewrite.py tests/test_headless.py tests/test_state_machine.py -v --noconftest`

Expected: All tests PASS (core rewrite + existing headless + state machine).

- [ ] **Step 5: Commit**

```bash
git add tsuchinoko/core/__init__.py tests/test_core_rewrite.py
git commit -m "refactor: rewrite Core._main() with async NATS lifecycle and event queue"
```

---

## Task 5: CLI integration

**Files:**
- Modify: `tsuchinoko/cli.py`

Wire up the `run` command with `--nats-url` and `--lucid-prefix` options.

- [ ] **Step 1: Replace cli.py**

```python
"""Headless CLI for Tsuchinoko adaptive experiment service."""

import click
from loguru import logger


@click.group()
@click.version_option()
def main():
    """Tsuchinoko — adaptive experiment service."""
    pass


@main.command()
@click.option("--nats-url", default="", help="NATS broker URL (e.g. nats://localhost:4222). Empty = no NATS.")
@click.option("--lucid-prefix", default="als.7011", help="LUCID instance topic prefix.")
@click.option("--config", "config_path", default=None, type=click.Path(exists=True), help="YAML config file.")
def run(nats_url, lucid_prefix, config_path):
    """Run the Tsuchinoko adaptive experiment service."""
    from tsuchinoko.config import AppConfig, get_config, set_config
    from tsuchinoko.nats.config import NATSConfig
    from tsuchinoko.core import Core

    if config_path:
        import yaml
        with open(config_path) as f:
            raw = yaml.safe_load(f)
        config = AppConfig(**(raw or {}))
    else:
        config = AppConfig()

    # CLI flags override config file
    if nats_url:
        config.nats.url = nats_url
    if lucid_prefix != "als.7011":
        config.nats.lucid_prefix = lucid_prefix

    set_config(config)

    logger.info(f"Tsuchinoko starting (NATS: {'enabled' if config.nats.url else 'disabled'})")
    if config.nats.url:
        logger.info(f"  NATS URL: {config.nats.url}")
        logger.info(f"  LUCID prefix: {config.nats.lucid_prefix}")

    core = Core(nats_config=config.nats)
    try:
        core.main()
    except KeyboardInterrupt:
        logger.info("Shutting down...")


@main.command()
def version():
    """Print version information."""
    from tsuchinoko import __version__
    click.echo(f"tsuchinoko {__version__}")
```

- [ ] **Step 2: Verify CLI runs**

Run: `cd /c/Users/rp/PycharmProjects/tsuchinoko-phase1 && .venv/Scripts/python -m tsuchinoko.cli run --help`

Expected: Shows help with --nats-url and --lucid-prefix options.

- [ ] **Step 3: Commit**

```bash
git add tsuchinoko/cli.py
git commit -m "feat: wire CLI run command with NATS options"
```

---

## Task 6: Integration tests

**Files:**
- Create: `tests/test_nats_integration.py`

Integration tests that require a real NATS broker at `localhost:4222`.
Skipped when NATS is unavailable. Uses a mock LUCID fixture pattern from
lucid-mcp-bridge.

- [ ] **Step 1: Write integration tests**

Create `tests/test_nats_integration.py`:

```python
"""Integration tests requiring a real NATS broker at localhost:4222.

Skipped when NATS is not reachable.
"""

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
from tsuchinoko.nats.service import NATSService

NATS_URL = "nats://localhost:4222"


@pytest.fixture(scope="session")
def nats_url():
    return NATS_URL


@pytest_asyncio.fixture
async def nats_available(nats_url):
    """Skip if NATS is not reachable."""
    try:
        nc = await nats_lib.connect(nats_url)
        await nc.drain()
    except Exception:
        pytest.skip("NATS server not available at localhost:4222")


@pytest_asyncio.fixture
async def mock_lucid(nats_url, nats_available):
    """A mock LUCID instance that auto-approves auth."""
    nc = await nats_lib.connect(nats_url)
    prefix = f"test.lucid.{uuid.uuid4().hex[:8]}"

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


@pytest.mark.asyncio
class TestNATSClientIntegration:
    async def test_connect_and_close(self, nats_url, nats_available):
        client = NATSClient()
        config = NATSConfig(url=nats_url)
        await client.connect(config)
        assert client.is_connected
        await client.close()
        assert not client.is_connected

    async def test_auth_handshake(self, nats_url, mock_lucid):
        client = NATSClient()
        config = NATSConfig(url=nats_url, lucid_prefix=mock_lucid["prefix"])
        await client.connect(config)

        creds = await client.authenticate(
            mock_lucid["prefix"], "tsuchinoko", "test", timeout=5.0
        )
        assert creds["tiled_token"] == "test-token-123"
        assert creds["tiled_url"] == "https://tiled.test.example.com"

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
        """Start a Core with NATSService, send a status request, get reply."""
        engine = RandomInProcess(dimensionality=2, parameter_bounds=[(0, 100), (0, 100)])
        execution = SimpleEngine(measure_func=_measure)

        config = NATSConfig(url=nats_url, lucid_prefix="")
        core = Core(
            execution_engine=execution,
            adaptive_engine=engine,
            compute_metrics=False,
            nats_config=config,
        )
        core.exit_at = [5]

        # Start core in background thread
        thread = Thread(target=core.main)
        thread.start()
        core.state = CoreState.Starting
        await asyncio.sleep(1)  # Let it start and connect

        # Send status request from external client
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
        """Verify Tsuchinoko responds to _tsuchinoko.discover."""
        engine = RandomInProcess(dimensionality=2, parameter_bounds=[(0, 100), (0, 100)])
        execution = SimpleEngine(measure_func=_measure)

        config = NATSConfig(url=nats_url, lucid_prefix="")
        core = Core(
            execution_engine=execution,
            adaptive_engine=engine,
            compute_metrics=False,
            nats_config=config,
        )
        core.exit_at = [3]

        thread = Thread(target=core.main)
        thread.start()
        core.state = CoreState.Starting
        await asyncio.sleep(1)

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
        assert not thread.is_alive()

    async def test_meta_actions(self, nats_url, nats_available):
        """Verify tsuchinoko.meta.actions returns action catalog."""
        engine = RandomInProcess(dimensionality=2, parameter_bounds=[(0, 100), (0, 100)])
        execution = SimpleEngine(measure_func=_measure)

        config = NATSConfig(url=nats_url, lucid_prefix="")
        core = Core(
            execution_engine=execution,
            adaptive_engine=engine,
            compute_metrics=False,
            nats_config=config,
        )
        core.exit_at = [3]

        thread = Thread(target=core.main)
        thread.start()
        core.state = CoreState.Starting
        await asyncio.sleep(1)

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
```

- [ ] **Step 2: Run unit tests (should all pass without broker)**

Run: `cd /c/Users/rp/PycharmProjects/tsuchinoko-phase1 && .venv/Scripts/python -m pytest tests/test_nats_config.py tests/test_nats_client.py tests/test_nats_service.py tests/test_core_rewrite.py tests/test_headless.py tests/test_parameter_tree.py tests/test_gpcam_engine.py tests/test_state_machine.py -v --noconftest`

Expected: All unit tests PASS.

- [ ] **Step 3: Run integration tests (if NATS available)**

Run: `cd /c/Users/rp/PycharmProjects/tsuchinoko-phase1 && .venv/Scripts/python -m pytest tests/test_nats_integration.py -v --noconftest`

Expected: Either all PASS (if NATS running) or all SKIPPED (if not).

- [ ] **Step 4: Commit**

```bash
git add tests/test_nats_integration.py
git commit -m "test: add NATS integration tests with mock LUCID fixture"
```

- [ ] **Step 5: Final full suite**

Run: `cd /c/Users/rp/PycharmProjects/tsuchinoko-phase1 && .venv/Scripts/python -m pytest tests/test_nats_config.py tests/test_nats_client.py tests/test_nats_service.py tests/test_core_rewrite.py tests/test_headless.py tests/test_parameter_tree.py tests/test_gpcam_engine.py tests/test_state_machine.py -v --noconftest`

Expected: All PASS. This is the complete Phase 2 validation.

- [ ] **Step 6: Final commit — update design doc status**

In `docs/design/2026-04-12-phase2-nats-integration.md`, change line 3 from `Draft` to `Phase 2 complete`.

```bash
git add docs/design/2026-04-12-phase2-nats-integration.md
git commit -m "docs: mark Phase 2 (NATS integration) complete"
```

---

## Verification Checklist

After all tasks, verify:

- [ ] `NATSConfig` has correct defaults and integrates with `AppConfig`
- [ ] `NATSClient` connects, authenticates, publishes, subscribes (mocked)
- [ ] `NATSService` registers 8 actions + 3 meta endpoints (11 subscriptions)
- [ ] Each action handler mutates Core state and replies with JSON
- [ ] Discovery response includes instance_id, app_name, prefix, counts
- [ ] `Core(nats_config=None)` works exactly as Phase 1 (all headless tests pass)
- [ ] `Core(nats_config=NATSConfig(url=...))` connects and registers on NATS
- [ ] Event queue: `emit_event()` queues, `_drain_events()` publishes
- [ ] CLI `run --nats-url ... --lucid-prefix ...` works
- [ ] Integration tests pass with real NATS broker (or skip cleanly)
- [ ] All Phase 1 tests still pass (no regressions)
