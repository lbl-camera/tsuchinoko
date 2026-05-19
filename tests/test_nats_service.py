"""Unit tests for NATSService using mock Core and NATSClient."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from tsuchinoko.nats.service import NATSService, ACTIONS, EVENTS


def _make_mock_core():
    from tsuchinoko.core import CoreState
    core = MagicMock()
    core.state = CoreState.Inactive
    core.data = MagicMock()
    core.data._completed_iterations = 0
    core.data.__len__ = MagicMock(return_value=0)
    core.adaptive_engine = MagicMock()
    core.adaptive_engine.parameters = MagicMock()
    core.adaptive_engine.parameters.saveState.return_value = {"name": "top", "children": []}
    return core


def _make_mock_client():
    client = AsyncMock()
    mock_sub = MagicMock()
    mock_sub.unsubscribe = AsyncMock()
    client.subscribe = AsyncMock(return_value=mock_sub)
    client.publish = AsyncMock()
    return client


def _make_nats_msg(data: dict, reply: str = "_INBOX.test"):
    msg = MagicMock()
    msg.data = json.dumps(data).encode()
    msg.reply = reply
    msg.respond = AsyncMock()
    return msg


class TestServiceLifecycle:
    async def test_start_registers_all_subscriptions(self):
        core = _make_mock_core()
        client = _make_mock_client()
        service = NATSService(core, client)

        await service.start()

        # action handlers + discover + meta.actions + meta.events
        assert client.subscribe.await_count == len(ACTIONS) + 3
        assert len(service._subscriptions) == len(ACTIONS) + 3

    async def test_stop_unsubscribes_all(self):
        core = _make_mock_core()
        # Give each subscribe call its own distinct mock sub
        client = AsyncMock()
        client.publish = AsyncMock()
        subs_created = []

        def make_sub(*args, **kwargs):
            sub = MagicMock()
            sub.unsubscribe = AsyncMock()
            subs_created.append(sub)
            return sub

        client.subscribe = AsyncMock(side_effect=make_sub)
        service = NATSService(core, client)

        await service.start()
        await service.stop()

        assert len(subs_created) == len(ACTIONS) + 3
        for sub in subs_created:
            sub.unsubscribe.assert_awaited_once()
        assert len(service._subscriptions) == 0


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
        assert reply["state"] == "Starting"

    async def test_handle_pause(self):
        from tsuchinoko.core import CoreState
        core = _make_mock_core()
        client = _make_mock_client()
        service = NATSService(core, client)

        msg = _make_nats_msg({})
        await service._handle_pause(msg)

        assert core.state == CoreState.Pausing
        reply = json.loads(msg.respond.call_args[0][0])
        assert reply["status"] == "ok"
        assert reply["state"] == "Pausing"

    async def test_handle_resume(self):
        from tsuchinoko.core import CoreState
        core = _make_mock_core()
        client = _make_mock_client()
        service = NATSService(core, client)

        msg = _make_nats_msg({})
        await service._handle_resume(msg)

        assert core.state == CoreState.Resuming
        reply = json.loads(msg.respond.call_args[0][0])
        assert reply["status"] == "ok"
        assert reply["state"] == "Resuming"

    async def test_handle_stop(self):
        from tsuchinoko.core import CoreState
        core = _make_mock_core()
        client = _make_mock_client()
        service = NATSService(core, client)

        msg = _make_nats_msg({})
        await service._handle_stop(msg)

        assert core.state == CoreState.Stopping
        reply = json.loads(msg.respond.call_args[0][0])
        assert reply["status"] == "ok"
        assert reply["state"] == "Stopping"

    async def test_handle_status(self):
        from tsuchinoko.core import CoreState
        core = _make_mock_core()
        core.state = CoreState.Running
        core.data._completed_iterations = 5
        core.data.__len__ = MagicMock(return_value=42)
        client = _make_mock_client()
        service = NATSService(core, client)

        msg = _make_nats_msg({})
        await service._handle_status(msg)

        reply = json.loads(msg.respond.call_args[0][0])
        assert reply["status"] == "ok"
        assert reply["state"] == "Running"
        assert reply["iteration"] == 5
        assert reply["data_count"] == 42

    async def test_handle_get_parameters(self):
        core = _make_mock_core()
        client = _make_mock_client()
        service = NATSService(core, client)

        msg = _make_nats_msg({})
        await service._handle_get_parameters(msg)

        core.adaptive_engine.parameters.saveState.assert_called_once()
        reply = json.loads(msg.respond.call_args[0][0])
        assert reply["status"] == "ok"
        assert reply["parameters"] == {"name": "top", "children": []}

    async def test_handle_set_parameter(self):
        core = _make_mock_core()
        client = _make_mock_client()
        service = NATSService(core, client)

        msg = _make_nats_msg({"path": ["gpcam", "noise"], "value": 0.01})
        await service._handle_set_parameter(msg)

        core.adaptive_engine.parameters.child.assert_called_once_with("gpcam", "noise")
        core.adaptive_engine.parameters.child().setValue.assert_called_once_with(0.01)
        reply = json.loads(msg.respond.call_args[0][0])
        assert reply["status"] == "ok"

    async def test_handler_error_returns_error_reply(self):
        core = _make_mock_core()
        # Make saveState raise
        core.adaptive_engine.parameters.saveState.side_effect = RuntimeError("boom")
        client = _make_mock_client()
        service = NATSService(core, client)

        msg = _make_nats_msg({})
        await service._handle_get_parameters(msg)

        reply = json.loads(msg.respond.call_args[0][0])
        assert reply["status"] == "error"
        assert "boom" in reply["message"]


class TestDiscovery:
    async def test_discover_response(self):
        from tsuchinoko.core import CoreState
        core = _make_mock_core()
        core.state = CoreState.Inactive
        client = _make_mock_client()
        service = NATSService(core, client)

        msg = _make_nats_msg({})
        await service._handle_discover(msg)

        reply = json.loads(msg.respond.call_args[0][0])
        assert reply["app_name"] == "tsuchinoko"
        assert reply["prefix"] == "tsuchinoko"
        assert reply["actions_count"] == len(ACTIONS)
        assert reply["events_count"] == len(EVENTS)
        assert "instance_id" in reply
        assert len(reply["instance_id"]) == 36  # UUID format

    async def test_meta_actions(self):
        core = _make_mock_core()
        client = _make_mock_client()
        service = NATSService(core, client)

        msg = _make_nats_msg({})
        await service._handle_meta_actions(msg)

        reply = json.loads(msg.respond.call_args[0][0])
        assert len(reply["actions"]) == len(ACTIONS)
        suffixes = {a["suffix"] for a in reply["actions"]}
        assert "experiment.configure" in suffixes
        assert "experiment.start" in suffixes
        assert "experiment.pause" in suffixes
        assert "experiment.resume" in suffixes
        assert "experiment.stop" in suffixes
        assert "engine.set_parameter" in suffixes
        assert "engine.get_parameters" in suffixes
        assert "status" in suffixes

    async def test_meta_events(self):
        core = _make_mock_core()
        client = _make_mock_client()
        service = NATSService(core, client)

        msg = _make_nats_msg({})
        await service._handle_meta_events(msg)

        reply = json.loads(msg.respond.call_args[0][0])
        assert len(reply["events"]) == len(EVENTS)
        suffixes = {e["suffix"] for e in reply["events"]}
        assert "state" in suffixes
        assert "targets" in suffixes
        assert "gp.updated" in suffixes
        assert "error" in suffixes
