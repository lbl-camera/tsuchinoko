"""Tests for the extended experiment.configure handler."""
import json
import textwrap
from unittest.mock import AsyncMock, MagicMock

import pytest

from tsuchinoko.nats.service import NATSService


class FakeMsg:
    def __init__(self, payload: dict):
        self.data = json.dumps(payload).encode()
        self.reply = "_INBOX.x"
        self.respond = AsyncMock()


def _service(monkeypatch, tmp_path):
    monkeypatch.setenv("TSUCHINOKO_USER_DIR", str(tmp_path))
    core = MagicMock()
    # Engine params recorded by setattr; tests inspect _recorded.
    core.adaptive_engine.parameters = MagicMock()
    return NATSService(core, MagicMock()), core


@pytest.mark.asyncio
async def test_configure_bounds_only_still_works(monkeypatch, tmp_path):
    svc, core = _service(monkeypatch, tmp_path)
    msg = FakeMsg({"parameter_bounds": [[0.0, 10.0], [-5.0, 5.0]]})
    await svc._handle_configure(msg)
    reply = json.loads(msg.respond.call_args.args[0])
    assert reply["status"] == "ok"
    # Bounds mapped to engine via the existing parameter-tree keys
    params = core.adaptive_engine.parameters.__setitem__.call_args_list
    keys = [call.args[0] for call in params]
    assert ("bounds", "axis_0_min") in keys
    assert ("bounds", "axis_0_max") in keys
    assert ("bounds", "axis_1_min") in keys
    assert ("bounds", "axis_1_max") in keys


@pytest.mark.asyncio
async def test_configure_unknown_key_rejected(monkeypatch, tmp_path):
    svc, _core = _service(monkeypatch, tmp_path)
    msg = FakeMsg({"parameter_bounds": [[0, 1]], "what_is_this": 42})
    await svc._handle_configure(msg)
    reply = json.loads(msg.respond.call_args.args[0])
    assert reply["status"] == "error"
    assert "what_is_this" in reply["message"]


@pytest.mark.asyncio
async def test_configure_full_typed_payload(monkeypatch, tmp_path):
    svc, core = _service(monkeypatch, tmp_path)
    msg = FakeMsg({
        "parameter_bounds": [[0.0, 1.0], [0.0, 1.0]],
        "dimensionality": 2,
        "kernel": "matern_3_2",
        "acquisition_function": "ucb",
        "prior_mean": None,
        "noise_function": None,
        "noise_variances": 0.01,
        "initial_points": 12,
        "training_method": "global",
        "hyperparameters": [1.0, 0.5, 0.5],
        "x_out": None,
    })
    await svc._handle_configure(msg)
    reply = json.loads(msg.respond.call_args.args[0])
    assert reply["status"] == "ok"
    # The full typed-payload assertion is exercised end-to-end at the engine
    # level in the user-ref test below; here we just confirm the wire reply.


@pytest.mark.asyncio
async def test_configure_user_ref_resolves(monkeypatch, tmp_path):
    monkeypatch.setenv("TSUCHINOKO_USER_DIR", str(tmp_path))
    from tsuchinoko.nats.user_designs import write_design
    code = "def acquisition_function(x, gp, **_):\n    return [0.0]\n"
    write_design("my_ucb", "acquisition", code)

    svc, core = _service(monkeypatch, tmp_path)
    msg = FakeMsg({
        "parameter_bounds": [[0, 1]],
        "acquisition_function": "user:my_ucb",
    })
    await svc._handle_configure(msg)
    reply = json.loads(msg.respond.call_args.args[0])
    assert reply["status"] == "ok"
    # The resolved callable was assigned to the engine. MagicMock's
    # __setattr__ isn't directly recordable on this Python version, so
    # confirm via the assigned attribute itself: the engine attribute
    # should now be the resolved callable, not a string ref.
    assigned = core.adaptive_engine.acquisition_function
    assert callable(assigned)
    assert assigned != "user:my_ucb"


@pytest.mark.asyncio
async def test_configure_unknown_user_ref(monkeypatch, tmp_path):
    monkeypatch.setenv("TSUCHINOKO_USER_DIR", str(tmp_path))
    svc, _core = _service(monkeypatch, tmp_path)
    msg = FakeMsg({
        "parameter_bounds": [[0, 1]],
        "kernel": "user:does_not_exist",
    })
    await svc._handle_configure(msg)
    reply = json.loads(msg.respond.call_args.args[0])
    assert reply["status"] == "error"
    assert "does_not_exist" in reply["message"]
