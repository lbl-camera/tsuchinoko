"""Tests for the NATS experiment.upload_design_code handler."""
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
    client = MagicMock()
    return NATSService(core, client)


@pytest.mark.asyncio
async def test_upload_design_code_happy(monkeypatch, tmp_path):
    svc = _service(monkeypatch, tmp_path)
    code = textwrap.dedent("""
        def acquisition_function(x, gp, **_):
            return [0.0]
    """).strip()
    msg = FakeMsg({"name": "my_ucb", "kind": "acquisition", "code": code})

    await svc._handle_upload_design_code(msg)

    reply = json.loads(msg.respond.call_args.args[0])
    assert reply["status"] == "ok"
    assert reply["ref"] == "user:my_ucb"
    assert reply["path"].endswith("acquisition/my_ucb.py") or reply["path"].endswith("acquisition\\my_ucb.py")
    assert (tmp_path / "user_designs" / "acquisition" / "my_ucb.py").is_file()


@pytest.mark.asyncio
async def test_upload_design_code_syntax_error(monkeypatch, tmp_path):
    svc = _service(monkeypatch, tmp_path)
    msg = FakeMsg({"name": "bad", "kind": "acquisition", "code": "def f(:\n"})

    await svc._handle_upload_design_code(msg)

    reply = json.loads(msg.respond.call_args.args[0])
    assert reply["status"] == "error"
    assert "SyntaxError" in reply["message"] or "invalid syntax" in reply["message"]
    assert not (tmp_path / "user_designs" / "acquisition" / "bad.py").exists()


@pytest.mark.asyncio
async def test_upload_design_code_missing_callable(monkeypatch, tmp_path):
    svc = _service(monkeypatch, tmp_path)
    msg = FakeMsg({"name": "nofn", "kind": "kernel", "code": "x = 1\n"})

    await svc._handle_upload_design_code(msg)

    reply = json.loads(msg.respond.call_args.args[0])
    assert reply["status"] == "error"
    assert "kernel" in reply["message"]


@pytest.mark.asyncio
async def test_upload_design_code_bad_name(monkeypatch, tmp_path):
    svc = _service(monkeypatch, tmp_path)
    msg = FakeMsg({"name": "../escape", "kind": "acquisition", "code": "def acquisition_function(x, gp): pass\n"})

    await svc._handle_upload_design_code(msg)

    reply = json.loads(msg.respond.call_args.args[0])
    assert reply["status"] == "error"
    assert "invalid design name" in reply["message"]
