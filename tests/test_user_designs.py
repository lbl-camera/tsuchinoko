"""Tests for tsuchinoko.nats.user_designs."""
import os
from pathlib import Path

import pytest

from tsuchinoko.nats.user_designs import (
    EXPECTED_CALLABLES,
    KINDS,
    UserDesignError,
    resolve_user_ref,
    user_designs_root,
    validate_name,
    write_design,
)


def test_validate_name_accepts_lowercase_underscore(tmp_path):
    validate_name("my_ucb")
    validate_name("a")
    validate_name("a1")
    validate_name("a_b_c_123")


@pytest.mark.parametrize("bad", [
    "", "1starts_with_digit", "Capital", "has-hyphen",
    "../foo", "foo/bar", "with space", "a" * 64,
])
def test_validate_name_rejects(bad):
    with pytest.raises(UserDesignError):
        validate_name(bad)


def test_user_designs_root_honors_env(monkeypatch, tmp_path):
    monkeypatch.setenv("TSUCHINOKO_USER_DIR", str(tmp_path))
    assert user_designs_root() == tmp_path / "user_designs"


def test_user_designs_root_default(monkeypatch, tmp_path):
    monkeypatch.delenv("TSUCHINOKO_USER_DIR", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Windows
    root = user_designs_root()
    assert root.name == "user_designs"
    assert str(tmp_path) in str(root)


def test_kinds_and_expected_callables_aligned():
    assert set(KINDS) == set(EXPECTED_CALLABLES.keys())
    for kind, name in EXPECTED_CALLABLES.items():
        assert isinstance(name, str) and name


def test_write_design_creates_file_and_returns_ref(monkeypatch, tmp_path):
    monkeypatch.setenv("TSUCHINOKO_USER_DIR", str(tmp_path))
    code = "def acquisition_function(x, gp, **_):\n    return 0.0\n"
    ref, path = write_design("my_ucb", "acquisition", code)
    assert ref == "user:my_ucb"
    assert path == tmp_path / "user_designs" / "acquisition" / "my_ucb.py"
    assert path.read_text() == code


def test_write_design_rejects_syntax_error(monkeypatch, tmp_path):
    monkeypatch.setenv("TSUCHINOKO_USER_DIR", str(tmp_path))
    with pytest.raises(UserDesignError) as excinfo:
        write_design("bad", "acquisition", "def acquisition_function(x, gp):\n    return 1 +\n")
    assert "SyntaxError" in str(excinfo.value) or "invalid syntax" in str(excinfo.value)
    # Must not have created the file
    assert not (tmp_path / "user_designs" / "acquisition" / "bad.py").exists()


def test_write_design_rejects_missing_callable(monkeypatch, tmp_path):
    monkeypatch.setenv("TSUCHINOKO_USER_DIR", str(tmp_path))
    with pytest.raises(UserDesignError) as excinfo:
        write_design("nope", "acquisition", "x = 1\n")
    assert "acquisition_function" in str(excinfo.value)
    assert not (tmp_path / "user_designs" / "acquisition" / "nope.py").exists()


def test_write_design_rejects_unknown_kind(monkeypatch, tmp_path):
    monkeypatch.setenv("TSUCHINOKO_USER_DIR", str(tmp_path))
    with pytest.raises(UserDesignError):
        write_design("x", "totally_made_up", "def foo(): pass\n")


def test_resolve_user_ref_imports_callable(monkeypatch, tmp_path):
    monkeypatch.setenv("TSUCHINOKO_USER_DIR", str(tmp_path))
    code = "def acquisition_function(x, gp, **_):\n    return 'hi'\n"
    write_design("my_aq", "acquisition", code)
    fn = resolve_user_ref("user:my_aq", "acquisition")
    assert callable(fn)
    assert fn(None, None) == "hi"


def test_resolve_user_ref_unknown_name(monkeypatch, tmp_path):
    monkeypatch.setenv("TSUCHINOKO_USER_DIR", str(tmp_path))
    with pytest.raises(UserDesignError):
        resolve_user_ref("user:does_not_exist", "acquisition")


def test_resolve_user_ref_bad_prefix(monkeypatch, tmp_path):
    monkeypatch.setenv("TSUCHINOKO_USER_DIR", str(tmp_path))
    with pytest.raises(UserDesignError):
        resolve_user_ref("builtin:variance", "acquisition")
