# tests/test_displays_types.py
"""Type annotation tests for Display widgets."""
from typing import get_type_hints
from tsuchinoko.widgets.displays import LogHandler, Log, Configuration, StateManager, GraphManager


def test_log_handler_emit_has_type_hints():
    """Verify LogHandler.emit() has proper type annotations."""
    hints = get_type_hints(LogHandler.emit)
    assert 'record' in hints
    assert 'return' in hints


def test_log_handler_sink_has_type_hints():
    """Verify LogHandler.sink() has proper type annotations."""
    hints = get_type_hints(LogHandler.sink)
    assert 'message' in hints
    assert 'return' in hints


def test_log_exception_has_type_hints():
    """Verify Log.log_exception() has proper type annotations."""
    hints = get_type_hints(Log.log_exception)
    assert 'ex' in hints
    assert 'return' in hints
