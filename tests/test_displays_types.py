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


def test_configuration_methods_have_type_hints():
    """Verify Configuration methods have proper type annotations."""
    hints = get_type_hints(Configuration.request_parameters)
    assert 'return' in hints

    hints = get_type_hints(Configuration.update_parameters)
    assert 'state' in hints
    assert 'return' in hints

    hints = get_type_hints(Configuration.push_changes)
    assert 'sender' in hints
    assert 'changes' in hints
    assert 'return' in hints


def test_state_manager_methods_have_type_hints():
    """Verify StateManager methods have proper type annotations."""
    hints = get_type_hints(StateManager.update_state)
    assert 'state' in hints
    assert 'compute_metrics' in hints
    assert 'return' in hints

    hints = get_type_hints(StateManager.update_compute_metrics)
    assert 'compute_metrics' in hints
    assert 'return' in hints


def test_graph_manager_methods_have_type_hints():
    """Verify GraphManager methods have proper type annotations."""
    from tsuchinoko.graphs import Graph

    hints = get_type_hints(GraphManager.set_graphs)
    assert 'graphs' in hints
    assert 'return' in hints

    hints = get_type_hints(GraphManager.register_graph)
    assert 'graph' in hints
    assert 'return' in hints

    hints = get_type_hints(GraphManager.update_graphs)
    assert 'data' in hints
    assert 'last_data_size' in hints
    assert 'return' in hints

    hints = get_type_hints(GraphManager.clear)
    assert 'return' in hints

    hints = get_type_hints(GraphManager.reset)
    assert 'return' in hints