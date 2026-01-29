# tests/test_core_types.py
"""Type annotation tests for Core module."""
from typing import get_type_hints
from tsuchinoko.core import Core, ZMQCore, CoreState


def test_core_main_has_type_hints():
    """Verify Core.main() has proper type annotations."""
    hints = get_type_hints(Core.main)
    assert 'debug' in hints, "Parameter 'debug' should have type hint"
    assert 'return' in hints, "Return type should be annotated"


def test_core_main_async_has_type_hints():
    """Verify Core._main() has proper type annotations."""
    hints = get_type_hints(Core._main)
    assert 'min_response_sleep' in hints, "Parameter 'min_response_sleep' should have type hint"
    assert 'return' in hints, "Return type should be annotated"


def test_core_experiment_loop_has_type_hints():
    """Verify Core.experiment_loop() has proper type annotations."""
    hints = get_type_hints(Core.experiment_loop)
    assert 'return' in hints, "Return type should be annotated"


def test_core_experiment_iteration_has_type_hints():
    """Verify Core.experiment_iteration() has proper type annotations."""
    hints = get_type_hints(Core.experiment_iteration)
    assert 'return' in hints, "Return type should be annotated"
