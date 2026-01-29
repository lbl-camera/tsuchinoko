# tests/test_adaptive_types.py
"""Type annotation tests for Adaptive module."""
from typing import get_type_hints
from tsuchinoko.adaptive import Data


def test_data_inject_new_has_type_hints():
    """Verify Data.inject_new() has proper type annotations."""
    hints = get_type_hints(Data.inject_new)
    assert 'data' in hints, "Parameter 'data' should have type hint"
    assert 'return' in hints, "Return type should be annotated"


def test_data_as_dict_has_type_hints():
    """Verify Data.as_dict() has proper type annotations."""
    hints = get_type_hints(Data.as_dict)
    assert 'return' in hints, "Return type should be annotated"


def test_data_extend_has_type_hints():
    """Verify Data.extend() has proper type annotations."""
    hints = get_type_hints(Data.extend)
    assert 'data' in hints, "Parameter 'data' should have type hint"
    assert 'return' in hints, "Return type should be annotated"


def test_data_dunder_methods_have_type_hints():
    """Verify Data dunder methods have proper type annotations."""
    # __len__
    hints = get_type_hints(Data.__len__)
    assert 'return' in hints

    # __bool__
    hints = get_type_hints(Data.__bool__)
    assert 'return' in hints

    # __contains__
    hints = get_type_hints(Data.__contains__)
    assert 'item' in hints
    assert 'return' in hints
