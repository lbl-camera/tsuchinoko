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


def test_core_notify_clients_has_type_hints():
    """Verify Core.notify_clients() has proper type annotations."""
    hints = get_type_hints(Core.notify_clients)
    assert 'return' in hints, "Return type should be annotated"


def test_core_update_graph_has_type_hints():
    """Verify Core.update_graph() has proper type annotations."""
    hints = get_type_hints(Core.update_graph)
    assert 'new_graph' in hints, "Parameter 'new_graph' should have type hint"
    assert 'return' in hints, "Return type should be annotated"


def test_core_initialize_data_has_type_hints():
    """Verify Core.initialize_data() has proper type annotations."""
    hints = get_type_hints(Core.initialize_data)
    assert 'x' in hints
    assert 'y' in hints
    assert 'v' in hints
    assert 'return' in hints


def test_core_save_checkpoint_has_type_hints():
    """Verify Core.save_checkpoint() has proper type annotations."""
    hints = get_type_hints(Core.save_checkpoint)
    assert 'directory' in hints, "Parameter 'directory' should have type hint"
    assert 'return' in hints, "Return type should be annotated"


def test_zmqcore_respond_methods_have_type_hints():
    """Verify ZMQCore respond_* methods have proper type annotations."""
    from tsuchinoko.core.messages import (
        FullDataRequest, PartialDataRequest, PushDataRequest, StartRequest,
        StopRequest, PauseRequest, StateRequest, GetParametersRequest,
        SetParameterRequest, MeasureRequest, ConnectRequest, ReplayRequest
    )

    respond_methods = [
        ('respond_FullDataRequest', FullDataRequest),
        ('respond_PartialDataRequest', PartialDataRequest),
        ('respond_PushDataRequest', PushDataRequest),
        ('respond_StartRequest', StartRequest),
        ('respond_StopRequest', StopRequest),
        ('respond_PauseRequest', PauseRequest),
        ('respond_StateRequest', StateRequest),
        ('respond_GetParametersRequest', GetParametersRequest),
        ('respond_SetParameterRequest', SetParameterRequest),
        ('respond_MeasureRequest', MeasureRequest),
        ('respond_ConnectRequest', ConnectRequest),
        ('respond_ReplayRequest', ReplayRequest),
    ]

    for method_name, request_type in respond_methods:
        method = getattr(ZMQCore, method_name)
        hints = get_type_hints(method)
        assert 'request' in hints, f"{method_name} should have 'request' parameter typed"
        assert 'return' in hints, f"{method_name} should have return type"


def test_core_state_has_docstring():
    """Verify CoreState enum has docstring."""
    assert CoreState.__doc__ is not None
    assert len(CoreState.__doc__) > 10


def test_core_class_has_docstring():
    """Verify Core class has docstring."""
    assert Core.__doc__ is not None
    assert len(Core.__doc__) > 20


def test_core_init_has_docstring():
    """Verify Core.__init__ has docstring."""
    assert Core.__init__.__doc__ is not None


def test_core_methods_have_docstrings():
    """Verify Core public methods have docstrings."""
    methods = [
        'set_execution_engine', 'set_adaptive_engine', 'main',
        'experiment_loop', 'experiment_iteration', 'update_graph',
        'initialize_data', 'save_checkpoint'
    ]
    for method_name in methods:
        method = getattr(Core, method_name)
        assert method.__doc__ is not None, f"{method_name} should have docstring"


def test_zmqcore_has_docstring():
    """Verify ZMQCore class has docstring."""
    assert ZMQCore.__doc__ is not None
    assert len(ZMQCore.__doc__) > 20


def test_zmqcore_notify_clients_has_docstring():
    """Verify ZMQCore.notify_clients has docstring."""
    assert ZMQCore.notify_clients.__doc__ is not None