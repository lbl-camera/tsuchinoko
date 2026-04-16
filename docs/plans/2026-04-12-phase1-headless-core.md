# Phase 1: Extract Headless Core — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make tsuchinoko runnable as a headless service with no Qt/PySide6 dependency, preserving all adaptive engine and core experiment loop functionality.

**Architecture:** Replace pyqtgraph ParameterTree with a pure-Python drop-in replacement (`parameters/tree.py`). Remove ZMQCore, graph system, and Qt imports from core modules. Qt-dependent code stays in the repo but is not imported by default — `import tsuchinoko` works without PySide6 installed.

**Tech Stack:** Python 3.9+, Pydantic >=2.0, transitions, gpCAM ~8.2.3, fvgp ~4.6.5, numpy, click, loguru, pytest

**Design doc:** `docs/design/2026-04-12-tsuchinoko-rescope.md`

---

## File Structure

### New files
| File | Responsibility |
|------|----------------|
| `tsuchinoko/parameters/tree.py` | Lightweight ParameterTree (replaces pyqtgraph.parametertree) |
| `tsuchinoko/core/zmq_core.py` | ZMQCore moved here for backward compatibility |
| `tsuchinoko/cli.py` | Headless CLI entry point |
| `tests/test_parameter_tree.py` | Tests for ParameterTree |
| `tests/test_headless.py` | End-to-end headless experiment |

### Modified files
| File | Changes |
|------|---------|
| `tsuchinoko/adaptive/__init__.py` | Remove pyqtgraph/Graph imports, make update_metrics non-abstract |
| `tsuchinoko/adaptive/gpCAM_in_process.py` | Use parameters.tree, remove graph instantiation |
| `tsuchinoko/adaptive/random_in_process.py` | Use parameters.tree |
| `tsuchinoko/adaptive/grid.py` | Use parameters.tree, remove graph imports |
| `tsuchinoko/adaptive/fvgp_gpCAM_in_process.py` | Remove graph imports |
| `tsuchinoko/core/__init__.py` | Remove ZMQCore, Graph imports, message imports |
| `tsuchinoko/parameters/__init__.py` | Wrap Qt imports in try/except |
| `tsuchinoko/__init__.py` | Remove top-level Qt imports |
| `tsuchinoko/config.py` | Remove NetworkConfig, UIConfig (ZMQ/Qt-specific) |
| `pyproject.toml` | Move Qt deps to optional `[gui]` extra |
| `tests/conftest.py` | Use Core instead of ZMQCore, skip Qt-dependent fixtures |
| `tests/test_gpcam_engine.py` | Remove graph-related assertions |

### Unchanged (no Qt deps)
`core/state_machine.py`, `core/messages.py`, `execution/__init__.py`, `execution/simple.py`, `execution/threaded_in_process.py`, `adaptive/acquisition_functions.py`, `adaptive/_adaptive.py`, `utils/mutex.py`, `utils/logging.py`, `config.py` (CoreConfig stays)

### Legacy (kept, not imported by default)
`widgets/`, `graphs/`, `graphics_items/`, `network/`, `patches/` — still importable if Qt is installed

---

## Task 1: Lightweight ParameterTree

**Files:**
- Create: `tsuchinoko/parameters/tree.py`
- Create: `tests/test_parameter_tree.py`

This replaces pyqtgraph's `Parameter`, `SimpleParameter`, `GroupParameter`, `ListParameter`, and the custom `TrainingParameter` with pure-Python equivalents. The API surface matches exactly what the adaptive engines use.

- [ ] **Step 1: Write failing tests**

Create `tests/test_parameter_tree.py`:

```python
"""Tests for the lightweight ParameterTree (pyqtgraph replacement)."""

import pytest

from tsuchinoko.parameters.tree import (
    Parameter, SimpleParameter, GroupParameter, ListParameter, TrainingParameter,
)


class TestParameter:
    """Core Parameter node functionality."""

    def test_create_with_value(self):
        p = Parameter(name='x', value=42, type='int')
        assert p.value() == 42
        assert p.name == 'x'

    def test_set_value(self):
        p = Parameter(name='x', value=1)
        p.setValue(2)
        assert p.value() == 2

    def test_children(self):
        child1 = Parameter(name='a', value=1)
        child2 = Parameter(name='b', value=2)
        parent = Parameter(name='top', children=[child1, child2])
        assert parent.hasChildren()
        assert len(parent.children()) == 2
        assert parent.child('a').value() == 1
        assert parent.child('b').value() == 2

    def test_no_children(self):
        p = Parameter(name='leaf', value=5)
        assert not p.hasChildren()
        assert p.children() == []

    def test_getitem_tuple_path(self):
        inner = Parameter(name='inner', value=10)
        group = Parameter(name='group', children=[inner])
        top = Parameter(name='top', children=[group])
        assert top[('group', 'inner')] == 10

    def test_setitem_tuple_path(self):
        inner = Parameter(name='inner', value=10)
        group = Parameter(name='group', children=[inner])
        top = Parameter(name='top', children=[group])
        top[('group', 'inner')] = 20
        assert top[('group', 'inner')] == 20

    def test_getitem_string_key(self):
        child = Parameter(name='n', value=5)
        top = Parameter(name='top', children=[child])
        assert top['n'] == 5

    def test_setitem_string_key(self):
        child = Parameter(name='n', value=5)
        top = Parameter(name='top', children=[child])
        top['n'] = 10
        assert top['n'] == 10

    def test_getitem_missing_key(self):
        top = Parameter(name='top')
        with pytest.raises(KeyError):
            _ = top['missing']

    def test_signal_fires_on_set(self):
        received = []
        p = Parameter(name='x', value=1)
        p.sigValueChanged.connect(lambda param, val: received.append((param, val)))
        p.setValue(42)
        assert len(received) == 1
        assert received[0] == (p, 42)

    def test_block_signal(self):
        received = []
        callback = lambda param, val: received.append(val)
        p = Parameter(name='x', value=1)
        p.sigValueChanged.connect(callback)
        p.setValue(42, blockSignal=callback)
        assert len(received) == 0  # blocked

    def test_block_signal_only_blocks_specified(self):
        received_a = []
        received_b = []
        cb_a = lambda param, val: received_a.append(val)
        cb_b = lambda param, val: received_b.append(val)
        p = Parameter(name='x', value=1)
        p.sigValueChanged.connect(cb_a)
        p.sigValueChanged.connect(cb_b)
        p.setValue(42, blockSignal=cb_a)
        assert len(received_a) == 0  # blocked
        assert received_b == [42]    # not blocked

    def test_save_state(self):
        child = Parameter(name='a', value=1)
        parent = Parameter(name='top', children=[child])
        state = parent.saveState()
        assert state['name'] == 'top'
        assert 'children' in state
        assert state['children'][0]['name'] == 'a'
        assert state['children'][0]['value'] == 1

    def test_add_child_from_dict(self):
        parent = Parameter(name='top')
        parent.addChild({'name': 'new', 'value': 99, 'type': 'int'})
        assert parent['new'] == 99

    def test_aliases(self):
        """SimpleParameter and GroupParameter are aliases for Parameter."""
        assert SimpleParameter is Parameter
        assert GroupParameter is Parameter


class TestListParameter:
    """ListParameter with constrained values."""

    def test_default_value(self):
        p = ListParameter(name='method', limits=['a', 'b', 'c'], default='b')
        assert p.value() == 'b'
        assert p.limits == ['a', 'b', 'c']

    def test_set_valid_value(self):
        p = ListParameter(name='method', limits=['a', 'b', 'c'], default='a')
        p.setValue('c')
        assert p.value() == 'c'

    def test_set_invalid_value_raises(self):
        p = ListParameter(name='method', limits=['a', 'b', 'c'], default='a')
        with pytest.raises(ValueError):
            p.setValue('d')


class TestTrainingParameter:
    """TrainingParameter with dynamic child addition."""

    def test_initial_children(self):
        tp = TrainingParameter(name='training', children=[
            Parameter(name='item1', value=20, type='int'),
        ])
        assert len(tp.children()) == 1
        assert tp.children()[0].value() == 20

    def test_add_new(self):
        tp = TrainingParameter(name='training', children=[
            Parameter(name='item1', value=20, type='int'),
        ])
        tp.addNew()
        assert len(tp.children()) == 2
        assert tp.children()[1].value() == 1  # default

    def test_iterate_children_values(self):
        tp = TrainingParameter(name='training', children=[
            Parameter(name='a', value=20, type='int'),
            Parameter(name='b', value=50, type='int'),
        ])
        values = set(child.value() for child in tp.children())
        assert values == {20, 50}


class TestEngineParameterPattern:
    """Integration: test the exact access patterns used by adaptive engines."""

    def test_gpcam_parameter_layout(self):
        """Mimics GPCAMInProcessEngine's parameter tree structure."""
        top = Parameter(name='top', children=[
            ListParameter(title='Method', name='method',
                          limits=['global', 'local', 'hgdl'], default='global'),
            ListParameter(title='Acquisition Function', name='acquisition_function',
                          limits=['variance', 'ucb'], default='variance'),
            Parameter(title='Queue Length', name='n', value=1, type='int'),
            Parameter(title='Pop Size', name='pop_size', value=20, type='int'),
            Parameter(title='Tolerance', name='tol', value=1e-6, type='float'),
            Parameter(name='bounds', children=[
                Parameter(name='axis_0_min', value=0.0, type='float'),
                Parameter(name='axis_0_max', value=100.0, type='float'),
            ]),
            Parameter(name='hyperparameters', children=[
                Parameter(name='hyperparameter_0', value=100.0, type='float'),
                Parameter(name='hyperparameter_0_min', value=0.1, type='float'),
                Parameter(name='hyperparameter_0_max', value=1e5, type='float'),
            ]),
            TrainingParameter(name='global_training', children=[
                Parameter(name='t1', value=20, type='int'),
                Parameter(name='t2', value=50, type='int'),
            ]),
        ])

        # String key access
        assert top['n'] == 1
        assert top['method'] == 'global'
        assert top['acquisition_function'] == 'variance'

        # Tuple path access
        assert top[('bounds', 'axis_0_min')] == 0.0
        assert top[('bounds', 'axis_0_max')] == 100.0
        assert top[('hyperparameters', 'hyperparameter_0')] == 100.0

        # Set via tuple path
        top[('bounds', 'axis_0_min')] = 5.0
        assert top[('bounds', 'axis_0_min')] == 5.0

        # Set via string key
        top['n'] = 3
        assert top['n'] == 3

        # Child navigation
        hp = top.child('hyperparameters', 'hyperparameter_0')
        hp.setValue(200.0)
        assert top[('hyperparameters', 'hyperparameter_0')] == 200.0

        # Training schedule iteration
        train_values = set(c.value() for c in top.child('global_training').children())
        assert train_values == {20, 50}

        # Save state
        state = top.saveState()
        assert state['name'] == 'top'
        assert len(state['children']) == 8

        # hasChildren
        assert top.hasChildren()
        assert top.child('bounds').hasChildren()
```

- [ ] **Step 2: Run tests — verify they fail**

Run: `cd /c/Users/rp/PycharmProjects/tsuchinoko && .venv/Scripts/python -m pytest tests/test_parameter_tree.py -v`

Expected: `ModuleNotFoundError: No module named 'tsuchinoko.parameters.tree'`

- [ ] **Step 3: Implement ParameterTree**

Create `tsuchinoko/parameters/tree.py`:

```python
"""Lightweight parameter tree, API-compatible with pyqtgraph.parametertree.

Provides the same hierarchical parameter access patterns used by
adaptive engines, without any Qt dependency.
"""

import uuid


class Signal:
    """Minimal callback signal (replaces Qt Signal)."""

    def __init__(self):
        self._callbacks = []

    def connect(self, callback):
        self._callbacks.append(callback)

    def disconnect(self, callback=None):
        if callback:
            self._callbacks.remove(callback)
        else:
            self._callbacks.clear()

    def emit(self, *args, exclude=None):
        for cb in self._callbacks:
            if cb is not exclude:
                cb(*args)


class Parameter:
    """A tree node holding a value and/or children.

    Drop-in replacement for pyqtgraph's SimpleParameter and GroupParameter.
    Supports the same dict-like access patterns:
        param[('group', 'key')]     → navigate path, return value
        param['key']                → direct child value
        param.child('a', 'b')       → navigate to child node
        param.sigValueChanged       → callback signal on setValue
    """

    def __init__(self, name='', title='', value=None, type=None,
                 children=None, removable=False, renamable=False, **kwargs):
        self.name = name
        self.title = title or name
        self._value = value
        self.type = type
        self._children_map = {}
        self._children_list = []
        self.sigValueChanged = Signal()

        if children:
            for child in children:
                self.addChild(child)

    def value(self):
        """Return this parameter's value."""
        return self._value

    def setValue(self, value, blockSignal=None):
        """Set value and notify callbacks (except blockSignal)."""
        self._value = value
        self.sigValueChanged.emit(self, value, exclude=blockSignal)

    def child(self, *path):
        """Navigate to a child by name path."""
        node = self
        for key in path:
            node = node._children_map[key]
        return node

    def children(self):
        """Return list of direct children (insertion order)."""
        return list(self._children_list)

    def hasChildren(self):
        """Return True if this parameter has children."""
        return bool(self._children_list)

    def addChild(self, child):
        """Add a child Parameter (or create one from a dict)."""
        if isinstance(child, dict):
            child = Parameter(**child)
        self._children_map[child.name] = child
        self._children_list.append(child)
        return child

    def __getitem__(self, key):
        if isinstance(key, tuple):
            return self.child(*key).value()
        return self._children_map[key].value()

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            self.child(*key).setValue(value)
        else:
            self._children_map[key].setValue(value)

    def saveState(self):
        """Serialize this parameter and its children to a dict."""
        state = {'name': self.name}
        if self._value is not None:
            state['value'] = self._value
        if self._children_list:
            state['children'] = [c.saveState() for c in self._children_list]
        return state


# Aliases for pyqtgraph API compatibility
SimpleParameter = Parameter
GroupParameter = Parameter


class ListParameter(Parameter):
    """Parameter with constrained allowed values."""

    def __init__(self, limits=None, default=None, **kwargs):
        super().__init__(value=default, **kwargs)
        self.limits = limits or []

    def setValue(self, value, blockSignal=None):
        if self.limits and value not in self.limits:
            raise ValueError(f"{value!r} not in allowed values: {self.limits}")
        super().setValue(value, blockSignal=blockSignal)


class TrainingParameter(Parameter):
    """Dynamic parameter group for training schedules."""

    def __init__(self, addText='', **kwargs):
        super().__init__(**kwargs)
        self.addText = addText

    def addNew(self):
        self.addChild(Parameter(
            name=str(uuid.uuid4()), title='N=', type='int', value=1
        ))
```

- [ ] **Step 4: Run tests — verify they pass**

Run: `cd /c/Users/rp/PycharmProjects/tsuchinoko && .venv/Scripts/python -m pytest tests/test_parameter_tree.py -v`

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tsuchinoko/parameters/tree.py tests/test_parameter_tree.py
git commit -m "feat: add lightweight ParameterTree (pyqtgraph replacement)"
```

---

## Task 2: Decouple adaptive module from Qt

**Files:**
- Modify: `tsuchinoko/adaptive/__init__.py`
- Modify: `tsuchinoko/parameters/__init__.py`

- [ ] **Step 1: Make parameters/__init__.py safe without Qt**

Wrap the pyqtgraph imports so importing `tsuchinoko.parameters.tree` doesn't trigger a Qt import failure.

In `tsuchinoko/parameters/__init__.py`, wrap the entire body in try/except:

```python
"""Parameter types for Tsuchinoko.

Qt-based parameter types (for the GUI) are registered when PySide6 is available.
Headless code should import from tsuchinoko.parameters.tree instead.
"""

try:
    import uuid

    from pyqtgraph.parametertree import registerParameterType
    from pyqtgraph.parametertree.parameterTypes import GroupParameter

    class TrainingParameter(GroupParameter):
        def __init__(self, **kwargs):
            kwargs['type'] = 'training'
            super(TrainingParameter, self).__init__(**kwargs)

        def addNew(self):
            self.addChild(dict(name=str(uuid.uuid4()), title='N=', type='int', value=1, removable=True, renamable=False))

    registerParameterType('training', TrainingParameter)
except ImportError:
    pass
```

- [ ] **Step 2: Remove Qt/Graph imports from adaptive Engine base**

In `tsuchinoko/adaptive/__init__.py`:

Remove these lines:
```python
from pyqtgraph.parametertree import Parameter
from tsuchinoko.graphs import Graph
```

Change the `Engine` class — remove `parameters` and `graphs` class attributes, make `update_metrics` a non-abstract default no-op:

```python
class Engine(ABC):
    """
    The Adaptive Engine base class. This component is generally to be responsible for determining future measurement targets.
    """

    dimensionality: int = None
    last_position: tuple = None

    @abstractmethod
    def update_measurements(self, data: Data):
        """Update internal variables with the provided new data."""
        ...

    @abstractmethod
    def request_targets(self, position: Tuple) -> Iterable[Tuple]:
        """Determine new targets to be measured."""
        ...

    @abstractmethod
    def reset(self):
        """Called when an experiment stops, or is about to start."""
        ...

    @abstractmethod
    def train(self):
        """Perform training."""
        ...

    def update_metrics(self, data: Data):
        """Compute visualization metrics. No-op by default; override for Tiled publication."""
        pass
```

The `Data` class stays unchanged (it has no Qt imports — `graphics_items` is just a plain dict).

- [ ] **Step 3: Run existing data tests**

Run: `cd /c/Users/rp/PycharmProjects/tsuchinoko && .venv/Scripts/python -m pytest tests/test_core_types.py tests/test_state_machine.py -v`

Expected: PASS (these don't depend on Qt or graphs).

- [ ] **Step 4: Commit**

```bash
git add tsuchinoko/adaptive/__init__.py tsuchinoko/parameters/__init__.py
git commit -m "refactor: decouple adaptive Engine base class from Qt/pyqtgraph"
```

---

## Task 3: Migrate GPCAMInProcessEngine

**Files:**
- Modify: `tsuchinoko/adaptive/gpCAM_in_process.py`
- Modify: `tests/test_gpcam_engine.py`

- [ ] **Step 1: Update GPCAMInProcessEngine imports and remove graphs**

In `tsuchinoko/adaptive/gpCAM_in_process.py`:

Replace the import block (lines 1-16) with:

```python
import sys
import uuid
from functools import cached_property
from typing import Callable

import numpy as np
from loguru import logger

from gpcam.gp_optimizer import GPOptimizer
from . import Engine, Data
from .acquisition_functions import explore_target_100, radical_gradient
from ..parameters.tree import SimpleParameter, GroupParameter, ListParameter, TrainingParameter
```

Remove graph imports entirely (the old `from ..graphs.common import ...` and `from ..parameters import TrainingParameter`).

In `__init__`, remove graph assignments — delete the `if dimensionality == 2: ... elif dimensionality > 2: ...` block that sets `self.graphs`.

Replace `update_metrics` method body:

```python
    def update_metrics(self, data: Data):
        pass
```

- [ ] **Step 2: Remove graph assertions from tests**

In `tests/test_gpcam_engine.py`:

In `TestInitialization.test_init_2d`, remove:
```python
        assert len(gpcam_engine_2d.graphs) > 0
```

In `TestInitialization.test_init_3d`, remove:
```python
        assert len(gpcam_engine_3d.graphs) > 0
```

Delete the entire `TestParameterProperty.test_graphs_property_2d` and `test_graphs_property_3d` methods (lines 352-367).

- [ ] **Step 3: Run gpCAM engine tests**

Run: `cd /c/Users/rp/PycharmProjects/tsuchinoko && .venv/Scripts/python -m pytest tests/test_gpcam_engine.py -v`

Expected: All remaining tests PASS.

- [ ] **Step 4: Commit**

```bash
git add tsuchinoko/adaptive/gpCAM_in_process.py tests/test_gpcam_engine.py
git commit -m "refactor: migrate GPCAMInProcessEngine to headless ParameterTree"
```

---

## Task 4: Migrate remaining adaptive engines

**Files:**
- Modify: `tsuchinoko/adaptive/random_in_process.py`
- Modify: `tsuchinoko/adaptive/grid.py`
- Modify: `tsuchinoko/adaptive/fvgp_gpCAM_in_process.py`

- [ ] **Step 1: Update RandomInProcess**

In `tsuchinoko/adaptive/random_in_process.py`, replace the import block:

```python
from functools import cached_property

import numpy as np

from tsuchinoko.adaptive import Engine, Data
from tsuchinoko.parameters.tree import SimpleParameter, GroupParameter
```

No other changes needed — RandomInProcess doesn't use graphs or TrainingParameter.

- [ ] **Step 2: Update Grid**

In `tsuchinoko/adaptive/grid.py`, replace the import block:

```python
from functools import cached_property
from itertools import product

import numpy as np

from tsuchinoko.adaptive import Engine, Data
from tsuchinoko.parameters.tree import SimpleParameter, GroupParameter
```

Remove `from tsuchinoko.graphs.common import Variance, Score`.

In `__init__`, remove `self.graphs = [Variance(), Score()]`.

- [ ] **Step 3: Update FvgpGPCAMInProcessEngine**

In `tsuchinoko/adaptive/fvgp_gpCAM_in_process.py`, replace the import block:

```python
import sys

import numpy as np

from gpcam.gp_optimizer import fvGPOptimizer
from .gpCAM_in_process import GPCAMInProcessEngine
```

Remove graph imports (`from ..graphs.common import ...`).

In `__init__`, remove the graph assignment blocks:
```python
        # DELETE these blocks:
        if dimensionality == 2:
            self.graphs = [...]
        elif dimensionality > 2:
            self.graphs = [...]
```

- [ ] **Step 4: Run all adaptive engine tests**

Run: `cd /c/Users/rp/PycharmProjects/tsuchinoko && .venv/Scripts/python -m pytest tests/test_gpcam_engine.py tests/test_parameter_tree.py -v`

Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add tsuchinoko/adaptive/random_in_process.py tsuchinoko/adaptive/grid.py tsuchinoko/adaptive/fvgp_gpCAM_in_process.py
git commit -m "refactor: migrate remaining adaptive engines to headless ParameterTree"
```

---

## Task 5: Clean Core module

**Files:**
- Modify: `tsuchinoko/core/__init__.py`
- Create: `tsuchinoko/core/zmq_core.py`

- [ ] **Step 1: Move ZMQCore to separate file**

Create `tsuchinoko/core/zmq_core.py` with the full ZMQCore class and its message imports. Copy the ZMQCore class (lines 403-582 of `core/__init__.py`) and the message imports it needs:

```python
"""ZMQ-based network core (legacy — requires zmq).

For new code, use tsuchinoko.core.Core directly with NATS integration.
"""

import time
from pickle import UnpicklingError

from loguru import logger

from tsuchinoko.core import Core, CoreState
from tsuchinoko.core.messages import (
    Message, FullDataRequest, FullDataResponse, PartialDataRequest, PartialDataResponse,
    StartRequest, UnknownResponse, PauseRequest, StateRequest, GetParametersRequest,
    SetParameterRequest, GetParametersResponse, SetParameterResponse, StopRequest, StateResponse,
    MeasureRequest, MeasureResponse, ConnectRequest, ConnectResponse, ExceptionResponse,
    PushDataRequest, PushDataResponse, GraphsResponse, PullGraphsRequest, PushGraphsRequest,
    ReplayRequest, ReplayResponse, ExitRequest, SetComputeMetricsRequest
)
from tsuchinoko.utils.logging import log_time


class ZMQCore(Core):
    """ZMQ-enabled Core providing network server functionality.

    Extends Core with a ZMQ REP socket server that handles client
    requests. Each request type has a corresponding respond_* method
    that processes the request and returns an appropriate response.
    """

    def __init__(self, *args, **kwargs):
        super(ZMQCore, self).__init__(*args, **kwargs)
        self.context = None
        self.poller = None

    def start_server(self):
        import zmq
        from zmq.asyncio import Context, Poller
        self.poller = Poller()
        self.context = Context()
        socket = self.context.socket(zmq.REP)
        socket.bind("tcp://*:5555")
        self.poller.register(socket, zmq.POLLIN)

    def respond_FullDataRequest(self, request):
        with self.data.r_lock():
            return FullDataResponse(self.data.as_dict())

    def respond_PartialDataRequest(self, request):
        if self.data and request.iteration <= len(self.data) and self.state == CoreState.Running:
            with self.data.r_lock():
                partial_data = self.data[request.iteration:]
            return PartialDataResponse(partial_data.as_dict(), request.iteration)
        else:
            return StateResponse(self.state, self.compute_metrics)

    def respond_PushDataRequest(self, request):
        from tsuchinoko.adaptive import Data
        self.data = Data(**request.data)
        return PushDataResponse()

    def respond_StartRequest(self, request):
        if self.state == CoreState.Paused:
            self.state = CoreState.Resuming
        elif self.state == CoreState.Inactive:
            self.state = CoreState.Starting
        return StateResponse(self.state, self.compute_metrics)

    def respond_StopRequest(self, request):
        self.state = CoreState.Stopping
        self.experiment_thread.join()
        return StateResponse(self.state, self.compute_metrics)

    def respond_ExitRequest(self, request):
        self.state = CoreState.Exiting
        return StateResponse(self.state, self.compute_metrics)

    def respond_PauseRequest(self, request):
        self.state = CoreState.Pausing
        return StateResponse(self.state, self.compute_metrics)

    def respond_StateRequest(self, request):
        if not self._exception_queue.empty():
            return ExceptionResponse(self._exception_queue.get())
        else:
            return StateResponse(self.state, self.compute_metrics)

    def respond_GetParametersRequest(self, request):
        return GetParametersResponse(self.adaptive_engine.parameters.saveState())

    def respond_SetParameterRequest(self, request):
        self.adaptive_engine.parameters.child(*request.child_path).setValue(request.value)
        return SetParameterResponse(True)

    def respond_MeasureRequest(self, request):
        self._forced_position_queue.put(request.position)
        return MeasureResponse(True)

    def respond_ConnectRequest(self, request):
        return ConnectResponse(self.state, self.compute_metrics)

    def respond_PullGraphsRequest(self, request):
        return GraphsResponse([])

    def respond_PushGraphsRequest(self, request):
        return StateResponse(self.state, self.compute_metrics)

    def respond_SetComputeMetricsRequest(self, request):
        self.compute_metrics = request.compute_metrics
        return StateResponse(self.state, self.compute_metrics)

    def respond_ReplayRequest(self, request):
        self._forced_measurement_queue.queue.clear()
        self._forced_position_queue.queue.clear()
        for position in request.positions:
            self._forced_position_queue.put(position)
        for measurement in request.measurements:
            self._forced_measurement_queue.put(measurement)
        return ReplayResponse(True)

    async def notify_clients(self):
        import zmq
        if not self.poller:
            self.start_server()
        sockets = dict(await self.poller.poll(timeout=.1))
        for socket in sockets:
            try:
                request = await socket.recv_pyobj(zmq.NOBLOCK)
            except (zmq.ZMQError, zmq.error.Again) as ex:
                logger.exception(ex)
            except UnpicklingError as ex:
                logger.exception(ex)
            else:
                if not request:
                    time.sleep(.1)
                    continue
                responder = getattr(self, f'respond_{request.__class__.__name__}', None)
                if responder:
                    try:
                        response = responder(request)
                    except Exception as ex:
                        response = ExceptionResponse(ex)
                else:
                    response = UnknownResponse()
                await socket.send_pyobj(response)

    def exit_later(self):
        self.state = CoreState.Exiting

    def exit(self):
        self.exit_later()
        if self.experiment_thread:
            self.experiment_thread.join()
```

- [ ] **Step 2: Clean core/__init__.py**

Replace `tsuchinoko/core/__init__.py` with the clean version. Key changes:

1. Remove all message imports (lines 18-24)
2. Remove `from tsuchinoko.graphs import Graph` (line 15)
3. Remove the entire ZMQCore class (lines 403-582)
4. Remove `graphs` property and `update_graph` method (lines 339-371)
5. Add `asyncio.sleep` to `notify_clients` to prevent busy-wait
6. Add `exit()` convenience method to Core

The cleaned imports:

```python
import asyncio
import os
import threading
import time
from asyncio import events
from enum import Enum, auto
from queue import Queue

from loguru import logger
from yaml import dump
from appdirs import user_state_dir

from ..adaptive import Engine as AdaptiveEngine, Data
from ..execution import Engine as ExecutionEngine
from ..utils.logging import log_time
```

Add to Core class:

```python
    async def notify_clients(self) -> None:
        """Hook for subclasses. Base implementation sleeps to prevent busy-wait."""
        await asyncio.sleep(0.1)

    def exit(self) -> None:
        """Request exit and wait for experiment thread to finish."""
        self.state = CoreState.Exiting
        if self.experiment_thread:
            self.experiment_thread.join()
```

Remove from Core class: `graphs` property (lines 339-343), `graphs.setter` (lines 345-347), `update_graph` method (lines 349-371).

- [ ] **Step 3: Run core and state machine tests**

Run: `cd /c/Users/rp/PycharmProjects/tsuchinoko && .venv/Scripts/python -m pytest tests/test_state_machine.py tests/test_core_types.py -v`

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tsuchinoko/core/__init__.py tsuchinoko/core/zmq_core.py
git commit -m "refactor: extract ZMQCore to separate module, clean Core of Qt/graph deps"
```

---

## Task 6: Update package structure

**Files:**
- Modify: `pyproject.toml`
- Modify: `tsuchinoko/__init__.py`
- Create: `tsuchinoko/cli.py`

- [ ] **Step 1: Update pyproject.toml**

Move Qt-specific dependencies to an optional `[gui]` extra. Update description and keywords.

Replace the `[project]` section's dependencies and optional-dependencies:

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
]

[project.optional-dependencies]
gui = [
  "pyside6<6.9.1",
  "pyqtgraph",
  "qtmodern",
  "pyopengl",
  "pyopengl-accelerate",
  "qtconsole",
  "bluesky",
  "ophyd",
  "zmq",
  "pillow",
]
dev = ["pyinstaller", "pillow"]
docs = ["matplotlib", "sphinx", "sphinx-markdown-tables", "numpydoc", "sphinx_copybutton", "myst_parser", "sphinx_rtd_theme", "sphinx_rtd_dark_mode"]
tests = ["pillow", "pytest", "coverage", "coveralls", "codecov", "pylint", "pytest-cov", "pytest-lazy-fixtures", "pytest-asyncio"]
```

Update description:
```toml
description = "An adaptive experiment service powered by Gaussian Processes"
keywords = ["autonomous", "self driving", "adaptive", "gaussian process", "optimization"]
```

Add headless entry point:
```toml
[project.scripts]
tsuchinoko = "tsuchinoko.cli:main"
tsuchinoko_demo = "tsuchinoko:launch_server"
tsuchinoko_bootstrap = "tsuchinoko:bootstrap"
```

Move gui-scripts into gui optional (they still work if gui is installed):
```toml
[project.gui-scripts]
tsuchinoko-gui = "tsuchinoko:launch_client"
```

- [ ] **Step 2: Clean tsuchinoko/__init__.py**

Replace with:

```python
import importlib
import os
import runpy
import sys

import click

try:
    from ._version import __version__
except (ImportError, ModuleNotFoundError) as ex:
    raise ImportError("You probably haven't installed tsuchinoko yet: pip install -e .") from ex


@click.command()
@click.argument('core_address', required=False, default='localhost')
def launch_client(core_address='localhost'):
    """Launch the Qt GUI client (requires tsuchinoko[gui])."""
    try:
        from pyqtgraph import mkQApp
        from . import parameters, patches
    except ImportError:
        raise click.ClickException("GUI requires PySide6: pip install tsuchinoko[gui]")

    import ctypes
    if os.name == 'nt':
        myappid = 'camera.tsuchinoko'
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

    from .widgets.mainwindow import MainWindow
    qapp = mkQApp('Tsuchinoko')
    main_window = MainWindow(core_address)
    main_window.show()
    sys.exit(qapp.exec_())


@click.command()
@click.argument('demo_name', required=False, default='server_demo')
def launch_server(demo_name='server_demo'):
    """Launch a demo server (may require tsuchinoko[gui] for some demos)."""
    demo_module = importlib.import_module(f'tsuchinoko.examples.{demo_name}')
    demo_module.core.main()


@click.command(context_settings=dict(ignore_unknown_options=True))
@click.argument('path', required=True)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def bootstrap(path, args):
    """PyInstaller bootstrap launcher."""
    print(path)
    sys.argv.pop(0)
    runpy.run_path(path, {}, "__main__")
```

- [ ] **Step 3: Create headless CLI**

Create `tsuchinoko/cli.py`:

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
@click.argument('config_path', required=False, default=None)
def run(config_path):
    """Run the adaptive experiment service (placeholder for Phase 2)."""
    logger.info("Tsuchinoko headless service")
    if config_path:
        logger.info(f"Config: {config_path}")
    logger.info("NATS integration not yet implemented (Phase 2)")


@main.command()
def version():
    """Print version information."""
    from tsuchinoko import __version__
    click.echo(f"tsuchinoko {__version__}")
```

- [ ] **Step 4: Verify headless import chain**

Run: `cd /c/Users/rp/PycharmProjects/tsuchinoko && .venv/Scripts/python -c "from tsuchinoko.core import Core, CoreState; from tsuchinoko.adaptive.gpCAM_in_process import GPCAMInProcessEngine; from tsuchinoko.execution.simple import SimpleEngine; print('Headless imports OK')"`

Expected: `Headless imports OK` (no Qt needed).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml tsuchinoko/__init__.py tsuchinoko/cli.py
git commit -m "refactor: make Qt optional, add headless CLI entry point"
```

---

## Task 7: Update test infrastructure

**Files:**
- Modify: `tests/conftest.py`
- Modify: `tests/test_gpcam_engine.py` (if not already done in Task 3)

- [ ] **Step 1: Update conftest.py**

Replace `tests/conftest.py`:

```python
from pathlib import Path
from threading import Thread

import pytest
import numpy as np
from PIL import Image
from loguru import logger
from pytest import fixture
from pytest_lazy_fixtures import lf as lazy_fixture
from scipy import ndimage

from tsuchinoko.adaptive.gpCAM_in_process import GPCAMInProcessEngine
from tsuchinoko.adaptive.random_in_process import RandomInProcess
from tsuchinoko.core import CoreState, Core
from tsuchinoko.execution.simple import SimpleEngine
from tsuchinoko.execution.threaded_in_process import ThreadedInProcessEngine


@fixture
def image_data():
    image = np.flipud(
        np.asarray(Image.open(Path(__file__).parent.parent / 'tsuchinoko' / 'examples' / 'sombrero_pug.jpg')))
    luminosity = np.average(image, axis=2)
    return luminosity


@fixture
def image_func(image_data):
    def bilinear_sample(pos):
        return pos, ndimage.map_coordinates(image_data, [[pos[1]], [pos[0]]], order=1)[0], 1, {}
    return bilinear_sample


@fixture
def simple_execution_engine(image_func):
    return SimpleEngine(measure_func=image_func)


@fixture
def gpcam_engine(image_data):
    return GPCAMInProcessEngine(dimensionality=2,
                                parameter_bounds=[(0, image_data.shape[1]),
                                                  (0, image_data.shape[0])],
                                hyperparameters=[255, 100, 100],
                                hyperparameter_bounds=[(0, 1e5), (0, 1e5), (0, 1e5)])


@fixture
def random_engine(image_data):
    return RandomInProcess(dimensionality=2,
                           parameter_bounds=[(0, image_data.shape[1]),
                                             (0, image_data.shape[0])],
                           max_targets=100)


@fixture
def bluesky_execution_engine(image_func):
    PySide6 = pytest.importorskip('PySide6')
    from tsuchinoko.execution.bluesky_in_process import BlueskyInProcessEngine
    from tsuchinoko.utils.runengine import get_run_engine

    from bluesky.plan_stubs import checkpoint, mov, trigger_and_read
    from ophyd import Device
    from ophyd.sim import SynAxis, SynSignal, Cpt

    class PointDetector(Device):
        motor1 = Cpt(SynAxis, name='motor1')
        motor2 = Cpt(SynAxis, name='motor2')
        value = Cpt(SynSignal, name='value')

        def __init__(self, name):
            super(PointDetector, self).__init__(name=name)
            self.value.sim_set_func(self.get_value)

        def get_value(self):
            return image_func([int(self.motor2.position), int(self.motor1.position)])

        def trigger(self, *args, **kwargs):
            return self.value.trigger(*args, **kwargs)

    point_detector = PointDetector('point_detector')

    def measure_target(target):
        yield from checkpoint()
        yield from mov(point_detector.motor1, target[0], point_detector.motor2, target[1])
        ret = (yield from trigger_and_read([point_detector]))
        return ret[point_detector.value.name]['value'], 2

    def get_position():
        yield from checkpoint()
        return point_detector.motor1.position, point_detector.motor2.position

    execution = BlueskyInProcessEngine(measure_target, get_position)
    yield execution

    logger.info('starting bluesky engine teardown')
    RE = get_run_engine()
    RE.RE.halt()
    RE.process_queue_thread.requestInterruption()
    RE.process_queue_thread.wait()
    logger.info('bluesky engine teardown finished')


@fixture
def threaded_execution_engine(image_func):
    execution = ThreadedInProcessEngine(image_func)
    yield execution
    execution.exiting = True
    execution.measure_thread.join()


@fixture(params=[lazy_fixture('random_engine'),
                 lazy_fixture('gpcam_engine')])
def adaptive_test_engines(simple_execution_engine, request):
    adaptive_engine = request.param
    return adaptive_engine, simple_execution_engine


@fixture(params=[lazy_fixture('threaded_execution_engine'),
                 lazy_fixture('bluesky_execution_engine')])
def execution_test_engines(random_engine, request):
    execution_engine = request.param
    return random_engine, execution_engine


@fixture(params=[lazy_fixture('execution_test_engines'),
                 lazy_fixture('adaptive_test_engines')])
def core(request):
    adaptive_engine, execution_engine = request.param
    logger.info('starting setup')
    core = Core()
    core.set_adaptive_engine(adaptive_engine)
    core.set_execution_engine(execution_engine)
    server_thread = Thread(target=core.main)
    server_thread.start()
    core.state = CoreState.Starting
    logger.info('setup complete')

    yield core

    core.exit()
    server_thread.join(timeout=10)
    logger.info('teardown complete')
```

Key changes from original:
- Import `Core` instead of `ZMQCore`
- Remove top-level bluesky/ophyd/runengine imports (moved inside fixture)
- `bluesky_execution_engine` uses `pytest.importorskip('PySide6')`
- Core fixture uses `core.exit()` (now defined on base Core)
- Added timeout to `server_thread.join()`

- [ ] **Step 2: Run the core test suite**

Run: `cd /c/Users/rp/PycharmProjects/tsuchinoko && .venv/Scripts/python -m pytest tests/test_parameter_tree.py tests/test_gpcam_engine.py tests/test_state_machine.py tests/test_core_types.py tests/test_config.py -v`

Expected: All PASS.

- [ ] **Step 3: Run the integration test**

Run: `cd /c/Users/rp/PycharmProjects/tsuchinoko && .venv/Scripts/python -m pytest tests/test_core.py -v --timeout=30`

Expected: PASS (Core runs experiment iterations, data accumulates).

- [ ] **Step 4: Commit**

```bash
git add tests/conftest.py
git commit -m "refactor: update test fixtures for headless Core"
```

---

## Task 8: End-to-end headless validation

**Files:**
- Create: `tests/test_headless.py`

- [ ] **Step 1: Write end-to-end headless test**

Create `tests/test_headless.py`:

```python
"""End-to-end test: headless adaptive experiment with no Qt dependency.

Validates that the full import chain and experiment loop work
without PySide6, pyqtgraph, or ZMQ.
"""

import time
from threading import Thread

import numpy as np
import pytest

from tsuchinoko.adaptive import Data
from tsuchinoko.adaptive.gpCAM_in_process import GPCAMInProcessEngine
from tsuchinoko.adaptive.random_in_process import RandomInProcess
from tsuchinoko.core import Core, CoreState
from tsuchinoko.execution.simple import SimpleEngine


def _measure_func(pos):
    """Simple 2D test function."""
    x, y = pos
    value = np.sin(x / 30) + np.cos(y / 30)
    return pos, value, 0.1, {}


class TestHeadlessExperiment:
    """Full experiment lifecycle without Qt."""

    def test_gpcam_headless(self):
        """Run gpCAM adaptive experiment headless."""
        engine = GPCAMInProcessEngine(
            dimensionality=2,
            parameter_bounds=[(0, 100), (0, 100)],
            hyperparameters=[100, 10, 10],
            hyperparameter_bounds=[(0.1, 1e5), (0.1, 1e5), (0.1, 1e5)],
        )
        execution = SimpleEngine(measure_func=_measure_func)

        core = Core(
            execution_engine=execution,
            adaptive_engine=engine,
            compute_metrics=False,
        )
        core.exit_at = [10]

        thread = Thread(target=core.main)
        thread.start()
        core.state = CoreState.Starting
        thread.join(timeout=60)

        assert not thread.is_alive(), "Core did not exit in time"
        assert len(core.data) >= 10

    def test_random_headless(self):
        """Run random sampling experiment headless."""
        engine = RandomInProcess(
            dimensionality=2,
            parameter_bounds=[(0, 100), (0, 100)],
            max_targets=20,
        )
        execution = SimpleEngine(measure_func=_measure_func)

        core = Core(
            execution_engine=execution,
            adaptive_engine=engine,
            compute_metrics=False,
        )
        core.exit_at = [15]

        thread = Thread(target=core.main)
        thread.start()
        core.state = CoreState.Starting
        thread.join(timeout=30)

        assert not thread.is_alive()
        assert len(core.data) >= 15

    def test_pause_resume_headless(self):
        """Test pause/resume lifecycle headless."""
        engine = RandomInProcess(
            dimensionality=2,
            parameter_bounds=[(0, 100), (0, 100)],
        )
        execution = SimpleEngine(measure_func=_measure_func)

        core = Core(
            execution_engine=execution,
            adaptive_engine=engine,
            compute_metrics=False,
        )

        thread = Thread(target=core.main)
        thread.start()
        core.state = CoreState.Starting

        # Let it run briefly
        time.sleep(1)
        assert core.state == CoreState.Running
        assert len(core.data) > 0

        # Pause
        core.state = CoreState.Pausing
        time.sleep(0.5)
        assert core.state == CoreState.Paused
        count_at_pause = len(core.data)

        # Data should not grow while paused
        time.sleep(0.5)
        assert len(core.data) == count_at_pause

        # Resume
        core.state = CoreState.Resuming
        time.sleep(1)
        assert len(core.data) > count_at_pause

        # Exit
        core.exit()
        thread.join(timeout=10)
        assert not thread.is_alive()

    def test_import_chain_no_qt(self):
        """Verify the headless import chain has no Qt dependency."""
        # These imports should work without PySide6
        import tsuchinoko
        from tsuchinoko.core import Core, CoreState
        from tsuchinoko.core.state_machine import CoreStateMachine
        from tsuchinoko.adaptive import Engine, Data
        from tsuchinoko.adaptive.gpCAM_in_process import GPCAMInProcessEngine
        from tsuchinoko.adaptive.random_in_process import RandomInProcess
        from tsuchinoko.adaptive.grid import Grid
        from tsuchinoko.execution import Engine as ExecEngine
        from tsuchinoko.execution.simple import SimpleEngine
        from tsuchinoko.execution.threaded_in_process import ThreadedInProcessEngine
        from tsuchinoko.parameters.tree import Parameter, ListParameter, TrainingParameter
        from tsuchinoko.config import AppConfig, CoreConfig
```

- [ ] **Step 2: Run end-to-end tests**

Run: `cd /c/Users/rp/PycharmProjects/tsuchinoko && .venv/Scripts/python -m pytest tests/test_headless.py -v --timeout=120`

Expected: All 4 tests PASS.

- [ ] **Step 3: Run full test suite (headless subset)**

Run: `cd /c/Users/rp/PycharmProjects/tsuchinoko && .venv/Scripts/python -m pytest tests/test_parameter_tree.py tests/test_gpcam_engine.py tests/test_state_machine.py tests/test_core_types.py tests/test_config.py tests/test_headless.py -v`

Expected: All PASS. These tests form the headless core validation suite.

- [ ] **Step 4: Commit**

```bash
git add tests/test_headless.py
git commit -m "test: add end-to-end headless experiment tests"
```

- [ ] **Step 5: Final commit — update design doc status**

In `docs/design/2026-04-12-tsuchinoko-rescope.md`, change line 3:

```markdown
**Status:** Phase 1 complete
```

```bash
git add docs/design/2026-04-12-tsuchinoko-rescope.md
git commit -m "docs: mark Phase 1 (headless core extraction) complete"
```

---

## Verification Checklist

After all tasks, verify:

- [ ] `import tsuchinoko` works without PySide6
- [ ] `from tsuchinoko.core import Core, CoreState` — no Qt import triggered
- [ ] `from tsuchinoko.adaptive.gpCAM_in_process import GPCAMInProcessEngine` — no Qt
- [ ] `GPCAMInProcessEngine` initializes, computes targets, trains — all without Qt
- [ ] `Core` runs a full experiment loop with `SimpleEngine` + `GPCAMInProcessEngine`
- [ ] All tests in `test_parameter_tree.py` pass
- [ ] All tests in `test_gpcam_engine.py` pass
- [ ] All tests in `test_state_machine.py` pass
- [ ] All tests in `test_headless.py` pass
- [ ] `test_core.py` integration test passes
- [ ] Qt-dependent tests (gui, graphics, network) skip cleanly if PySide6 absent
