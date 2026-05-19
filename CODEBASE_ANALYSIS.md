# Tsuchinoko Codebase Analysis

## Executive Summary

Tsuchinoko is a Python/Qt application for adaptive experimental optimization using Gaussian Process-based methods. The codebase contains **84 Python files** with approximately **13,400 lines of source code** and **377 lines of test code** (2.8% test coverage).

**Critical concerns:**
- Blocking I/O operations on the main Qt thread can freeze the UI
- Minimal test coverage leaves most functionality untested
- Bare exception handlers silently swallow errors
- Monolithic MainWindow class with multiple responsibilities

The codebase would benefit from architectural refactoring to separate concerns, improved error handling, and significantly expanded test coverage.

---

## 1. Critical Issues (Must Fix)

### 1.1 Blocking I/O on Main Thread

**Location:** `tsuchinoko/widgets/mainwindow.py:222-227`

```python
try:
    self.socket.send_pyobj(request)
    response = self.socket.recv_pyobj()
except (ZMQError, Again) as ex:
    logger.warning(f'Unable to connect to core server at {self.core_address}...')
    time.sleep(1)
```

**Impact:** The ZMQ socket has a 5-second receive timeout (`RCVTIMEO = 5000` at line 149). If the server is slow or unresponsive, `recv_pyobj()` blocks the thread for up to 5 seconds. While this runs in a `QThreadFutureIterator`, the callback invocations at lines 244-248 use `invoke_as_event` which marshals back to the main thread, creating potential UI freeze conditions.

**Recommendation:** Implement proper async/await patterns or use non-blocking ZMQ patterns with `zmq.NOBLOCK` and polling.

---

### 1.2 Minimal Test Coverage

**Metrics:**
- Source LOC: ~13,408
- Test LOC: 377
- Coverage: **2.8%**
- Test files: 6 (including fixtures and conftest)

**Example of shallow testing:** `tests/test_core.py:4-6`
```python
def test_core(core):
    time.sleep(1)
    assert len(core.data)
```

This test only verifies that data exists after 1 second - it doesn't validate correctness, edge cases, or error handling.

**Recommendation:** Implement comprehensive unit tests for:
- Core state machine transitions
- Data serialization/deserialization
- Widget state management
- Network protocol handling
- Graph computation logic

---

### 1.3 Bare Exception Handlers

**Location 1:** `tsuchinoko/widgets/graph_widgets.py:126-129`
```python
try:
    del self.keysPressed[ev.key()]
except:
    self.keysPressed = {}
```

**Location 2:** `tsuchinoko/experiments/multi-task.py:57-60`
```python
try:
    res[ind0] = np.mean(adaptive.optimizer.y_data[ind0])
    res[ind1] = np.mean(adaptive.optimizer.y_data[ind1])
except:
    pass
```

**Impact:** Bare `except:` catches *all* exceptions including `KeyboardInterrupt`, `SystemExit`, and programming errors like `TypeError`. Silent `pass` makes debugging nearly impossible.

**Recommendation:**
- Catch specific exceptions (`except KeyError:`)
- Log exceptions even when recovering
- Never use bare `except:` without re-raising

---

## 2. High Priority Issues

### 2.1 Monolithic MainWindow Class

**Location:** `tsuchinoko/widgets/mainwindow.py` (448 lines)

The `MainWindow` class handles:
- Socket management and ZMQ communication (lines 106-150)
- Message queue processing (lines 189-248)
- Data management and callbacks (lines 250-292)
- File I/O operations (lines 294-356)
- Demo/server process management (lines 393-447)
- UI layout and widget management (lines 51-136)

**Recommendation:** Extract into separate classes:
- `NetworkManager` - ZMQ socket lifecycle and message passing
- `DataManager` - Data storage, serialization, callbacks
- `ProcessManager` - Demo and server subprocess handling

---

### 2.2 State Management Chaos

**Multiple sources of truth:**

1. `MainWindow` references state: `mainwindow.py:197,201,204,210,242`
   ```python
   if self.state_manager_widget.state == CoreState.Stopping:
   ```

2. `StateManager._state`: `displays.py:121,160-162,170-171,195`
   ```python
   self._state = CoreState.Connecting
   ...
   self._state = state
   ```

3. `Core._state`: `core/__init__.py:51,69-76`
   ```python
   self._state = CoreState.Inactive
   ```

State is duplicated between the server (`Core._state`), the client widget (`StateManager._state`), and accessed transitively through `MainWindow.state_manager_widget.state`.

**Recommendation:** Implement a single source of truth with observer pattern for state changes.

---

### 2.3 No State Machine Pattern

**Location:** `tsuchinoko/widgets/displays.py:173-195`

```python
@state.setter
def state(self, state):
    if state in [CoreState.Starting, CoreState.Pausing, CoreState.Restarting, CoreState.Connecting]:
        self.start_pause_button.setDisabled(True)
        self.stop_button.setDisabled(True)
    elif state in [CoreState.Running]:
        self.start_pause_button.setText('Pause')
        ...
    elif state in [CoreState.Paused]:
        self.start_pause_button.setText('Resume')
        ...
```

State transitions are implicit if/elif chains rather than explicit state machine transitions.

**Recommendation:** Use a state machine library like `transitions` to define:
- Valid states
- Valid transitions between states
- Guards and callbacks for transitions

---

### 2.4 Tight Widget Coupling via Singletons

**Location:** `tsuchinoko/widgets/displays.py:21-27`

```python
class Singleton(type(QObject)):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
```

Used by: `Configuration`, `StateManager`, `GraphManager`, `LogHandler`

**Impact:**
- Makes testing difficult (can't create fresh instances)
- Hidden dependencies between components
- Potential issues with widget lifecycle in Qt

**Recommendation:** Use dependency injection instead of singletons. Pass widget references explicitly through constructors.

---

### 2.5 Inconsistent Type Hints

**Metrics:**
- Files with return type hints: **9 of 84** (10.7%)
- Functions with type hints: ~10-15% estimated

Files with type hints:
- `tsuchinoko/execution/bluesky_adaptive.py`
- `tsuchinoko/adaptive/__init__.py`
- `tsuchinoko/execution/simple.py`
- `tsuchinoko/execution/replay.py`
- `tsuchinoko/adaptive/quadtree.py`
- `tsuchinoko/utils/dependencies.py`
- `tsuchinoko/utils/runengine.py`
- `tsuchinoko/utils/logging.py`
- `tsuchinoko/execution/__init__.py`

**Recommendation:** Add comprehensive type hints across the codebase, starting with public APIs and core modules.

---

## 3. Medium Priority Issues

### 3.1 Weak Configuration Management

**Magic numbers scattered throughout:**

- `mainwindow.py:87`: `self.resize(1700, 1000)` - hardcoded window size
- `mainwindow.py:147`: `self.socket.setsockopt(zmq.LINGER, 5)` - hardcoded linger time
- `mainwindow.py:149`: `self.socket.RCVTIMEO = 5000` - hardcoded timeout
- `core/__init__.py:38`: `SLEEP_FOR_FRESH_DATA_TIME = .1` - module-level constant
- `displays.py:53-54,60-61`: `while self.log_widget.count() > 100` - hardcoded log limit

**Recommendation:** Create a `Config` dataclass with validation (consider Pydantic):

```python
@dataclass
class AppConfig:
    window_width: int = 1700
    window_height: int = 1000
    socket_timeout_ms: int = 5000
    socket_linger: int = 5
    max_log_entries: int = 100
```

---

### 3.2 Code Duplication

**File dialog patterns repeated 4x in mainwindow.py:**

```python
# Lines 295-297
name, filter = QFileDialog.getOpenFileName(filter=("YAML (*.yml)"))
if not name:
    return

# Lines 325-327
name, filter = QFileDialog.getSaveFileName(filter=("YAML (*.yml)"))
if not name:
    return

# Lines 332-334
name, filter = QFileDialog.getOpenFileName(filter=("YAML (*.yml)"))
if not name:
    return

# Lines 339-341
name, filter = QFileDialog.getSaveFileName(filter=("YAML (*.yml)"))
if not name:
    return
```

**Recommendation:** Extract helper methods:
```python
def _get_open_file(self, filter_str="YAML (*.yml)") -> Optional[str]:
    name, _ = QFileDialog.getOpenFileName(filter=filter_str)
    return name or None
```

---

### 3.3 Partial Documentation

**Metrics:**
- Docstring occurrences: ~76 (152 triple-quote pairs / 2)
- Function definitions: ~731
- Docstring coverage: **~10%**

Most public APIs lack docstrings explaining purpose, parameters, and return values.

**Recommendation:** Prioritize documenting:
1. Public API methods in `Core`, `ZMQCore`
2. Widget public interfaces
3. Graph computation methods
4. Data serialization/deserialization

---

### 3.4 Mixed Signal/Callback Patterns

**Qt Signals:** `displays.py:79-80,112-116`
```python
sigRequestParameters = Signal()
sigPushParameter = Signal(list, object)
...
sigStart = Signal()
sigStop = Signal()
sigPause = Signal()
```

**Custom callback registry:** `mainwindow.py:126,288-292`
```python
self.callbacks = defaultdict(list)
...
def subscribe(self, callback, response_type: Union[Type[Message], None] = None, invoke_as_event: bool = False):
    self.callbacks[response_type].append((callback, invoke_as_event))
```

**Impact:** Two different event systems create confusion about which to use when.

**Recommendation:** Consolidate on Qt signals for UI events and use the callback system only for network message routing (or vice versa).

---

## 4. Low Priority Issues

### 4.1 Logging Inconsistencies

Some exceptions are logged, others silently handled:

**Logged:** `displays.py:248-249`
```python
except Exception as ex:
    logger.exception(ex)
```

**Silent:** `graph_widgets.py:128-129`
```python
except:
    self.keysPressed = {}
```

**Recommendation:** Establish logging policy - all exceptions should be logged at minimum DEBUG level.

---

### 4.2 YAML Security Concerns

**Location:** `mainwindow.py:17-19,308,335`

```python
try:
    from yaml import CLoader as Loader, CDumper as Dumper, dump, load
except ImportError:
    from yaml import Loader, Dumper
...
self.data = Data(**load(open(name, 'r'), Loader=Loader))
```

Using `Loader` (or `CLoader`) instead of `safe_load` can execute arbitrary Python code in malicious YAML files.

**Risk:** Low for this application (user loads their own files), but poor practice.

**Recommendation:** Use `yaml.safe_load()` or `yaml.SafeLoader`:
```python
self.data = Data(**yaml.safe_load(open(name, 'r')))
```

---

## 5. Refactoring Opportunities

### 5.1 Extract NetworkManager from MainWindow

```python
class NetworkManager:
    def __init__(self, address: str):
        self.context = zmq.Context()
        self.socket = None
        self.address = address
        self.message_queue = Queue()

    def connect(self): ...
    def send_request(self, request: Message) -> Message: ...
    def close(self): ...
```

### 5.2 Implement Async I/O

Replace blocking socket operations with `asyncio` or `zmq.asyncio`:

```python
async def send_request(self, request: Message) -> Message:
    await self.socket.send_pyobj(request)
    return await self.socket.recv_pyobj()
```

### 5.3 Config Dataclass with Validation

```python
from pydantic import BaseModel, Field

class NetworkConfig(BaseModel):
    address: str = "localhost"
    port: int = Field(5555, ge=1024, le=65535)
    timeout_ms: int = Field(5000, ge=100, le=60000)

class UIConfig(BaseModel):
    window_width: int = 1700
    window_height: int = 1000
    max_log_entries: int = 100
```

### 5.4 Define Widget Protocols

```python
from typing import Protocol

class GraphWidget(Protocol):
    def update_data(self, data: Data, update_slice: slice) -> None: ...
    def clear(self) -> None: ...
    def reset(self) -> None: ...
```

### 5.5 Add State Machine

```python
from transitions import Machine

class CoreStateMachine:
    states = ['connecting', 'inactive', 'starting', 'running', 'pausing', 'paused', 'stopping']

    def __init__(self):
        self.machine = Machine(model=self, states=self.states, initial='connecting')
        self.machine.add_transition('connect', 'connecting', 'inactive')
        self.machine.add_transition('start', 'inactive', 'starting')
        self.machine.add_transition('run', 'starting', 'running')
        # ... etc
```

### 5.6 Consolidate Event Systems

Choose one pattern:
- **Option A:** Use Qt signals everywhere, remove custom callback registry
- **Option B:** Use callback registry for all events, wrap Qt signals

---

## 6. Key Files to Address

| File | Priority | Issues |
|------|----------|--------|
| `tsuchinoko/widgets/mainwindow.py` | **Critical** | Blocking I/O, monolithic design, multiple responsibilities |
| `tsuchinoko/widgets/displays.py` | High | State management, singleton pattern, implicit state machine |
| `tsuchinoko/widgets/graph_widgets.py` | High | Bare exception handlers, silent error recovery |
| `tsuchinoko/core/__init__.py` | High | Thread coordination, state duplication, exception handling |
| `tsuchinoko/adaptive/gpCAM_in_process.py` | Medium | Exception handling patterns |
| `tsuchinoko/utils/threads.py` | Medium | Exception handling, thread safety |

---

## 7. Recommended Action Plan

### Phase 1: Critical Fixes
1. Add specific exception handling (replace bare `except:`)
2. Review and fix potential UI-blocking I/O paths
3. Add basic test coverage for core state transitions

### Phase 2: Architecture Improvements
1. Extract `NetworkManager` from `MainWindow`
2. Implement proper state machine pattern
3. Remove singleton metaclass, use dependency injection

### Phase 3: Code Quality
1. Add type hints to public APIs
2. Add docstrings to core modules
3. Consolidate signal/callback patterns

### Phase 4: Infrastructure
1. Expand test coverage to 60%+
2. Add configuration management
3. Implement async I/O patterns

---

*Generated: 2026-01-29*
