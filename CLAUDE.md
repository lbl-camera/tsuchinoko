# Tsuchinoko Project Context

## Development Plan
See `CODEBASE_ANALYSIS.md` for a comprehensive codebase analysis and phased improvement plan covering:
- Critical fixes (exception handling, blocking I/O, test coverage)
- Architecture improvements (extract NetworkManager, state machine, dependency injection)
- Code quality (type hints, docstrings, signal/callback consolidation)

## Overview
Tsuchinoko is a Qt application for adaptive experiment tuning and execution, powered by gpCAM for Gaussian Process optimization.

## Tech Stack
- **GUI Framework**: PySide6 (migrated from qtpy abstraction layer)
- **Adaptive Engine**: gpCAM ~8.2.3, fvgp ~4.6.5
- **Visualization**: pyqtgraph
- **Experiment Framework**: Bluesky, ophyd

## Critical Migration Notes

### PySide6 Import Changes (from qtpy)
When migrating from qtpy to PySide6, note these module relocations:
- `QAction` moved from `QtWidgets` to `QtGui`
- `QActionGroup` moved from `QtWidgets` to `QtGui`
- `QShortcut` moved from `QtWidgets` to `QtGui`

Example correct imports:
```python
from PySide6.QtCore import Qt, Signal, QObject
from PySide6.QtGui import QAction, QActionGroup, QShortcut, QIcon
from PySide6.QtWidgets import QMainWindow, QWidget, QMenuBar
```

### Removed Dependencies (as of test/gpcam-coverage branch)
- `qtpy` - replaced with direct PySide6 imports
- `pyqode.python`, `pyqode.core` - server editor feature removed

## Testing Notes

### pytest-lazy-fixtures
Tests use `pytest-lazy-fixtures` (note the 's') which is compatible with pytest 8.x. The import is:
```python
from pytest_lazy_fixtures import lf as lazy_fixture
```

### gpCAM Testing Considerations
When testing gpCAM's `request_targets()` with trained GP:
- The default "global" optimization method can fail with scipy errors related to vectorized acquisition functions
- Using `method='local'` is more stable for unit tests but may also fail
- Tests should focus on verifying data flow and state rather than full optimization runs
- Random target generation (before GP is initialized) is reliable for testing bounds

### Test Coverage
37 tests with 36% overall coverage. Key modules:
- `gpCAM_in_process.py`: 98% (30 tests)
- `random_in_process.py`: 100%
- `core/__init__.py`: 74%
- `mainwindow.py`: 75%
- `displays.py`: 79%

Run tests: `pytest tests/ -v --cov=tsuchinoko`

## Architecture Notes

### Data Flow
1. `ZMQCore` manages server-side experiment loop
2. `MainWindow` is the client GUI
3. Communication via ZMQ REQ/REP sockets
4. `Data` class holds experiment state with read/write locks

### Adaptive Engine Interface
Engines implement:
- `update_measurements(data)` - receive new measurements
- `request_targets(position)` - get next measurement targets
- `train()` - trigger hyperparameter optimization
- `reset()` - reinitialize state
- `parameters` - pyqtgraph ParameterTree for configuration
- `graphs` - list of visualization Graph objects

## File Structure
```
tsuchinoko/
├── adaptive/          # Adaptive engines (gpCAM, random, etc.)
├── core/              # ZMQ server, state management
├── execution/         # Execution engines (Bluesky, simple, threaded)
├── graphs/            # Visualization graph definitions
├── graphics_items/    # Custom pyqtgraph items
├── widgets/           # Qt widgets (mainwindow, displays, etc.)
├── utils/             # Threading, run engine utilities
└── patches/           # Compatibility patches for dependencies
```
