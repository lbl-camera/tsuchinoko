# Phase 3: Tiled Integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Tiled reader/writer and LUCIDEngine so Tsuchinoko can read measurements from LUCID's Tiled runs, write GP outputs back, and coordinate measurement handoff over NATS.

**Architecture:** Three new modules — `TiledReader` (reads primary stream), `TiledPublisher` (writes adaptive stream), `LUCIDEngine` (ExecutionEngine subclass coordinating NATS+Tiled). Core gains a TiledPublisher call after each iteration. Tests use real Tiled catalogs via `tiled.catalog.in_memory()`.

**Tech Stack:** tiled >=0.2.0, numpy, nats-py, pytest

**Design spec:** `docs/design/2026-04-12-phase3-tiled-integration.md`

---

## File Structure

### New files
| File | Responsibility |
|------|----------------|
| `tsuchinoko/tiled/__init__.py` | Package exports |
| `tsuchinoko/tiled/config.py` | TiledConfig Pydantic model |
| `tsuchinoko/tiled/reader.py` | Read measurements from Tiled primary stream |
| `tsuchinoko/tiled/writer.py` | Write GP outputs to Tiled adaptive stream |
| `tsuchinoko/execution/lucid.py` | LUCIDEngine ExecutionEngine subclass |
| `tests/test_tiled_reader.py` | TiledReader tests with real catalog |
| `tests/test_tiled_writer.py` | TiledPublisher tests with real catalog |
| `tests/test_lucid_engine.py` | LUCIDEngine unit tests |

### Modified files
| File | Changes |
|------|---------|
| `tsuchinoko/config.py` | Add TiledConfig to AppConfig |
| `tsuchinoko/core/__init__.py` | Add tiled_publisher call in experiment_iteration |
| `tsuchinoko/cli.py` | Add --tiled-url option |
| `pyproject.toml` | Add tiled dependency |

### Tiled test fixture pattern (used by all test files)

```python
import tempfile
import numpy as np
import pytest
from tiled.catalog import in_memory
from tiled.client import Context, from_context
from tiled.server.app import build_app


@pytest.fixture
def tiled_client():
    """Writable in-memory Tiled client for testing."""
    tmpdir = tempfile.mkdtemp()
    catalog = in_memory(writable_storage=tmpdir)
    app = build_app(catalog)
    with Context.from_app(app) as ctx:
        yield from_context(ctx)
```

---

## Task 1: TiledConfig and dependency

**Files:**
- Create: `tsuchinoko/tiled/__init__.py`
- Create: `tsuchinoko/tiled/config.py`
- Modify: `tsuchinoko/config.py`
- Modify: `pyproject.toml`
- Test: `tests/test_tiled_config.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_tiled_config.py`:

```python
"""Tests for Tiled configuration."""

from tsuchinoko.tiled.config import TiledConfig
from tsuchinoko.config import AppConfig


class TestTiledConfig:
    def test_defaults(self):
        cfg = TiledConfig()
        assert cfg.url == ""
        assert cfg.token == ""

    def test_custom(self):
        cfg = TiledConfig(url="https://tiled.example.com", token="jwt123")
        assert cfg.url == "https://tiled.example.com"
        assert cfg.token == "jwt123"

    def test_disabled_by_default(self):
        cfg = TiledConfig()
        assert not cfg.url

    def test_app_config_includes_tiled(self):
        cfg = AppConfig()
        assert hasattr(cfg, 'tiled')
        assert isinstance(cfg.tiled, TiledConfig)

    def test_app_config_from_dict(self):
        cfg = AppConfig(**{"tiled": {"url": "https://t.example.com", "token": "tok"}})
        assert cfg.tiled.url == "https://t.example.com"
```

- [ ] **Step 2: Run test — verify fail**

Run: `cd /c/Users/rp/PycharmProjects/tsuchinoko-phase1 && .venv/Scripts/python -m pytest tests/test_tiled_config.py -v --noconftest`

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Create TiledConfig**

Create `tsuchinoko/tiled/__init__.py`:
```python
"""Tiled integration for Tsuchinoko."""

from .config import TiledConfig
```

Create `tsuchinoko/tiled/config.py`:
```python
"""Tiled configuration model."""

from pydantic import BaseModel


class TiledConfig(BaseModel):
    """Configuration for the Tiled connection."""
    url: str = ""
    token: str = ""
```

- [ ] **Step 4: Add TiledConfig to AppConfig**

In `tsuchinoko/config.py`, add import and field:

```python
from tsuchinoko.tiled.config import TiledConfig
```

Add to AppConfig:
```python
    tiled: TiledConfig = Field(default_factory=TiledConfig)
```

- [ ] **Step 5: Add tiled dependency to pyproject.toml**

Add `"tiled[client]"` to the dependencies list. Also add `"tiled[server]"` to the tests optional dependencies (needed for `in_memory()` and `build_app()`).

- [ ] **Step 6: Run tests — verify pass**

Run: `cd /c/Users/rp/PycharmProjects/tsuchinoko-phase1 && .venv/Scripts/python -m pytest tests/test_tiled_config.py -v --noconftest`

Expected: All 5 PASS.

- [ ] **Step 7: Commit**

```bash
git add tsuchinoko/tiled/ tsuchinoko/config.py pyproject.toml tests/test_tiled_config.py
git commit -m "feat: add TiledConfig and tiled dependency"
```

---

## Task 2: TiledReader

**Files:**
- Create: `tsuchinoko/tiled/reader.py`
- Create: `tests/test_tiled_reader.py`
- Modify: `tsuchinoko/tiled/__init__.py`

TiledReader reads measurement data from a Tiled run's `primary` stream. It tracks rows already consumed so each `read_new()` call returns only fresh data.

- [ ] **Step 1: Write failing tests**

Create `tests/test_tiled_reader.py`:

```python
"""Tests for TiledReader with a real in-memory Tiled catalog."""

import tempfile
import numpy as np
import pytest
from tiled.catalog import in_memory
from tiled.client import Context, from_context
from tiled.server.app import build_app

from tsuchinoko.tiled.reader import TiledReader


@pytest.fixture
def tiled_context():
    """In-memory Tiled context for testing."""
    tmpdir = tempfile.mkdtemp()
    catalog = in_memory(writable_storage=tmpdir)
    app = build_app(catalog)
    with Context.from_app(app) as ctx:
        yield ctx


@pytest.fixture
def tiled_client(tiled_context):
    return from_context(tiled_context)


@pytest.fixture
def populated_run(tiled_client):
    """A run with 5 measurement points in primary stream."""
    run = tiled_client.create_container(key="run_001")
    primary = run.create_container(key="primary")
    primary.write_array(np.array([10.0, 20.0, 30.0, 40.0, 50.0]), key="x_motor")
    primary.write_array(np.array([15.0, 25.0, 35.0, 45.0, 55.0]), key="y_motor")
    primary.write_array(np.array([0.5, 0.8, 0.3, 0.9, 0.1]), key="detector")
    return "run_001"


class TestTiledReader:
    def test_read_all(self, tiled_client, populated_run):
        reader = TiledReader(
            tiled_client, populated_run,
            motor_names=["x_motor", "y_motor"],
            detector_name="detector",
        )
        measurements = reader.read_new()
        assert len(measurements) == 5
        pos, val, var, metrics = measurements[0]
        assert pos == (10.0, 15.0)
        assert val == 0.5

    def test_read_returns_tuples(self, tiled_client, populated_run):
        reader = TiledReader(
            tiled_client, populated_run,
            motor_names=["x_motor", "y_motor"],
            detector_name="detector",
        )
        measurements = reader.read_new()
        for pos, val, var, metrics in measurements:
            assert isinstance(pos, tuple)
            assert isinstance(val, float)
            assert isinstance(var, float)
            assert isinstance(metrics, dict)

    def test_incremental_read(self, tiled_client, populated_run):
        reader = TiledReader(
            tiled_client, populated_run,
            motor_names=["x_motor", "y_motor"],
            detector_name="detector",
        )
        # First read gets all 5
        m1 = reader.read_new()
        assert len(m1) == 5

        # Second read with no new data returns empty
        m2 = reader.read_new()
        assert len(m2) == 0

    def test_incremental_read_after_new_data(self, tiled_client, populated_run):
        reader = TiledReader(
            tiled_client, populated_run,
            motor_names=["x_motor", "y_motor"],
            detector_name="detector",
        )
        m1 = reader.read_new()
        assert len(m1) == 5

        # Simulate LUCID adding more data by rewriting arrays with more points
        # (In real use, LUCID's TiledWriter appends to the stream)
        run = tiled_client[populated_run]
        primary = run["primary"]
        # Delete old and write new (in real Tiled, the stream grows; here we simulate)
        primary.write_array(
            np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]),
            key="x_motor",
        )
        primary.write_array(
            np.array([15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0]),
            key="y_motor",
        )
        primary.write_array(
            np.array([0.5, 0.8, 0.3, 0.9, 0.1, 0.7, 0.2]),
            key="detector",
        )

        m2 = reader.read_new()
        assert len(m2) == 2
        assert m2[0][0] == (60.0, 65.0)

    def test_custom_variance(self, tiled_client, populated_run):
        reader = TiledReader(
            tiled_client, populated_run,
            motor_names=["x_motor", "y_motor"],
            detector_name="detector",
            default_variance=0.5,
        )
        measurements = reader.read_new()
        assert measurements[0][2] == 0.5

    def test_empty_run(self, tiled_client):
        run = tiled_client.create_container(key="empty_run")
        run.create_container(key="primary")
        reader = TiledReader(
            tiled_client, "empty_run",
            motor_names=["x_motor"],
            detector_name="detector",
        )
        measurements = reader.read_new()
        assert len(measurements) == 0
```

- [ ] **Step 2: Run tests — verify fail**

Run: `cd /c/Users/rp/PycharmProjects/tsuchinoko-phase1 && .venv/Scripts/python -m pytest tests/test_tiled_reader.py -v --noconftest`

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement TiledReader**

Create `tsuchinoko/tiled/reader.py`:

```python
"""Read measurement data from a Tiled run's primary stream."""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger


class TiledReader:
    """Reads measurements from Tiled, tracking position for incremental reads.

    Each call to read_new() returns only rows added since the last call.
    """

    def __init__(
        self,
        tiled_client: Any,
        run_uid: str,
        motor_names: list[str],
        detector_name: str,
        default_variance: float = 1.0,
    ) -> None:
        self._client = tiled_client
        self._run_uid = run_uid
        self._motor_names = motor_names
        self._detector_name = detector_name
        self._default_variance = default_variance
        self._rows_read = 0

    def read_new(self) -> list[tuple]:
        """Read rows added since last call.

        Returns list of (position_tuple, value, variance, metrics_dict).
        """
        try:
            run = self._client[self._run_uid]
            primary = run["primary"]
        except (KeyError, Exception) as e:
            logger.warning(f"Cannot read primary stream: {e}")
            return []

        # Read detector to determine total row count
        if self._detector_name not in primary:
            return []

        detector_data = primary[self._detector_name].read()
        total_rows = len(detector_data)

        if total_rows <= self._rows_read:
            return []

        # Read motor positions and detector values for new rows only
        motor_arrays = []
        for name in self._motor_names:
            arr = primary[name].read()
            motor_arrays.append(arr[self._rows_read:total_rows])

        values = detector_data[self._rows_read:total_rows]
        n_new = total_rows - self._rows_read

        measurements = []
        for i in range(n_new):
            position = tuple(float(arr[i]) for arr in motor_arrays)
            value = float(values[i])
            measurements.append((position, value, self._default_variance, {}))

        self._rows_read = total_rows
        logger.info(f"Read {n_new} new measurements (total: {total_rows})")
        return measurements
```

- [ ] **Step 4: Update tiled __init__.py**

```python
"""Tiled integration for Tsuchinoko."""

from .config import TiledConfig
from .reader import TiledReader
```

- [ ] **Step 5: Run tests — verify pass**

Run: `cd /c/Users/rp/PycharmProjects/tsuchinoko-phase1 && .venv/Scripts/python -m pytest tests/test_tiled_reader.py -v --noconftest`

Expected: All 6 PASS.

- [ ] **Step 6: Commit**

```bash
git add tsuchinoko/tiled/reader.py tsuchinoko/tiled/__init__.py tests/test_tiled_reader.py
git commit -m "feat: add TiledReader for incremental measurement reads"
```

---

## Task 3: TiledPublisher

**Files:**
- Create: `tsuchinoko/tiled/writer.py`
- Create: `tests/test_tiled_writer.py`
- Modify: `tsuchinoko/tiled/__init__.py`

TiledPublisher writes GP outputs to the `adaptive` stream using per-iteration
sub-containers. Extracts posteriors from the GP optimizer.

- [ ] **Step 1: Write failing tests**

Create `tests/test_tiled_writer.py`:

```python
"""Tests for TiledPublisher with a real in-memory Tiled catalog."""

import tempfile
from unittest.mock import MagicMock

import numpy as np
import pytest
from tiled.catalog import in_memory
from tiled.client import Context, from_context
from tiled.server.app import build_app

from tsuchinoko.tiled.writer import TiledPublisher


@pytest.fixture
def tiled_context():
    tmpdir = tempfile.mkdtemp()
    catalog = in_memory(writable_storage=tmpdir)
    app = build_app(catalog)
    with Context.from_app(app) as ctx:
        yield ctx


@pytest.fixture
def tiled_client(tiled_context):
    return from_context(tiled_context)


@pytest.fixture
def run_with_adaptive(tiled_client):
    """A Tiled run ready for adaptive stream writing."""
    tiled_client.create_container(key="run_001")
    return "run_001"


def _make_mock_engine(dimensionality=2):
    """Mock adaptive engine with GP optimizer."""
    engine = MagicMock()
    engine.dimensionality = dimensionality

    # Mock optimizer with posterior computation
    engine.optimizer = MagicMock()
    engine.optimizer.get_hyperparameters.return_value = np.array([100.0, 10.0, 10.0])

    # Mock posterior_mean — returns dict with "f(x)" key
    engine.optimizer.posterior_mean.return_value = {
        "f(x)": np.random.rand(50 * 50).reshape(-1)
    }
    engine.optimizer.posterior_covariance.return_value = {
        "v(x)": np.random.rand(50 * 50).reshape(-1)
    }

    # Parameter bounds
    engine.parameters = MagicMock()
    engine.parameters.__getitem__ = MagicMock(side_effect=lambda key: {
        ("bounds", "axis_0_min"): 0.0,
        ("bounds", "axis_0_max"): 100.0,
        ("bounds", "axis_1_min"): 0.0,
        ("bounds", "axis_1_max"): 50.0,
    }.get(key, 0.0))

    return engine


class TestTiledPublisher:
    def test_write_config(self, tiled_client, run_with_adaptive):
        engine = _make_mock_engine()
        publisher = TiledPublisher(tiled_client, run_with_adaptive, dimensionality=2, grid_resolution=50)
        publisher.write_config(engine)

        adaptive = tiled_client[run_with_adaptive]["adaptive"]
        config = adaptive["config"]
        assert "evaluation_grid_x" in config
        assert "evaluation_grid_y" in config
        grid_x = config["evaluation_grid_x"].read()
        assert len(grid_x) == 50

    def test_write_iteration(self, tiled_client, run_with_adaptive):
        engine = _make_mock_engine()
        publisher = TiledPublisher(tiled_client, run_with_adaptive, dimensionality=2, grid_resolution=50)
        publisher.write_config(engine)
        publisher.write_iteration(1, engine, targets=np.array([[25.0, 12.5]]))

        adaptive = tiled_client[run_with_adaptive]["adaptive"]
        iter_001 = adaptive["iter_001"]
        assert "posterior_mean" in iter_001
        assert "posterior_variance" in iter_001
        assert "hyperparameters" in iter_001
        assert "targets" in iter_001

        hp = iter_001["hyperparameters"].read()
        assert len(hp) == 3
        targets = iter_001["targets"].read()
        assert targets.shape == (1, 2)

    def test_write_multiple_iterations(self, tiled_client, run_with_adaptive):
        engine = _make_mock_engine()
        publisher = TiledPublisher(tiled_client, run_with_adaptive, dimensionality=2, grid_resolution=50)
        publisher.write_config(engine)
        publisher.write_iteration(1, engine, targets=np.array([[10.0, 20.0]]))
        publisher.write_iteration(2, engine, targets=np.array([[30.0, 40.0]]))
        publisher.write_iteration(3, engine, targets=np.array([[50.0, 10.0]]))

        adaptive = tiled_client[run_with_adaptive]["adaptive"]
        # config + 3 iterations
        assert "iter_001" in adaptive
        assert "iter_002" in adaptive
        assert "iter_003" in adaptive

    def test_high_dimensionality_skips_posterior(self, tiled_client, run_with_adaptive):
        engine = _make_mock_engine(dimensionality=4)
        publisher = TiledPublisher(tiled_client, run_with_adaptive, dimensionality=4, grid_resolution=50)
        publisher.write_config(engine)
        publisher.write_iteration(1, engine, targets=np.array([[1, 2, 3, 4]]))

        adaptive = tiled_client[run_with_adaptive]["adaptive"]
        iter_001 = adaptive["iter_001"]
        assert "hyperparameters" in iter_001
        assert "targets" in iter_001
        # Posterior arrays should NOT be present for D > 3
        assert "posterior_mean" not in iter_001

    def test_posterior_shape(self, tiled_client, run_with_adaptive):
        engine = _make_mock_engine()
        publisher = TiledPublisher(tiled_client, run_with_adaptive, dimensionality=2, grid_resolution=50)
        publisher.write_config(engine)
        publisher.write_iteration(1, engine, targets=np.array([[25.0, 12.5]]))

        pm = tiled_client[run_with_adaptive]["adaptive"]["iter_001"]["posterior_mean"].read()
        assert pm.shape == (50, 50)
```

- [ ] **Step 2: Run tests — verify fail**

Run: `cd /c/Users/rp/PycharmProjects/tsuchinoko-phase1 && .venv/Scripts/python -m pytest tests/test_tiled_writer.py -v --noconftest`

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement TiledPublisher**

Create `tsuchinoko/tiled/writer.py`:

```python
"""Write GP outputs to a Tiled run's adaptive stream."""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger


class TiledPublisher:
    """Writes GP outputs to Tiled per-iteration sub-containers.

    Layout:
        adaptive/
        ├── config/
        │   ├── evaluation_grid_x
        │   └── evaluation_grid_y
        ├── iter_001/
        │   ├── posterior_mean
        │   ├── posterior_variance
        │   ├── hyperparameters
        │   └── targets
        └── iter_NNN/
    """

    MAX_GRID_DIMENSIONALITY = 3

    def __init__(
        self,
        tiled_client: Any,
        run_uid: str,
        dimensionality: int,
        grid_resolution: int = 50,
    ) -> None:
        self._client = tiled_client
        self._run_uid = run_uid
        self._dimensionality = dimensionality
        self._grid_resolution = grid_resolution
        self._adaptive = None
        self._grid_points = None

    def write_config(self, engine) -> None:
        """Write evaluation grid (once, at experiment start)."""
        run = self._client[self._run_uid]
        self._adaptive = run.create_container(key="adaptive")
        config = self._adaptive.create_container(key="config")

        if self._dimensionality <= self.MAX_GRID_DIMENSIONALITY:
            bounds = []
            for i in range(self._dimensionality):
                lo = engine.parameters[("bounds", f"axis_{i}_min")]
                hi = engine.parameters[("bounds", f"axis_{i}_max")]
                bounds.append((lo, hi))

            grids = []
            for i, (lo, hi) in enumerate(bounds):
                grid = np.linspace(lo, hi, self._grid_resolution)
                config.write_array(grid, key=f"evaluation_grid_{'xyz'[i]}")
                grids.append(grid)

            # Build grid points for posterior evaluation
            mesh = np.meshgrid(*grids, indexing="ij")
            self._grid_points = np.column_stack([m.ravel() for m in mesh])

        logger.info(f"TiledPublisher config written for run {self._run_uid}")

    def write_iteration(self, iteration: int, engine, targets: np.ndarray) -> None:
        """Write GP outputs for one iteration."""
        if self._adaptive is None:
            logger.warning("write_config() not called yet, skipping iteration")
            return

        iter_key = f"iter_{iteration:03d}"
        iter_container = self._adaptive.create_container(key=iter_key)

        # Always write hyperparameters and targets
        try:
            hp = engine.optimizer.get_hyperparameters()
            iter_container.write_array(np.asarray(hp), key="hyperparameters")
        except Exception as e:
            logger.warning(f"Could not write hyperparameters: {e}")

        iter_container.write_array(np.asarray(targets), key="targets")

        # Write posterior arrays only for low-dimensional problems
        if self._dimensionality <= self.MAX_GRID_DIMENSIONALITY and self._grid_points is not None:
            try:
                if hasattr(engine, 'optimizer') and engine.optimizer.gp is not None:
                    pm = engine.optimizer.posterior_mean(self._grid_points)
                    mean_vals = np.asarray(pm["f(x)"]).reshape(
                        *[self._grid_resolution] * self._dimensionality
                    )
                    iter_container.write_array(mean_vals, key="posterior_mean")

                    pv = engine.optimizer.posterior_covariance(
                        self._grid_points, variance_only=True
                    )
                    var_vals = np.asarray(pv["v(x)"]).reshape(
                        *[self._grid_resolution] * self._dimensionality
                    )
                    iter_container.write_array(var_vals, key="posterior_variance")
            except Exception as e:
                logger.warning(f"Could not write posterior: {e}")

        logger.debug(f"Wrote iteration {iteration} to Tiled")
```

- [ ] **Step 4: Update tiled __init__.py**

```python
"""Tiled integration for Tsuchinoko."""

from .config import TiledConfig
from .reader import TiledReader
from .writer import TiledPublisher
```

- [ ] **Step 5: Run tests — verify pass**

Run: `cd /c/Users/rp/PycharmProjects/tsuchinoko-phase1 && .venv/Scripts/python -m pytest tests/test_tiled_writer.py -v --noconftest`

Expected: All 5 PASS.

- [ ] **Step 6: Commit**

```bash
git add tsuchinoko/tiled/writer.py tsuchinoko/tiled/__init__.py tests/test_tiled_writer.py
git commit -m "feat: add TiledPublisher for GP output writing"
```

---

## Task 4: LUCIDEngine

**Files:**
- Create: `tsuchinoko/execution/lucid.py`
- Create: `tests/test_lucid_engine.py`

LUCIDEngine is an ExecutionEngine subclass that coordinates with LUCID over
NATS for measurement handoff, reading results from Tiled.

- [ ] **Step 1: Write failing tests**

Create `tests/test_lucid_engine.py`:

```python
"""Tests for LUCIDEngine."""

import json
import tempfile
import threading
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from tiled.catalog import in_memory
from tiled.client import Context, from_context
from tiled.server.app import build_app

from tsuchinoko.execution.lucid import LUCIDEngine
from tsuchinoko.tiled.reader import TiledReader


@pytest.fixture
def tiled_context():
    tmpdir = tempfile.mkdtemp()
    catalog = in_memory(writable_storage=tmpdir)
    app = build_app(catalog)
    with Context.from_app(app) as ctx:
        yield ctx


@pytest.fixture
def tiled_client(tiled_context):
    return from_context(tiled_context)


@pytest.fixture
def populated_run(tiled_client):
    """A run with measurement data in primary stream."""
    run = tiled_client.create_container(key="run_001")
    primary = run.create_container(key="primary")
    primary.write_array(np.array([10.0, 20.0, 30.0]), key="x_motor")
    primary.write_array(np.array([15.0, 25.0, 35.0]), key="y_motor")
    primary.write_array(np.array([0.5, 0.8, 0.3]), key="detector")
    return "run_001"


@pytest.fixture
def mock_nats_client():
    client = MagicMock()
    client.publish_threadsafe = MagicMock()
    client.is_connected = True
    return client


@pytest.fixture
def lucid_engine(tiled_client, populated_run, mock_nats_client):
    reader = TiledReader(
        tiled_client, populated_run,
        motor_names=["x_motor", "y_motor"],
        detector_name="detector",
    )
    engine = LUCIDEngine(
        nats_client=mock_nats_client,
        tiled_reader=reader,
        lucid_prefix="test.lucid",
        run_uid=populated_run,
    )
    return engine


class TestLUCIDEngine:
    def test_update_targets(self, lucid_engine, mock_nats_client):
        targets = [(25.0, 12.5), (50.0, 25.0)]
        lucid_engine.update_targets(targets)

        mock_nats_client.publish_threadsafe.assert_called_once()
        subject, payload = mock_nats_client.publish_threadsafe.call_args[0]
        assert subject == "tsuchinoko.targets"
        assert len(payload["targets"]) == 2

    def test_get_position_default(self, lucid_engine):
        pos = lucid_engine.get_position()
        assert pos == (0, 0)

    def test_get_position_after_targets(self, lucid_engine):
        lucid_engine.update_targets([(42.0, 17.0)])
        pos = lucid_engine.get_position()
        assert pos == (42.0, 17.0)

    def test_get_measurements_after_signal(self, lucid_engine):
        # Signal that measurements are ready (simulates NATS callback)
        lucid_engine.signal_measurements_ready()

        measurements = lucid_engine.get_measurements()
        assert len(measurements) == 3
        pos, val, var, metrics = measurements[0]
        assert pos == (10.0, 15.0)
        assert val == 0.5

    def test_get_measurements_blocks(self, lucid_engine):
        """get_measurements blocks until signaled."""
        results = []

        def reader_thread():
            m = lucid_engine.get_measurements()
            results.append(m)

        t = threading.Thread(target=reader_thread)
        t.start()

        # Thread should be blocked
        t.join(timeout=0.5)
        assert t.is_alive()  # still blocked

        # Signal and wait
        lucid_engine.signal_measurements_ready()
        t.join(timeout=5)
        assert not t.is_alive()
        assert len(results[0]) == 3
```

- [ ] **Step 2: Run tests — verify fail**

Run: `cd /c/Users/rp/PycharmProjects/tsuchinoko-phase1 && .venv/Scripts/python -m pytest tests/test_lucid_engine.py -v --noconftest`

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement LUCIDEngine**

Create `tsuchinoko/execution/lucid.py`:

```python
"""LUCID execution engine — coordinates measurement via NATS + Tiled."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, List, Tuple

from loguru import logger

if TYPE_CHECKING:
    from tsuchinoko.nats.client import NATSClient
    from tsuchinoko.tiled.reader import TiledReader

from . import Engine


class LUCIDEngine(Engine):
    """ExecutionEngine that coordinates with LUCID over NATS.

    Publishes targets via NATS, waits for LUCID's measurement signal,
    then reads results from Tiled.
    """

    def __init__(
        self,
        nats_client: NATSClient,
        tiled_reader: TiledReader,
        lucid_prefix: str,
        run_uid: str,
    ) -> None:
        self._nats_client = nats_client
        self._tiled_reader = tiled_reader
        self._lucid_prefix = lucid_prefix
        self._run_uid = run_uid
        self._position: tuple = (0, 0)
        self._measured_event = threading.Event()
        self._iteration = 0

    def update_targets(self, targets: List[Tuple]) -> None:
        """Publish targets to NATS for LUCID to measure."""
        self._iteration += 1
        if targets:
            self._position = tuple(targets[-1])

        self._nats_client.publish_threadsafe("tsuchinoko.targets", {
            "run_uid": self._run_uid,
            "targets": [list(t) for t in targets],
            "iteration": self._iteration,
        })
        logger.info(f"Published {len(targets)} targets (iteration {self._iteration})")

    def get_position(self) -> Tuple:
        """Return last known target position."""
        return self._position

    def get_measurements(self) -> List[Tuple]:
        """Block until LUCID signals measurements ready, then read from Tiled."""
        self._measured_event.wait(timeout=300)
        self._measured_event.clear()
        return self._tiled_reader.read_new()

    def signal_measurements_ready(self) -> None:
        """Called by NATS callback when {prefix}.adaptive.measured arrives."""
        self._measured_event.set()
```

- [ ] **Step 4: Run tests — verify pass**

Run: `cd /c/Users/rp/PycharmProjects/tsuchinoko-phase1 && .venv/Scripts/python -m pytest tests/test_lucid_engine.py -v --noconftest`

Expected: All 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add tsuchinoko/execution/lucid.py tests/test_lucid_engine.py
git commit -m "feat: add LUCIDEngine for NATS+Tiled measurement handoff"
```

---

## Task 5: Core integration

**Files:**
- Modify: `tsuchinoko/core/__init__.py`
- Modify: `tsuchinoko/cli.py`
- Create: `tests/test_core_tiled.py`

Add TiledPublisher call to `experiment_iteration()` and `--tiled-url` to CLI.
Core accepts an optional `tiled_publisher` that it calls after each iteration.

- [ ] **Step 1: Write failing test**

Create `tests/test_core_tiled.py`:

```python
"""Tests for Core with TiledPublisher integration."""

import tempfile
import time
from threading import Thread

import numpy as np
import pytest
from tiled.catalog import in_memory
from tiled.client import Context, from_context
from tiled.server.app import build_app

from tsuchinoko.adaptive.gpCAM_in_process import GPCAMInProcessEngine
from tsuchinoko.core import Core, CoreState
from tsuchinoko.execution.simple import SimpleEngine
from tsuchinoko.tiled.writer import TiledPublisher


def _measure(pos):
    x, y = pos
    return pos, np.sin(x / 30) + np.cos(y / 30), 0.1, {}


@pytest.fixture
def tiled_context():
    tmpdir = tempfile.mkdtemp()
    catalog = in_memory(writable_storage=tmpdir)
    app = build_app(catalog)
    with Context.from_app(app) as ctx:
        yield ctx


@pytest.fixture
def tiled_client(tiled_context):
    return from_context(tiled_context)


class TestCoreWithTiledPublisher:
    def test_publishes_after_iterations(self, tiled_client):
        # Set up run container
        tiled_client.create_container(key="test_run")

        engine = GPCAMInProcessEngine(
            dimensionality=2,
            parameter_bounds=[(0, 100), (0, 100)],
            hyperparameters=[100, 10, 10],
            hyperparameter_bounds=[(0.1, 1e5), (0.1, 1e5), (0.1, 1e5)],
        )
        execution = SimpleEngine(measure_func=_measure)

        publisher = TiledPublisher(tiled_client, "test_run", dimensionality=2, grid_resolution=20)
        publisher.write_config(engine)

        core = Core(
            execution_engine=execution,
            adaptive_engine=engine,
            compute_metrics=False,
            tiled_publisher=publisher,
        )
        core.exit_at = [5]

        thread = Thread(target=core.main)
        thread.start()
        core.state = CoreState.Starting
        thread.join(timeout=60)

        assert not thread.is_alive()
        assert len(core.data) >= 5

        # Verify iterations were written to Tiled
        adaptive = tiled_client["test_run"]["adaptive"]
        children = list(adaptive)
        iter_keys = [k for k in children if k.startswith("iter_")]
        assert len(iter_keys) >= 1  # at least some iterations published

    def test_no_publisher_no_error(self):
        """Core without tiled_publisher should work fine."""
        engine = GPCAMInProcessEngine(
            dimensionality=2,
            parameter_bounds=[(0, 100), (0, 100)],
            hyperparameters=[100, 10, 10],
            hyperparameter_bounds=[(0.1, 1e5), (0.1, 1e5), (0.1, 1e5)],
        )
        execution = SimpleEngine(measure_func=_measure)

        core = Core(
            execution_engine=execution,
            adaptive_engine=engine,
            compute_metrics=False,
        )
        core.exit_at = [3]

        thread = Thread(target=core.main)
        thread.start()
        core.state = CoreState.Starting
        thread.join(timeout=30)

        assert not thread.is_alive()
        assert len(core.data) >= 3
```

- [ ] **Step 2: Modify Core.__init__ to accept tiled_publisher**

In `tsuchinoko/core/__init__.py`, add `tiled_publisher=None` parameter to `__init__`:

```python
    def __init__(self,
                 execution_engine: ExecutionEngine = None,
                 adaptive_engine: AdaptiveEngine = None,
                 compute_metrics: bool = True,
                 nats_config=None,
                 tiled_publisher=None):
```

Add at end of __init__:
```python
        self._tiled_publisher = tiled_publisher
```

- [ ] **Step 3: Add tiled publication to experiment_iteration**

In `experiment_iteration()`, after the training block (after `self.adaptive_engine.train()`), add:

```python
            if self._has_fresh_data:
                with log_time('training', cumulative_key='training'):
                    self.adaptive_engine.train()

                # Publish GP outputs to Tiled
                if self._tiled_publisher:
                    try:
                        targets = getattr(self, '_last_targets', [])
                        self._tiled_publisher.write_iteration(
                            self.data._completed_iterations,
                            self.adaptive_engine,
                            targets=np.asarray(targets) if targets else np.array([]),
                        )
                        self.emit_event("tsuchinoko.gp.updated", {
                            "iteration": self.data._completed_iterations,
                        })
                    except Exception as e:
                        logger.warning(f"Tiled publication failed: {e}")
```

Also, store targets when they're computed (add `self._last_targets = targets` after targets are determined, around line where `targets = self.adaptive_engine.request_targets(position)` is called).

Add `import numpy as np` to the top of the file if not present.

- [ ] **Step 4: Add --tiled-url to CLI**

In `tsuchinoko/cli.py`, add option to `run` command:

```python
@click.option("--tiled-url", default="", help="Tiled server URL. Empty = no Tiled publication.")
```

Add it to the function signature and config handling:

```python
def run(nats_url, lucid_prefix, tiled_url, config_path):
```

After the CLI flag override section:
```python
    if tiled_url:
        config.tiled.url = tiled_url
```

- [ ] **Step 5: Run tests — verify pass**

Run: `cd /c/Users/rp/PycharmProjects/tsuchinoko-phase1 && .venv/Scripts/python -m pytest tests/test_core_tiled.py tests/test_headless.py -v --noconftest`

Expected: All PASS (core_tiled + headless regression).

- [ ] **Step 6: Commit**

```bash
git add tsuchinoko/core/__init__.py tsuchinoko/cli.py tests/test_core_tiled.py
git commit -m "feat: integrate TiledPublisher into Core experiment loop"
```

---

## Task 6: Full validation

**Files:**
- None new — run the complete test suite

- [ ] **Step 1: Run all unit tests**

Run: `cd /c/Users/rp/PycharmProjects/tsuchinoko-phase1 && .venv/Scripts/python -m pytest tests/test_tiled_config.py tests/test_tiled_reader.py tests/test_tiled_writer.py tests/test_lucid_engine.py tests/test_core_tiled.py tests/test_core_rewrite.py tests/test_headless.py tests/test_parameter_tree.py tests/test_gpcam_engine.py tests/test_state_machine.py tests/test_nats_config.py tests/test_nats_client.py tests/test_nats_service.py -v --noconftest`

Expected: All PASS. This is the complete Phase 1 + 2 + 3 validation.

- [ ] **Step 2: Update design doc status**

In `docs/design/2026-04-12-phase3-tiled-integration.md`, change `**Status:** Draft` to `**Status:** Phase 3 complete`.

- [ ] **Step 3: Commit**

```bash
git add docs/design/2026-04-12-phase3-tiled-integration.md
git commit -m "docs: mark Phase 3 (Tiled integration) complete"
```

---

## Verification Checklist

- [ ] TiledReader reads new rows incrementally from Tiled primary stream
- [ ] TiledPublisher writes per-iteration sub-containers to adaptive stream
- [ ] TiledPublisher skips posterior arrays for dimensionality > 3
- [ ] LUCIDEngine.update_targets publishes to NATS
- [ ] LUCIDEngine.get_measurements blocks until signaled, then reads from Tiled
- [ ] Core with tiled_publisher writes GP outputs after each iteration
- [ ] Core without tiled_publisher works exactly as before (no regression)
- [ ] All Phase 1 + 2 tests still pass
- [ ] CLI accepts --tiled-url option
