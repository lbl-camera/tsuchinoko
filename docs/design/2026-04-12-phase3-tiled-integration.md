# Phase 3: Tiled Integration — Design Spec

**Status:** Draft
**Date:** 2026-04-12
**Authors:** Ron Pandolfi, Ayaka (Claude)
**Parent:** `docs/design/2026-04-12-tsuchinoko-rescope.md`

## Goal

Add Tiled reader and writer modules so Tsuchinoko can read measurement data from
LUCID's Tiled runs and write GP outputs back to the same runs. Build `LUCIDEngine`
as an `ExecutionEngine` subclass that coordinates measurement handoff over NATS+Tiled.
This completes the data pipeline between Tsuchinoko and LUCID.

Scope: Tsuchinoko side only. The LUCID-side adaptive plan is out of scope.

## Architecture

```
                        NATS Bus
               ┌──────────┴──────────┐
               │                      │
         ┌─────┴──────┐        ┌─────┴──────────┐
         │   LUCID     │        │  Tsuchinoko     │
         │             │        │                 │
         │ adaptive    │        │ ┌─────────────┐ │
         │ plan        │◀──────▶│ │ LUCIDEngine │ │
         │             │ NATS   │ │ (targets,   │ │
         │ TiledWriter │        │ │  measured)  │ │
         └─────┬──────┘        │ └──────┬──────┘ │
               │                │        │        │
               │   write        │   read │        │
               │   primary      │   primary       │
               │                │        │        │
         ┌─────┴───────────────┴────────┴──┐     │
         │            Tiled                 │     │
         │                                  │◀────┘
         │  run[uid]/primary   (LUCID)      │  write
         │  run[uid]/adaptive  (Tsuchinoko) │  adaptive
         └──────────────────────────────────┘
```

## Module Responsibilities

### `tsuchinoko/tiled/reader.py` — TiledReader

Reads measurement data from a Tiled run's `primary` stream. Stateful — tracks
the number of rows already consumed so each call returns only new data.

```python
class TiledReader:
    def __init__(self, tiled_client, run_uid: str, motor_names: list[str],
                 detector_name: str, variance: float = 1.0):
        ...

    def read_new(self) -> list[tuple]:
        """Read rows added since last call.

        Returns list of (position_tuple, value, variance, metrics_dict).
        """
```

Converts Tiled's column-oriented data (xarray/pandas from `run["primary"]["data"]`)
into the `(position, value, variance, metrics)` tuple format that `Data.inject_new()`
expects. Handles the mapping from motor/detector column names to positional tuples.

### `tsuchinoko/tiled/writer.py` — TiledPublisher

Writes GP outputs to a Tiled run's `adaptive` stream using per-iteration
sub-containers. Each iteration creates an immutable snapshot.

```python
class TiledPublisher:
    def __init__(self, tiled_client, run_uid: str, dimensionality: int):
        ...

    def write_config(self, engine) -> None:
        """Write evaluation grid (once, at experiment start)."""

    def write_iteration(self, iteration: int, engine, data) -> None:
        """Write GP outputs for one iteration.

        Creates adaptive/iter_NNN/ sub-container with:
        - posterior_mean (Nx, Ny) — if dimensionality <= 3
        - posterior_variance (Nx, Ny)
        - acquisition_function (Nx, Ny)
        - hyperparameters (K,)
        - targets (M, D)
        """
```

Extracts GP state from the adaptive engine's optimizer. For GPCAMInProcessEngine,
this means calling `optimizer.posterior_mean()`, `optimizer.posterior_covariance()`,
etc. on an evaluation grid. For engines without GP state (Grid, Random), this is
a no-op.

### `tsuchinoko/execution/lucid.py` — LUCIDEngine

ExecutionEngine subclass that coordinates with LUCID over NATS for measurement
execution. Fits into Core's existing experiment loop without changes to the loop
itself.

```python
class LUCIDEngine(ExecutionEngine):
    def __init__(self, nats_client: NATSClient, tiled_reader: TiledReader,
                 lucid_prefix: str):
        ...

    def update_targets(self, targets: list[tuple]) -> None:
        """Publish targets to NATS for LUCID to measure."""

    def get_position(self) -> tuple:
        """Return last known target position."""

    def get_measurements(self) -> list[tuple]:
        """Block until LUCID signals measurements ready, then read from Tiled."""
```

**Blocking behavior:** `get_measurements()` waits on a `threading.Event` that is
set by a NATS subscription callback when `{prefix}.adaptive.measured` arrives. Then
it delegates to `TiledReader.read_new()` to fetch the data. This matches the
existing ExecutionEngine contract where `get_measurements()` may block.

**Experiment startup sequence:**

When `tsuchinoko.experiment.start` is received with `motors` and `detectors`:
1. LUCIDEngine sends `{prefix}.commands.plan.run` to LUCID with `plan_name: "adaptive_experiment"`
2. LUCID replies with `{run_uid}`
3. LUCIDEngine creates a TiledReader for `run[run_uid]["primary"]`
4. Core creates a TiledPublisher for `run[run_uid]["adaptive"]`
5. The experiment loop begins

### `tsuchinoko/tiled/config.py` — TiledConfig

```python
class TiledConfig(BaseModel):
    url: str = ""           # Tiled server URL (empty = disabled)
    token: str = ""         # API key / JWT (can be set by LUCID auth)
```

Integrated into AppConfig. The token may come from LUCID's auth handshake
(Phase 2 already stores `tiled_token` and `tiled_url` in NATSClient).

## Data Flow: One Adaptive Iteration

```
1. Core.experiment_iteration():
   │
   ├── engine.request_targets(position)
   │   └── returns [[x1,y1], [x2,y2]]
   │
   ├── lucid_engine.update_targets([[x1,y1], [x2,y2]])
   │   └── nats_client.publish_threadsafe("tsuchinoko.targets", {...})
   │
   ├── lucid_engine.get_measurements()   ← BLOCKS
   │   ├── wait for {prefix}.adaptive.measured event
   │   └── tiled_reader.read_new()
   │       └── returns [(pos, val, var, {}), ...]
   │
   ├── data.inject_new(measurements)
   ├── engine.update_measurements(data)
   ├── engine.train()
   │
   └── tiled_publisher.write_iteration(iteration, engine, data)
       └── writes adaptive/iter_NNN/ to Tiled
```

## Tiled Layout (from rescope doc)

One Tiled Run per experiment. LUCID writes `primary`, Tsuchinoko writes `adaptive`.

```
run[uid]/
├── primary/                    ← Written by LUCID
│   └── data/
│       ├── x_motor    (N,)
│       ├── y_motor    (N,)
│       └── detector   (N,)
│
└── adaptive/                   ← Written by Tsuchinoko
    ├── config/
    │   ├── evaluation_grid_x  (Nx,)
    │   └── evaluation_grid_y  (Ny,)
    ├── iter_001/
    │   ├── posterior_mean       (Nx, Ny)
    │   ├── posterior_variance   (Nx, Ny)
    │   ├── acquisition_function (Nx, Ny)
    │   ├── hyperparameters      (K,)
    │   └── targets              (M, D)
    └── iter_NNN/
        └── ...
```

Per-iteration sub-containers. All writes immutable. No append, no overwrite.

### High-Dimensional Experiments

For dimensionality > 3, the publisher skips posterior/acquisition arrays
(they'd be prohibitively large). Only hyperparameters and targets are written.
This matches the rescope doc's Decision #8.

## Core Changes

Minimal additions to `experiment_iteration()`:

```python
def experiment_iteration(self) -> None:
    with self.data.iteration():
        # ... existing target/measurement/training logic unchanged ...

        # After training, publish GP outputs to Tiled
        if self._tiled_publisher and self._has_fresh_data:
            try:
                self._tiled_publisher.write_iteration(
                    self.data._completed_iterations, self.adaptive_engine, self.data
                )
                self.emit_event("tsuchinoko.gp.updated", {
                    "run_uid": self._run_uid,
                    "iteration": self.data._completed_iterations,
                })
            except Exception as e:
                logger.warning(f"Tiled publication failed: {e}")
```

Core's `__init__` gains `tiled_config` parameter. The `_tiled_publisher` and
`_run_uid` are set when the experiment starts (either via the NATS start handler
or programmatically).

## NATSService Changes

The `_handle_start` action handler is extended to:
1. Accept `motors` and `detectors` in the request
2. If NATS + Tiled are configured, create a LUCIDEngine and send the plan to LUCID
3. Store the `run_uid` from LUCID's reply
4. Set up TiledReader and TiledPublisher for the run
5. Swap Core's execution engine to the new LUCIDEngine

This is the bridge between "user sends start" and "experiment runs against LUCID."

## Testing

All tests use real Tiled catalogs via `tiled.catalog.in_memory()` — no mocks.

### `tests/test_tiled_reader.py`
- Write sample data to a Tiled catalog, read it back via TiledReader
- Verify incremental reads (read_new returns only new rows)
- Verify position/value/variance tuple conversion
- Verify motor name → position mapping

### `tests/test_tiled_writer.py`
- Write GP outputs via TiledPublisher, read back and verify
- Verify per-iteration sub-container layout
- Verify config container (evaluation grid)
- Verify high-dimensional skip (no posterior arrays for D > 3)

### `tests/test_lucid_engine.py`
- LUCIDEngine with mock NATS client and real Tiled catalog
- update_targets publishes to NATS
- get_measurements blocks until event, then reads from Tiled
- get_position returns last target

### `tests/test_tiled_integration.py`
- Full loop: Core + LUCIDEngine + TiledPublisher + mock LUCID on NATS
- Verify measurements flow from Tiled primary → engine → GP → Tiled adaptive
- Skipped if NATS unavailable

## Dependencies

Add to `pyproject.toml` core dependencies:

```toml
"tiled[client]",
```

The `[client]` extra installs the Tiled HTTP client without the server components.
For testing, the full `tiled` package provides `in_memory()`.

Add to test dependencies:

```toml
"tiled[server]",
```

## File Structure

### New files
| File | Responsibility |
|------|----------------|
| `tsuchinoko/tiled/__init__.py` | Package exports |
| `tsuchinoko/tiled/reader.py` | Read measurements from Tiled primary stream |
| `tsuchinoko/tiled/writer.py` | Write GP outputs to Tiled adaptive stream |
| `tsuchinoko/tiled/config.py` | TiledConfig Pydantic model |
| `tsuchinoko/execution/lucid.py` | LUCIDEngine ExecutionEngine subclass |
| `tests/test_tiled_reader.py` | TiledReader tests with real catalog |
| `tests/test_tiled_writer.py` | TiledPublisher tests with real catalog |
| `tests/test_lucid_engine.py` | LUCIDEngine tests |
| `tests/test_tiled_integration.py` | Full loop integration tests |

### Modified files
| File | Changes |
|------|---------|
| `tsuchinoko/core/__init__.py` | Add tiled_publisher call in experiment_iteration |
| `tsuchinoko/nats/service.py` | Extend _handle_start for LUCIDEngine setup |
| `tsuchinoko/config.py` | Add TiledConfig to AppConfig |
| `tsuchinoko/cli.py` | Add --tiled-url option |
| `pyproject.toml` | Add tiled dependency |
