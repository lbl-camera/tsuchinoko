# Tsuchinoko Rescope: Headless Adaptive Experiment Service

**Status:** Phase 1 complete
**Date:** 2026-04-12
**Authors:** Ron Pandolfi, Ayaka (Claude)

## Problem Statement

Tsuchinoko is a standalone Qt application for GP-based adaptive experiments. LUCID is the
beamline control system with visualization, AI integration, and a NATS IPC bus. Maintaining
two Qt GUIs with overlapping visualization systems is unsustainable. Meanwhile, LUCID
explicitly avoids embedding data processing — a hard boundary learned from Xi-CAM, where
integrating analysis code snowballed into an unmanageable mess.

Tsuchinoko's adaptive engine is valuable. Its GUI is not. The goal is to turn Tsuchinoko
into a headless service on the NATS bus, where LUCID provides visualization and experiment
control, and Tiled is the shared data substrate.

## Design Principles

1. **No processing in LUCID.** GP training, acquisition function evaluation, posterior
   computation — all of it stays in Tsuchinoko. LUCID renders arrays from Tiled. Period.

2. **Tiled is the data contract.** Tsuchinoko publishes computed results to Tiled.
   LUCID reads them. Neither system needs to understand the other's internals.

3. **NATS is the coordination layer.** Tsuchinoko is a "tool" on the NATS bus, like any
   other domain-specialized service. It authenticates with LUCID, sends measurement
   targets, and receives run IDs.

4. **Tsuchinoko owns experiment design.** The GP configuration, parameter bounds,
   acquisition strategy, and engine selection all live in Tsuchinoko. LUCID provides
   runtime controls (start/pause/stop) and visualization.

5. **bluesky-adaptive is not a dependency.** Its maintainer has left and the project is
   unreliable. Tsuchinoko replaces its role with a cleaner interface.

## Architecture

```
                           NATS Bus
                    ┌─────────┴─────────┐
                    │                     │
              ┌─────┴──────┐      ┌──────┴───────┐
              │   LUCID     │      │  Tsuchinoko   │
              │             │      │  (headless)    │
              │ - RunEngine │      │               │
              │ - Viz panels│      │ - GP engine    │
              │ - AI panel  │      │ - Experiment   │
              │ - Controls  │      │   loop         │
              └─────┬──────┘      └──────┬────────┘
                    │                     │
                    │   read/write        │  read/write
                    └────────┬────────────┘
                             │
                      ┌──────┴──────┐
                      │    Tiled     │
                      └─────────────┘
```

### Data Flow: One Adaptive Iteration (within a single Run)

```
1. Tsuchinoko computes next measurement targets
        │
        ▼
2. Tsuchinoko publishes targets via NATS
   Subject: tsuchinoko.targets
        │
        ▼
3. LUCID's adaptive plan receives targets, executes move-and-measure
   (all within the same long-running bluesky Run)
        │
        ▼
4. TiledWriter appends measurement events to Tiled (same run, primary stream)
   LUCID notifies: {prefix}.adaptive.measured {run_uid, iteration}
        │
        ▼
5. Tsuchinoko reads new measurement rows from Tiled
   client[run_uid]["primary"] — read latest rows
        │
        ▼
6. Tsuchinoko updates internal GP model (hot data, in-memory)
   Trains hyperparameters if scheduled
        │
        ▼
7. Tsuchinoko writes GP outputs to the same Tiled run (adaptive stream)
   Publishes: tsuchinoko.gp.updated {run_uid, iteration}
        │
        ▼
8. LUCID visualization panels read updated GP outputs from Tiled
   Renders posterior mean, acquisition function, etc.
        │
        ▼
   → back to step 1
```

## NATS Interface

Tsuchinoko has its **own NATS namespace** (`tsuchinoko.*`), separate from LUCID's
`{prefix}.*` namespace. The interface is stable and does not need LUCID's pluggable
action registration system.

Tsuchinoko authenticates with LUCID using the standard IPC protocol
(`ncs/docs/superpowers/specs/2026-04-09-ipc-design.md`) to obtain Tiled credentials,
then operates independently on the bus.

### Authentication

Tsuchinoko authenticates with LUCID on startup:

```
Tsuchinoko → {prefix}.auth.request
  {app_name: "tsuchinoko", app_version: "x.y.z"}

LUCID → reply (ephemeral inbox)
  {status: "approved", tiled_token: "<jwt>", tiled_url: "https://..."}
```

On approval, Tsuchinoko receives Tiled credentials and connects to the catalog.

### Actions Tsuchinoko Receives (inbound)

Commands that LUCID, Claude, or other tools can send to Tsuchinoko.

| Subject | Description | Schema |
|---------|-------------|--------|
| `tsuchinoko.experiment.configure` | Set up experiment parameters | See §Experiment Configuration |
| `tsuchinoko.experiment.start` | Begin the adaptive loop | `{}` |
| `tsuchinoko.experiment.pause` | Pause the loop | `{}` |
| `tsuchinoko.experiment.resume` | Resume from pause | `{}` |
| `tsuchinoko.experiment.stop` | Stop and finalize | `{}` |
| `tsuchinoko.engine.set_parameter` | Update a single engine parameter | `{path: str, value: any}` |
| `tsuchinoko.engine.get_parameters` | Retrieve current engine parameters | `{}` |
| `tsuchinoko.status` | Query current state and progress | `{}` |

### Events Tsuchinoko Publishes (outbound)

| Subject | Description | Schema |
|---------|-------------|--------|
| `tsuchinoko.state` | State changes | `{state: str, iteration: int}` |
| `tsuchinoko.targets` | New targets computed | `{targets: [[x,y,...]], iteration: int}` |
| `tsuchinoko.gp.updated` | GP outputs written to Tiled | `{run_uid: str, iteration: int}` |
| `tsuchinoko.error` | Error in adaptive loop | `{message: str, traceback: str}` |

### Subscribes To

| Subject | Purpose |
|---------|---------|
| `{prefix}.runs.complete` | Detect when LUCID finishes executing a measurement |
| `{prefix}.runs.new` | Track run starts for UI feedback |

## Experiment Lifecycle & Measurement Handoff

**Decision:** An entire autonomous experiment is a **single bluesky Run** — one start
doc, many measurement events, one stop doc. LUCID manages a long-running adaptive plan
that internally coordinates with Tsuchinoko over NATS. Tsuchinoko writes GP outputs
to the same Tiled run in a separate stream.

All plan execution goes through LUCID, including future queueserver support.

### Sequence: Full Experiment

```
EXPERIMENT START
================

1. User (or Claude) → tsuchinoko.experiment.configure
     {engine: "gpCAM", dimensionality: 2, parameter_bounds: [...], ...}

2. User (or Claude) → tsuchinoko.experiment.start
     {motors: ["x_motor", "y_motor"], detectors: ["detector"]}

3. Tsuchinoko → {prefix}.commands.plan.run
     {
       plan_name: "adaptive_experiment",
       params: {
         experiment_id: "<uuid>",
         motors: ["x_motor", "y_motor"],
         detectors: ["detector"]
       }
     }

4. LUCID starts the plan, emits start document, replies:
     {status: "started", run_uid: "<tiled_uid>"}

   This single run_uid is used for the entire experiment.
   Tsuchinoko stores it for all subsequent Tiled reads/writes.


ADAPTIVE LOOP (repeats)
=======================

5. Tsuchinoko computes next targets, sends:
     tsuchinoko.targets {run_uid, targets: [[x1,y1], ...], iteration: 42}

6. LUCID's adaptive plan receives targets, executes:
     for target in targets:
         yield from mv(*interleave(motors, target))
         yield from trigger_and_read(dets)

7. LUCID notifies Tsuchinoko that measurements are ready:
     {prefix}.adaptive.measured {run_uid, iteration: 42, n_new_points: 3}

8. Tsuchinoko reads new measurements from Tiled:
     run = client[run_uid]
     data = run["primary"]  # growing table, read latest rows

9. Tsuchinoko updates GP model, trains if scheduled

10. Tsuchinoko writes GP outputs to the same run's adaptive stream
    (see §Tiled Publication)

11. Tsuchinoko publishes:
      tsuchinoko.gp.updated {run_uid, iteration: 42}

12. → back to step 5


EXPERIMENT END
==============

13. User (or Claude) → tsuchinoko.experiment.stop

14. Tsuchinoko writes final GP outputs to Tiled

15. Tsuchinoko → {prefix}.adaptive.done {run_uid}
    LUCID's adaptive plan exits, emits close_run()
    TiledWriter writes stop document
```

### The Adaptive Plan (LUCID-side)

LUCID registers one generic `adaptive_experiment` plan. Beamline-specific variants
can be written as user-plans for specialized use cases (out of scope for this doc).

The plan is a long-running coroutine that:
- Opens a single bluesky run tagged with the experiment ID
- Enters a loop, waiting for target messages from Tsuchinoko via NATS
- Executes move-and-measure for each batch of targets
- Signals Tsuchinoko when measurements are written
- Exits when Tsuchinoko signals done

```python
def adaptive_experiment(dets, motors, experiment_id, nats_bridge):
    """Long-running adaptive experiment plan. Coordinates with Tsuchinoko."""
    _md = {"tsuchinoko": {"experiment_id": experiment_id}}
    yield from open_run(md=_md)

    while True:
        # Wait for targets from Tsuchinoko (via NATS callback → queue)
        msg = yield from wait_for_message(nats_bridge, "tsuchinoko.targets")
        if msg is None:  # stop signal
            break

        targets = msg["targets"]
        for target in targets:
            yield from mv(*interleave(motors, target))
            yield from trigger_and_read(dets)

        # Notify Tsuchinoko that measurements are in Tiled
        yield from notify(nats_bridge, "adaptive.measured",
                         {"iteration": msg["iteration"],
                          "n_new_points": len(targets)})

    yield from close_run()
```

**Note:** The `wait_for_message` / `notify` stubs represent NATS integration within
a bluesky plan. The exact mechanism (plan stubs wrapping NATS, or a bridge object
passed to the plan) needs to be designed during implementation. This is nontrivial —
bluesky plans are generator-based coroutines, not async, so NATS callbacks need to
be bridged into the plan's yield-based flow.

## Tiled Publication

**Decision:** One autonomous experiment = one Tiled Run. LUCID writes measurements
to `primary`. Tsuchinoko writes GP outputs to `adaptive`. Both streams grow over
the life of the experiment within the same run container.

### Multi-Writer to Same Run

The Tiled client API supports this:

```python
# LUCID creates the run via TiledWriter (start doc → primary stream grows)

# Tsuchinoko accesses the same run and creates the adaptive stream:
run = tiled_client[run_uid]
adaptive = run.create_container(key="adaptive")
```

**Key constraint:** Each process writes to distinct stream keys. LUCID writes
`primary` (and `baseline`, etc). Tsuchinoko writes `adaptive`. No conflicts.

**Verified:** LUCID's `stop` document does NOT make the run immutable. The
TiledWriter's `stop()` method only calls `root_node.update_metadata()` — the
container remains writable. Tsuchinoko can write after the stop doc if needed.

### Stream Schemas

**Measurement stream** (`primary`) — written by LUCID's TiledWriter:
Standard bluesky event stream. Grows with each measurement. Tsuchinoko reads this.

**GP output stream** (`adaptive`) — written by Tsuchinoko:

The adaptive stream contains **per-iteration snapshots** of the GP state. Each
iteration appends or overwrites arrays within the stream.

| Field | Shape | Update Pattern | Description |
|-------|-------|----------------|-------------|
| `posterior_mean` | `(Nx, Ny)` | Overwrite | GP predicted values on evaluation grid |
| `posterior_variance` | `(Nx, Ny)` | Overwrite | GP uncertainty on evaluation grid |
| `acquisition_function` | `(Nx, Ny)` | Overwrite | Acquisition function values on grid |
| `hyperparameters` | `(I, K)` | Append row | Hyperparameter values per iteration |
| `targets` | `(I, M, D)` | Append row | Targets per iteration |
| `evaluation_grid_x` | `(Nx,)` | Write once | Grid x-coordinates |
| `evaluation_grid_y` | `(Ny,)` | Write once | Grid y-coordinates |
| `iteration` | `(I,)` | Append | Iteration indices (for time axis) |

Where `I` = iterations so far, `K` = hyperparameter count, `M` = targets per
iteration, `D` = dimensionality.

**Proposed layout: per-iteration sub-containers.** Tiled's `write_array()` creates
immutable arrays — no overwrite, no append. Rather than fighting this, each iteration
creates a new sub-container with a full snapshot:

```
adaptive/
├── config/                      ← written once at experiment start
│   ├── evaluation_grid_x  (Nx,)
│   └── evaluation_grid_y  (Ny,)
├── iter_001/
│   ├── posterior_mean       (Nx, Ny)
│   ├── posterior_variance   (Nx, Ny)
│   ├── acquisition_function (Nx, Ny)
│   ├── hyperparameters      (K,)
│   └── targets              (M, D)
├── iter_002/
│   ├── posterior_mean       (Nx, Ny)
│   └── ...
└── iter_042/
    └── ...
```

Every write is immutable. No mutation, no growth, no delete-and-recreate.

- **Latest state:** LUCID reads `adaptive/iter_042/posterior_mean`
- **History:** LUCID iterates `iter_*/hyperparameters` for time-series plots
- **Crash safety:** All completed iterations are preserved; no partial overwrites

The cost is one Tiled container per iteration. For a typical experiment (tens to
low hundreds of iterations), this is manageable.

### Metadata

The run's start doc (written by LUCID's adaptive plan) includes:

```json
{
  "tsuchinoko": {
    "experiment_id": "<uuid>",
    "engine": "gpCAM",
    "engine_config": {
      "dimensionality": 2,
      "parameter_bounds": [[0, 100], [0, 50]],
      "acquisition_function": "variance",
      "kernel": "matern"
    }
  }
}
```

Tsuchinoko can update run metadata via `run.update_metadata()` to record final
iteration count, elapsed time, convergence status, etc.

### Tiled Structure

```
Tiled catalog
├── [run_uid]  ← One run = one autonomous experiment
│   ├── metadata.start.tsuchinoko.experiment_id = "<exp_uuid>"
│   ├── metadata.start.tsuchinoko.engine = "gpCAM"
│   │
│   ├── primary/                            ← Written by LUCID (TiledWriter)
│   │   ├── internal/                       ← appendable table
│   │   │   ├── detector_image  (N, 512, 512)
│   │   │   ├── x_motor         (N,)
│   │   │   └── y_motor         (N,)
│   │   └── (external assets if any)
│   │
│   └── adaptive/                           ← Written by Tsuchinoko
│       ├── config/
│       │   ├── evaluation_grid_x  (Nx,)
│       │   └── evaluation_grid_y  (Ny,)
│       ├── iter_001/
│       │   ├── posterior_mean       (Nx, Ny)
│       │   ├── posterior_variance   (Nx, Ny)
│       │   ├── acquisition_function (Nx, Ny)
│       │   ├── hyperparameters      (K,)
│       │   └── targets              (M, D)
│       ├── iter_002/
│       │   └── ...
│       └── iter_042/
│           └── ...
│
├── [other_run]  ← normal (non-adaptive) beamline run
│   └── primary/ ...
```

### How LUCID Reads GP Outputs

LUCID subscribes to `tsuchinoko.gp.updated` events. On each event:

```python
run = tiled_client[event["run_uid"]]
adaptive = run["adaptive"]
iteration = event["iteration"]

# Latest GP state
latest = adaptive[f"iter_{iteration:03d}"]
posterior = latest["posterior_mean"]  # → ArrayClient, shape (Nx, Ny)

# Hyperparameter history (iterate sub-containers)
hp_history = [adaptive[k]["hyperparameters"].read()
              for k in sorted(adaptive) if k.startswith("iter_")]
```

LUCID's lazy visualization panels render these using the standard ArrayClient →
ImageView pattern. No computation in LUCID.

### High-Dimensional Experiments

For >2D problems, the posterior array on a full grid becomes prohibitively large.
**Decision:** Tsuchinoko simply skips publishing the full posterior/acquisition
function arrays for high-dimensional experiments. The `adaptive` stream still
includes hyperparameters, targets, and other scalar/small fields. Visualization
for high-D experiments focuses on scatter plots of measurements and hyperparameter
evolution rather than heatmaps.

The dimensionality threshold for skipping grid-based outputs (e.g., D > 3) is
configurable in the experiment configuration.

### Multi-Task GP Outputs

For multi-task experiments, the `adaptive` stream includes per-task fields:

| Field | Shape | Description |
|-------|-------|-------------|
| `posterior_mean_task0` | `(Nx, Ny)` | Task 0 posterior |
| `posterior_mean_task1` | `(Nx, Ny)` | Task 1 posterior |
| `posterior_variance_task0` | `(Nx, Ny)` | Task 0 uncertainty |
| `posterior_variance_task1` | `(Nx, Ny)` | Task 1 uncertainty |

Alternatively, a single `posterior_mean` with shape `(Nx, Ny, T)` where T is the
number of tasks. TBD based on LUCID visualization plugin preferences.

## Experiment Configuration

### Schema

```json
{
  "engine": "gpCAM",
  "dimensionality": 2,
  "parameter_bounds": [[0, 100], [0, 50]],
  "hyperparameters": {
    "initial": [1.0, 10.0, 10.0],
    "bounds": [[0.01, 100], [0.01, 100], [0.01, 100]],
    "optimization": "global"
  },
  "acquisition_function": "variance",
  "training_schedule": {
    "initial_delay": 10,
    "interval": 5
  },
  "targets_per_iteration": 1,
  "publication_interval": 1,
  "evaluation_grid": {
    "resolution": [100, 50]
  }
}
```

### Configuration Sources (in priority order)

1. **NATS command** (`tsuchinoko.experiment.configure`) — from LUCID UI or Claude
2. **Config file** (YAML/JSON) — for scripted/reproducible setups
3. **CLI arguments** — for quick launches
4. **Defaults** — sensible defaults per engine type

## Execution Model

### Who Owns What

| Component | Owner | Notes |
|-----------|-------|-------|
| GP model, training, inference | Tsuchinoko | In-memory, never leaves the process |
| Target computation | Tsuchinoko | Acquisition function optimization |
| Motor moves, detector triggers | LUCID (RunEngine) | Or queueserver in future |
| Data persistence | LUCID (TiledWriter) | For measurement data |
| GP output persistence | Tsuchinoko | Writes to Tiled directly |
| Visualization | LUCID | Reads arrays from Tiled |
| Experiment lifecycle (start/stop) | Shared | LUCID sends controls, Tsuchinoko manages state |
| Experiment design (config) | Tsuchinoko | LUCID/Claude can configure via NATS |

### Queueserver Support (Future)

**Decision:** All plan execution routes through LUCID, including future queueserver.
Tsuchinoko always sends targets to `{prefix}.commands.plan.run`; LUCID decides
whether to execute via direct RunEngine or submit to queueserver. Tsuchinoko is
unaware of the execution backend.

## What Changes in Tsuchinoko

### Keep

| Module | Reason |
|--------|--------|
| `adaptive/` | All engines: gpCAM, fvGP, random, grid, quadtree |
| `core/` state machine | Clean, recently added, well-tested |
| `core/` experiment loop | Core business logic (needs adaptation for NATS/Tiled) |
| `execution/simple.py` | Useful for testing/simulation |
| `execution/threaded_in_process.py` | Same |
| Tests for above | 98% coverage on gpCAM engine |

### Drop

| Module | Reason |
|--------|--------|
| `widgets/` | LUCID is the GUI |
| `graphs/` | Visualization logic moves to LUCID plugins |
| `graphics_items/` | pyqtgraph items, replaced by LUCID viz |
| `network/` (ZMQ) | Replaced by NATS |
| `patches/` | Qt/pyqtgraph patches, no longer needed |
| PySide6 dependency | Headless service, no Qt |

### Add

| Module | Purpose |
|--------|---------|
| `nats/` | NATS client, action registration, event publishing |
| `tiled/` | Tiled reader (measurements) and writer (GP outputs) |
| `cli.py` | Click-based CLI for headless service |
| `config.py` (expand) | Extended Pydantic config for NATS, Tiled, experiment |

### Adapt

| Module | Changes |
|--------|---------|
| `core/` experiment loop | Replace ZMQ message pump with NATS subscribe/publish. Replace direct execution engine calls with "publish targets, wait for run_id, read from Tiled" cycle. |
| `execution/` | May simplify to just `SimpleEngine` for simulation/testing. Real execution goes through LUCID. Keep interface for headless testing. |
| `adaptive/` engines | No changes to engine internals. Add Tiled publication of computed outputs. |

## LUCID-Side Changes

### New: Autonomous Experiment Panel

A LUCID panel that provides runtime controls and status for a connected Tsuchinoko
instance:

- **State display:** Current iteration, GP state (idle/training/computing), engine name
- **Controls:** Start / Pause / Resume / Stop buttons (send NATS commands)
- **Parameter tuning:** Expose Tsuchinoko's engine parameters (fetched via
  `tsuchinoko.engine.get_parameters`, updated via `tsuchinoko.engine.set_parameter`)
- **Connection status:** Whether Tsuchinoko is on the bus

This panel sends NATS messages. It does not run any adaptive logic.

### New: GP Visualization Plugins

LUCID visualization plugins that know how to render GP-specific Tiled data. These are
standard LUCID visualizations — they read ArrayClients from Tiled, nothing more.

| Plugin | Reads | Renders |
|--------|-------|---------|
| PosteriorMeanView | `adaptive/posterior_mean` | 2D heatmap with measurement overlay |
| PosteriorVarianceView | `adaptive/posterior_variance` | 2D heatmap (uncertainty map) |
| AcquisitionFunctionView | `adaptive/acquisition_function` | 2D heatmap with target markers |
| HyperparameterView | `adaptive/hyperparameters` across iterations | Line plot over time |
| MeasurementScatterView | `adaptive/measurement_positions` + `values` | Colored scatter plot |

These could auto-select when LUCID detects Tiled entries with `tsuchinoko` metadata,
using the existing SelectionEngine / DocumentProcessor pattern.

### Adaptive Plan

See §Experiment Lifecycle — the `adaptive_experiment` plan is defined there. It is
a single generic plan. Beamline-specific variants are user-plans, out of scope.

## AI Integration

### Claude's Role

Claude interacts with Tsuchinoko through the NATS bus. Since Tsuchinoko has its own
namespace, it needs its own MCP bridge (or an extension to lucid-mcp-bridge that
also subscribes to `tsuchinoko.*` subjects).

Capabilities:

- **Experiment design:** "Set up a 2D GP scan over x=[0,100] y=[0,50] with variance
  acquisition" → `tsuchinoko.experiment.configure`
- **Parameter tuning:** "Increase the length scale bounds" →
  `tsuchinoko.engine.set_parameter`
- **Monitoring:** "How's the GP converging?" → read hyperparameter history from Tiled,
  interpret posterior variance trends
- **Interpretation:** "What region should we focus on?" → read acquisition function
  from Tiled, describe high-value areas
- **Lifecycle:** "Pause the adaptive loop" → `tsuchinoko.experiment.pause`

### MCP Bridge (Open — see §Open Questions)

Three options under consideration:

**Option A: Extend lucid-mcp-bridge.** Add `tsuchinoko.*` subjects to the existing
bridge. Claude sees both LUCID and Tsuchinoko tools in one MCP server.
- Pro: Single deployment, shared NATS connection, one `.mcp.json` entry.
- Con: Couples the two — Tsuchinoko interface changes require bridge updates.
  Doesn't work without LUCID.

**Option B: Standalone tsuchinoko-mcp-bridge.** Separate MCP server, separate process.
Claude sees Tsuchinoko as a distinct MCP server.
- Pro: Fully independent, works without LUCID running.
- Con: Another process to deploy and configure.

**Option C: Generic NATS-MCP bridge.** A single bridge that auto-discovers services
on the NATS bus and exposes their actions as MCP tools. Both LUCID and Tsuchinoko
register via a discovery convention; the bridge auto-generates tool definitions.
- Pro: Scales to N services without per-service bridge code. Future-proof for
  additional NATS services (ptychography, peak fitting, etc.).
- Con: More complex upfront. Needs a cross-service discovery convention.
- Note: LUCID already has `_lucid.discover` for instance discovery. Tsuchinoko
  could register similarly via `_tsuchinoko.discover` or a generic `_nats.discover`.

## Resolved Decisions

| # | Question | Decision |
|---|----------|----------|
| 1 | NATS namespace | Own namespace (`tsuchinoko.*`), not under LUCID prefix |
| 2 | Run correlation | LUCID replies with `run_uid` from start doc in plan.run response |
| 3 | GP output storage | Same Tiled run, separate `adaptive` stream |
| 4 | One run per experiment | Single bluesky Run encapsulates the entire autonomous experiment |
| 5 | Queueserver routing | Always through LUCID; Tsuchinoko is backend-agnostic |
| 6 | Adaptive plan scope | One generic plan; beamline-specific = user-plans (out of scope) |
| 7 | Stop doc immutability | Not an issue — container stays writable after stop doc |
| 8 | High-dimensional GP | Skip full posterior; show scatter + hyperparameters only |
| 9 | MCP bridge strategy | Option C: generic NATS-MCP bridge with auto-discovery |
| 10 | Tiled array updates | Per-iteration sub-containers; all writes immutable |
| 11 | Plan ↔ NATS bridge | Background thread reads NATS, buffers; plan checks buffer via stubs |
| 12 | Reading growing streams | Re-read from Tiled each iteration (no mutation dependency) |
| 13 | Multi-task shape | Per-task fields — TBD if 3D array is better |

## Open Questions

1. **Generic NATS-MCP bridge design:** What discovery convention do NATS services
   use to self-describe their actions? LUCID has `_lucid.discover`; Tsuchinoko
   would need an analogous `_tsuchinoko.discover`. Should there be a shared
   convention (e.g., `_service.discover`) that the bridge scans? This becomes
   the contract for any future NATS service to be auto-exposed to Claude.

2. **NATS ↔ plan buffer details:** The background thread buffers NATS messages
   for the bluesky plan to consume. What happens on pause/stop — does the buffer
   drain? What's the timeout if Tsuchinoko goes silent? Does the plan yield a
   special `Msg` that the RunEngine treats as "wait for external input"?
   Implementation detail, but worth spiking early.

3. **Tiled container limits:** For long experiments (hundreds of iterations), the
   `adaptive/` container will have hundreds of `iter_NNN/` sub-containers. Is
   there a practical limit on Tiled container children? If so, we may want to
   batch iterations (e.g., write every 5th iteration).

## Migration Path

### Phase 1: Extract headless core
- Factor out `tsuchinoko-core`: adaptive engines + experiment loop + state machine
- Remove Qt dependency
- Validate with `SimpleEngine` (function-based, no hardware)
- All existing adaptive engine tests must pass

### Phase 2: NATS integration
- Add NATS client conforming to LUCID IPC spec
- Implement auth handshake, action registration, event publishing
- Replace ZMQ message pump in experiment loop
- Test against LUCID IPC on a local NATS broker

### Phase 3: Tiled integration
- Add Tiled reader for measurement data (replace in-memory Data injection)
- Add Tiled writer for GP outputs (posterior, acquisition, hyperparameters)
- Define publication schema and frequency
- Test with real Tiled server

### Phase 4: LUCID visualization plugins
- Implement GP visualization plugins in LUCID
- Auto-selection for `tsuchinoko`-tagged Tiled entries
- Autonomous Experiment Panel for controls

### Phase 5: End-to-end
- Full loop: configure → start → measure → update → visualize
- Test at beamline with real hardware
- Claude-assisted experiment design via MCP
