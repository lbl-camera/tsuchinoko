---
name: parameter-tree
description: Use when you need to read or modify live engine state in Tsuchinoko mid-run, when configure doesn't expose a field (e.g. queue length, optimizer tolerance), or when you need to understand how/when parameter changes take effect. Also explains the difference between training cost (rare, milestone-driven) and per-iteration acquisition cost (every step, O(N³) in data count).
---

# Skill: Tsuchinoko Parameter Tree

The adaptive engine's live state is stored in a **pyqtgraph `GroupParameter` tree**, not on plain Python attributes. The GUI binds widgets to it; the NATS API exposes it for headless callers.

For setting up a new experiment, prefer `tsuchinoko.experiment.configure` — it handles the common fields (parameter bounds, hyperparameter bounds + initials, training schedules, acquisition function, etc.) with shape validation, user-ref resolution, and transactional rollback on failure. This skill covers the cases configure doesn't:

1. **Live edits mid-run** without losing accumulated data — configure semantics assume `stop → Inactive → reconfigure`, which discards state. `set_parameter` writes to the running engine.
2. **Niche fields not in configure's schema** — queue length `n`, optimizer `pop_size`, `tol`, the live `method` selection.
3. **Reading the full state** — `get_parameters` returns the serialized parameter tree, useful for snapshotting or path discovery.

It also documents engine timing semantics — *when* a write you just made actually affects behavior — which apply identically whether the write came from configure or set_parameter.

## When to use configure vs set_parameter

| Field | Use |
|---|---|
| `parameter_bounds`, `dimensionality` | `configure` |
| `hyperparameters` (initial values), `hyperparameter_bounds` | `configure` |
| `global_training`, `local_training`, `mcmc_training` (schedules) | `configure` |
| `acquisition_function` (built-in name or `user:<ref>`) | `configure` |
| `noise_variances`, `x_out` | `configure` |
| Live tweaks during a running experiment (any of the above) | `set_parameter` |
| `n`, `pop_size`, `tol`, live `method` | `set_parameter` |

Configure-supported keys also silently no-op for these (engine doesn't consume them yet, even with the v8.x configure rewrite): `kernel`, `prior_mean`, `noise_function`, `training_method`, `initial_points`. There's no parameter-tree path for them either — these need engine work, not a NATS workaround. If a user requests one, tell them it's not currently effective and link to the Tsuchinoko issue tracker.

## NATS API

### Write a single field

Subject: `tsuchinoko.engine.set_parameter`

```json
{
  "path": ["hyperparameters", "hyperparameter_0_min"],
  "value": 1.0
}
```

Reply: `{"status": "ok"}` or `{"status": "error", "message": "..."}`.

The path is a list of strings — the navigation path through the parameter tree's `child(...)` calls. Each `set_parameter` is one leaf; loop the call for batch updates. Unlike `configure`, there's no shape validation — write the wrong path or value type and you'll either get an exception in the reply or silently miss.

### Read the full state

Subject: `tsuchinoko.engine.get_parameters` (no payload).

Returns `{"status": "ok", "parameters": <saveState>}`. The reply is pyqtgraph's serialized parameter tree — useful when you need to discover paths without reading engine source, or to snapshot config before a run.

### From a LUCID embedded agent

The MCP tool layer doesn't wrap `set_parameter`. From LUCID, call via the IPC service:

```python
from lucid.ipc.service import get_ipc_service
ipc = get_ipc_service()
reply = ipc.request(
    "tsuchinoko.engine.set_parameter",
    {"path": ["n"], "value": 4},
    timeout_ms=5000,
)
```

## Path reference (GPCAMInProcessEngine)

For an engine with `d` axes and `H` hyperparameters:

| Field | Path | Also writable via configure? |
|---|---|---|
| Axis lower bound, dim `i` | `("bounds", f"axis_{i}_min")` | yes (`parameter_bounds`) |
| Axis upper bound, dim `i` | `("bounds", f"axis_{i}_max")` | yes (`parameter_bounds`) |
| Hyperparameter initial value, index `j` | `("hyperparameters", f"hyperparameter_{j}")` | yes (`hyperparameters`) |
| Hyperparameter lower bound, index `j` | `("hyperparameters", f"hyperparameter_{j}_min")` | yes (`hyperparameter_bounds`) |
| Hyperparameter upper bound, index `j` | `("hyperparameters", f"hyperparameter_{j}_max")` | yes (`hyperparameter_bounds`) |
| Acquisition function (current selection) | `("acquisition_function",)` | yes (`acquisition_function`) |
| Optimization method (`global`/`local`/`hgdl`) | `("method",)` | no — set_parameter only |
| Queue length | `("n",)` | no — set_parameter only |
| Population size (global only) | `("pop_size",)` | no — set_parameter only |
| Tolerance | `("tol",)` | no — set_parameter only |
| Global training milestones | children of `("global_training",)` | yes (`global_training`) — replaces whole schedule |
| Local training milestones | children of `("local_training",)` | yes (`local_training`) — replaces whole schedule |
| MCMC training milestones | children of `("mcmc_training",)` | yes (`mcmc_training`) — replaces whole schedule |

For the hyperparameter count `H`: the default kernel is anisotropic Matérn, so `H = 1 + d` (signal variance + one length scale per axis). `H` is fixed at engine construction — see gpCAM's `kernel-designer` skill for the count under other kernels. If you need a different `H`, you must rebuild the engine, not just reconfigure.

## Training schedule

`GPCAMInProcessEngine` triggers hyperparameter training at iteration **milestones**, not every step. Defaults:

| Method | Default milestones |
|---|---|
| `global` | `(20, 50, 100, 400, 1000)` |
| `local`  | `(20, 40, 60, 80, 100, 200, 400, 1000)` |
| `mcmc`   | `()` (off) |

Each `N` fires **exactly once**: `engine.train()` checks `len(y_data) > N and N not in self._completed_training[method]`, then adds `N` to the completed set. So `N=20` triggers training once at the first iteration where data exceeds 20, and never again.

When training actually fires, Tsuchinoko logs:
```
Training in progress. This make take a while...
New hyperparameters: [...]
```
If you don't see those lines, no training happened on that iteration — even if the iteration was slow.

### Customizing the schedule

The clean path is `configure`:

```python
ipc.request("tsuchinoko.experiment.configure", {
    "global_training": [25, 100, 500],
    "local_training":  [30, 60, 120, 250, 500],
    "mcmc_training":   [],
})
```

Each key replaces the whole list for that method (via `TrainingParameter.setSchedule`). Configure validates entries are positive ints and rolls back on failure.

For live edits during a run (e.g. adding a near-term milestone without stopping), use `set_parameter` — but note the milestone children are dynamically named (UUIDs), so you have to fetch the tree with `get_parameters` first to find a leaf path you can overwrite. Adding *new* children programmatically isn't exposed over NATS; if you need to extend a schedule live, prefer `configure` (which replaces it wholesale) at the next `stop`.

## Per-iteration cost: training vs acquisition

This trips people up:

- **Every iteration** runs **acquisition-function optimization** — gpCAM evaluates posterior mean/variance and optimizes the acquisition function across the domain to pick the next point. This is O(N³) in data count for GP inference, plus the cost of the chosen optimizer (global is most expensive). It grows visibly as iterations accumulate, but **it is not training**.
- **Training** runs only at the milestones above, and only updates the GP's hyperparameters. It's a few times per run, not per iteration.

If iterations feel slow, check `len(data)` and whether you're using `method="global"` (slower) vs `method="local"` (faster, but local-only). Don't conclude the engine is training every step — check the log lines.

## When parameter changes take effect

Independent of whether the change came via `configure` or `set_parameter`:

- **Axis bounds and `acquisition_function`/`method`/`n`/`pop_size`/`tol`**: read on every `request_targets()` call. Changes take effect on the next iteration.
- **Hyperparameter initial values**: read by `init_optimizer()`, which `reset()` calls. `reset()` runs on construction and on the `Stopping → Inactive → Starting` transition. **Changing initial hyperparameters mid-run does not re-seed the GP — those changes will be picked up at the next `Starting`.** (Note: `_set_hyperparameter` no-ops cleanly when the GP isn't built yet, so writing initials before `bind_run` is safe and they'll be applied at `init_gp()` time.)
- **Hyperparameter bounds**: read inside `train()` — take effect on the next training milestone. (No need to restart.)
- **Training-schedule milestones**: read inside `train()` — take effect immediately. Lowering a future milestone (e.g., adding `N=25` while at iteration 24) will trigger training on the next iteration.

## Common workflow

Configuring a new experiment with non-default hyperparameter bounds:

```python
# 1. Stop and wait for Inactive (configure won't reset state otherwise).
ipc.request("tsuchinoko.experiment.stop", {})
# poll tsuchinoko.status until state == "Inactive"

# 2. One configure call now covers axes, hyperparameter initials/bounds,
#    training schedules, and the acquisition function.
ipc.request("tsuchinoko.experiment.configure", {
    "parameter_bounds":     [[0, 5], [0, 5]],
    "dimensionality":       2,
    "hyperparameters":      [100.0, 4.0, 4.0],
    "hyperparameter_bounds": [[1, 2000], [1, 20], [1, 20]],
    "acquisition_function": "variance",
    # optional: tighter training cadence
    # "local_training":      [15, 30, 60, 120],
})

# 3. If you need niche knobs not in configure (queue length, pop size, tol,
#    or the live optimization method), use set_parameter:
ipc.request("tsuchinoko.engine.set_parameter", {"path": ["n"], "value": 1})

# 4. Bind a run and start (in LUCID: ncs_run_plan "adaptive_experiment").
```

After the first training milestone fires, follow gpCAM's `experiment-designer` § "Diagnosing Trained Hyperparameters" — if anything pegged at a bound, stop, widen via another `configure` call, and re-run.
