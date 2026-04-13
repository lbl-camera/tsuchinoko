# Phase 2: NATS Integration — Design Spec

**Status:** Phase 2 complete
**Date:** 2026-04-12
**Authors:** Ron Pandolfi, Ayaka (Claude)
**Parent:** `docs/design/2026-04-12-tsuchinoko-rescope.md`

## Goal

Add a NATS client to Tsuchinoko so it can be remotely controlled over the NATS bus
by LUCID, Claude, or other tools. Rewrite Core's `_main()` to properly use async.
Build the self-describing service infrastructure that enables the future generic
NATS-MCP bridge (Decision #9 from the rescope design doc).

The experiment itself still runs locally with `SimpleEngine` for testing. The
NATS-based measurement handoff (publish targets → LUCID measures → read from Tiled)
is a `LUCIDEngine` (`ExecutionEngine` subclass) that will be built in Phase 3 on
top of this NATS infrastructure.

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  Core                            │
│                                                  │
│  ┌──────────────┐   ┌────────────────────────┐  │
│  │ Experiment    │   │ _main() async loop     │  │
│  │ Thread        │   │                        │  │
│  │               │──▶│ drain event_queue      │  │
│  │ (sync)        │   │ handle state           │  │
│  │               │   │ sleep                  │  │
│  └──────────────┘   └───────────┬────────────┘  │
│                                  │               │
└──────────────────────────────────┼───────────────┘
                                   │
                          ┌────────┴────────┐
                          │   NATSClient    │
                          │                 │
                          │ connect/close   │
                          │ publish (safe)  │
                          │ subscribe       │
                          │ request/reply   │
                          │ auth state      │
                          └────────┬────────┘
                                   │
                          ┌────────┴────────┐
                          │  NATSService    │
                          │                 │
                          │ action handlers │
                          │ event publish   │
                          │ discovery       │
                          │ meta.actions    │
                          │ meta.events     │
                          └────────┬────────┘
                                   │
                              NATS Bus
```

### Module Responsibilities

**`tsuchinoko/nats/client.py` — NATSClient**

Shared NATS connection owner. Handles:
- Connection lifecycle (connect, reconnect, close)
- Auth handshake with LUCID (request/reply on `{lucid_prefix}.auth.request`)
- Auth state caching (approved/denied per prefix, session-scoped)
- Thread-safe `publish()` for the experiment thread (via `run_coroutine_threadsafe`)
- Async `subscribe()`, `request()`, `unsubscribe()`
- Connection status tracking

Does NOT know about Tsuchinoko's domain concepts (experiments, engines, etc.).

**`tsuchinoko/nats/service.py` — NATSService**

Tsuchinoko's identity on the bus. Handles:
- Action registration on `tsuchinoko.*` subjects
- Discovery responses (`_tsuchinoko.discover`)
- Meta endpoints (`tsuchinoko.meta.actions`, `tsuchinoko.meta.events`)
- Inbound action dispatch → Core state mutations
- Outbound event publishing from experiment thread via queue

Knows about Core's state machine and adaptive engine parameters. Receives a
reference to Core and NATSClient at construction.

**`tsuchinoko/nats/config.py` — NATSConfig**

Pydantic model:

```python
class NATSConfig(BaseModel):
    url: str = ""                        # Empty = NATS disabled
    lucid_prefix: str = "als.7011"       # LUCID instance to authenticate with
    app_name: str = "tsuchinoko"
    app_version: str = ""                # Populated from __version__ at runtime
    auth_timeout: float = 70.0           # >60s for LUCID trust dialog
    connect_timeout: float = 5.0
    reconnect: bool = True
```

Integrated into `AppConfig`:

```python
class AppConfig(BaseModel):
    nats: NATSConfig = Field(default_factory=NATSConfig)
    core: CoreConfig = Field(default_factory=CoreConfig)
```

`NetworkConfig` and `UIConfig` remain for backward compatibility but are unused
by the headless path.

## Core Rewrite: `_main()`

The current `_main()` is an async polling loop that barely uses async — it checks
state in a while loop and calls `await notify_clients()` which just sleeps. This
was left over from ZMQCore's async socket polling.

The rewrite makes `_main()` a proper async loop with real async work:

```python
async def _main(self) -> None:
    # Connect to NATS if configured
    if self._nats_config and self._nats_config.url:
        await self._nats_client.connect(self._nats_config)
        await self._nats_service.start()

    try:
        while self.state != CoreState.Exiting:
            # Drain outbound events from experiment thread
            await self._drain_events()

            # State transition logic
            if self.state == CoreState.Starting:
                if not len(self.data):
                    self.data = Data(dimensionality=self.adaptive_engine.dimensionality)
                self.adaptive_engine.reset()
                self.experiment_thread = threading.Thread(
                    target=self.experiment_loop, daemon=True
                )
                self.experiment_thread.start()
                self.state = CoreState.Running

            elif self.state == CoreState.Pausing:
                self.state = CoreState.Paused

            elif self.state == CoreState.Resuming:
                self.state = CoreState.Running

            elif self.state == CoreState.Stopping:
                self.state = CoreState.Inactive
                self.data = Data()

            await asyncio.sleep(0.05)
    finally:
        if self._nats_service:
            await self._nats_service.stop()
        if self._nats_client:
            await self._nats_client.close()
```

### Event Queue

The experiment thread cannot call async `publish()` directly. Instead it puts
events on a thread-safe `asyncio.Queue`:

```python
# In experiment thread (sync):
self._event_queue.put_nowait(("tsuchinoko.state", {"state": "running", "iteration": 42}))

# In _main() (async):
async def _drain_events(self):
    while not self._event_queue.empty():
        subject, payload = self._event_queue.get_nowait()
        if self._nats_client and self._nats_client.is_connected:
            await self._nats_client.publish(subject, payload)
```

When NATS is not configured, the queue is never drained (or events are simply
discarded). No performance cost.

### Thread-Safe Publish (for LUCIDEngine, Phase 3)

The NATSClient also exposes a sync `publish_threadsafe()` for cases where the
experiment thread needs to publish and doesn't want to go through the queue
(e.g., LUCIDEngine sending targets and waiting for a reply):

```python
def publish_threadsafe(self, subject: str, payload: dict) -> None:
    """Publish from any thread. Fire-and-forget."""
    asyncio.run_coroutine_threadsafe(
        self._publish(subject, payload), self._loop
    )

async def request_threadsafe(self, subject: str, payload: dict, timeout: float) -> dict:
    """Request/reply from any thread. Blocks caller."""
    future = asyncio.run_coroutine_threadsafe(
        self._request(subject, payload, timeout), self._loop
    )
    return future.result(timeout=timeout + 1)
```

These use Core's asyncio loop (`self._loop`), which is running in the main thread
via `loop.run_until_complete(_main())`.

## NATS Interface

### Authentication

On startup, if a `lucid_prefix` is configured, Tsuchinoko authenticates with
LUCID using the standard IPC protocol:

```
Tsuchinoko → {lucid_prefix}.auth.request
  {"app_name": "tsuchinoko", "app_version": "1.0.0"}

LUCID → reply (ephemeral inbox)
  {"status": "approved", "tiled_token": "<jwt>", "tiled_url": "https://..."}
```

The Tiled credentials are stored in NATSClient for use by the future LUCIDEngine
(Phase 3). Auth state is cached per prefix — re-auth only on reconnect.

If LUCID is not available or denies access, Tsuchinoko logs a warning and
continues without NATS. It can still run local experiments.

### Actions (inbound — Tsuchinoko receives)

Registered as NATS subscriptions. Each handler is an async callback that receives
the message, deserializes JSON, mutates Core state, and replies with JSON.

| Subject | Request Schema | Reply Schema | Handler |
|---------|---------------|--------------|---------|
| `tsuchinoko.experiment.configure` | `{engine, dimensionality, parameter_bounds, ...}` | `{status, config}` | Set adaptive engine params |
| `tsuchinoko.experiment.start` | `{motors?, detectors?}` | `{status, state}` | `state = Starting` |
| `tsuchinoko.experiment.pause` | `{}` | `{status, state}` | `state = Pausing` |
| `tsuchinoko.experiment.resume` | `{}` | `{status, state}` | `state = Resuming` |
| `tsuchinoko.experiment.stop` | `{}` | `{status, state}` | `state = Stopping` |
| `tsuchinoko.engine.set_parameter` | `{path: [str], value: any}` | `{status}` | Update parameter tree |
| `tsuchinoko.engine.get_parameters` | `{}` | `{parameters: dict}` | `saveState()` |
| `tsuchinoko.status` | `{}` | `{state, iteration, data_count, engine}` | Read-only status |

All handlers follow the same pattern:

```python
async def _handle_start(self, msg):
    try:
        self._core.state = CoreState.Starting
        reply = {"status": "ok", "state": self._core.state.name}
    except Exception as e:
        reply = {"status": "error", "message": str(e)}
    await msg.respond(json.dumps(reply).encode())
```

### Events (outbound — Tsuchinoko publishes)

Published via the event queue from the experiment thread.

| Subject | When | Payload |
|---------|------|---------|
| `tsuchinoko.state` | State machine transitions | `{state: str, iteration: int}` |
| `tsuchinoko.targets` | New targets computed | `{targets: [[x,y,...]], iteration: int}` |
| `tsuchinoko.gp.updated` | GP model updated | `{run_uid: str, iteration: int}` |
| `tsuchinoko.error` | Exception in experiment loop | `{message: str, traceback: str}` |

State events are published from `Core.state.setter` (already a centralized
transition point). Target and GP events are published from `experiment_iteration()`.
Error events are published from the exception handler in `experiment_loop()`.

### Discovery

Tsuchinoko registers as a self-describing service on the bus:

**`_tsuchinoko.discover`** — Responds to broadcast with:
```json
{
  "instance_id": "<uuid>",
  "app_name": "tsuchinoko",
  "app_version": "1.0.0",
  "prefix": "tsuchinoko",
  "actions_count": 8,
  "events_count": 4,
  "state": "running"
}
```

**`tsuchinoko.meta.actions`** — Returns JSON array of action descriptors:
```json
[
  {
    "suffix": "experiment.start",
    "description": "Begin the adaptive loop",
    "schema": {"type": "object", "properties": {"motors": {...}, "detectors": {...}}}
  },
  ...
]
```

**`tsuchinoko.meta.events`** — Returns JSON array of event descriptors:
```json
[
  {
    "suffix": "state",
    "description": "State machine transitions",
    "schema": {"type": "object", "properties": {"state": {"type": "string"}, ...}}
  },
  ...
]
```

This follows the same convention as LUCID's `_lucid.discover` /
`{prefix}.meta.actions` / `{prefix}.meta.events`, enabling a future generic
NATS-MCP bridge to auto-discover Tsuchinoko.

## NATS Is Optional

NATS adds no import cost and no runtime cost when disabled:

- `NATSConfig.url = ""` (default) → no NATS connection attempted
- `_main()` skips NATS connect/service start
- Event queue draining is a no-op (queue stays empty or events are discarded)
- All existing tests continue to work unchanged
- `SimpleEngine` + `Core()` with no NATS config = same behavior as Phase 1

## CLI Integration

The existing `tsuchinoko/cli.py` placeholder gets a real `run` command:

```
tsuchinoko run --nats-url nats://localhost:4222 --lucid-prefix als.7011
tsuchinoko run                    # No NATS, local experiment only
tsuchinoko run --config setup.yaml  # Load from file
```

Config file (YAML) maps directly to `AppConfig`:

```yaml
nats:
  url: "nats://localhost:4222"
  lucid_prefix: "als.7011"
core:
  sleep_for_fresh_data: 0.1
```

## Testing

### Unit Tests (no broker)

Test the NATS modules with a mock `nats.NATS` connection object.

**`tests/test_nats_client.py`:**
- Connection lifecycle (connect, close, reconnect flag)
- Auth handshake: approved → stores token, denied → logs warning
- Auth state caching (don't re-auth on second call)
- `publish()` serializes and calls `nc.publish()`
- `publish_threadsafe()` dispatches to event loop
- Connection status tracking

**`tests/test_nats_service.py`:**
- Action handler registration (correct subjects)
- Each handler: receives JSON, mutates Core state, replies with JSON
- Discovery response format
- Meta actions/events response format
- Event queue: experiment thread puts, service publishes

**`tests/test_core_rewrite.py`:**
- `_main()` starts and stops cleanly without NATS
- `_main()` with NATS config calls connect/start/stop
- Event queue draining publishes accumulated events
- State transitions still work (existing test_core.py patterns)
- Exit cleans up NATS connection

### Integration Tests (real broker, skipped if unavailable)

Require NATS server at `localhost:4222`. Skipped with
`pytest.mark.skipif(not nats_available)`.

**`tests/test_nats_integration.py`:**
- Mock LUCID fixture (handles auth.request, echoes commands)
- Full auth handshake: connect → auth → approved → token stored
- Action round-trip: send `tsuchinoko.experiment.start` → receive reply with state
- Event reception: start experiment → subscribe to `tsuchinoko.state` → receive state events
- Discovery: broadcast `_tsuchinoko.discover` → receive response
- Meta endpoints: request `tsuchinoko.meta.actions` → receive action list
- Reconnect behavior: disconnect mock LUCID → reconnect → actions still work

## Dependencies

Add to `pyproject.toml` core dependencies:

```toml
"nats-py>=2.0",
```

`nats-py` is pure Python, async-native, zero external dependencies. Safe for
headless environments and Briefcase packaging.

## File Structure

### New files

| File | Responsibility |
|------|----------------|
| `tsuchinoko/nats/__init__.py` | Package init, exports NATSClient, NATSService, NATSConfig |
| `tsuchinoko/nats/client.py` | Connection, auth, publish/subscribe |
| `tsuchinoko/nats/service.py` | Action handlers, discovery, event publishing |
| `tsuchinoko/nats/config.py` | NATSConfig Pydantic model |
| `tests/test_nats_client.py` | Unit tests for NATSClient |
| `tests/test_nats_service.py` | Unit tests for NATSService |
| `tests/test_nats_integration.py` | Integration tests (real broker) |
| `tests/test_core_rewrite.py` | Tests for rewritten _main() |

### Modified files

| File | Changes |
|------|---------|
| `tsuchinoko/core/__init__.py` | Rewrite `_main()`, add event queue, NATS lifecycle |
| `tsuchinoko/config.py` | Add NATSConfig to AppConfig |
| `tsuchinoko/cli.py` | Real `run` command with NATS options |
| `pyproject.toml` | Add `nats-py>=2.0` dependency |

### Unchanged

All adaptive engines, execution engines, parameter tree, state machine, existing
tests — Phase 1 deliverables are untouched.
