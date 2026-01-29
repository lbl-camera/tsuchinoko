# Signal and Callback Patterns in Tsuchinoko

This document describes the two communication patterns used in Tsuchinoko
and provides guidance on when to use each.

## Overview

Tsuchinoko uses two complementary patterns for component communication:

1. **Qt Signals**: For UI component communication and thread-safe events
2. **Callback Registry**: For network response dispatch

## Qt Signals

### When to Use

- Communication between Qt widgets
- Events that need thread-safe delivery to the main thread
- User-initiated actions (button clicks, menu selections)
- State change notifications within the UI

### Signal Locations

| Class | Signal | Purpose |
|-------|--------|---------|
| `Configuration` | `sigRequestParameters` | Request params from server |
| `Configuration` | `sigPushParameter` | Push changed param to server |
| `StateManager` | `sigStart` | Start experiment |
| `StateManager` | `sigStop` | Stop experiment |
| `StateManager` | `sigPause` | Pause experiment |
| `StateManager` | `sigReplay` | Replay experiment |
| `StateManager` | `sigSetComputeMetrics` | Toggle metrics |
| `GraphManager` | `sigPush` | Push graph to server |
| `RequestRelay` | `sigRequestMeasure` | Request measurement at position |
| `GraphSignalRelay` | `sigPush` | Relay graph push from graph items |

### Connection Pattern

```python
# In MainWindow.__init__
self.state_manager_widget.sigStart.connect(self.network.start_experiment)
self.configuration_widget.sigPushParameter.connect(self.network.set_parameter)
```

## Callback Registry

### When to Use

- Handling network responses from the server
- Multiple subscribers need the same response
- Responses need optional main-thread invocation

### Registry Location

`NetworkManager.callbacks` - Dict mapping response types to callback lists

### Registration Pattern

```python
# Subscribe to a response type
self.network.subscribe(
    self.state_manager_widget.update_state,
    StateResponse
)

# Subscribe with main-thread invocation
self.network.subscribe(
    self.configuration_widget.update_parameters,
    GetParametersResponse,
    invoke_as_event=True
)

# Unsubscribe
self.network.unsubscribe(callback, ResponseType)
```

### Response Dispatch

In `NetworkManager._handle_response`:

```python
for callback, as_event in self.callbacks[type(response)]:
    if as_event:
        invoke_as_event(callback, *response.payload)
    else:
        callback(*response.payload)
```

## Guidelines

### Use Qt Signals When

1. The sender is a Qt widget
2. The receiver needs thread-safe delivery
3. You want loose coupling via signal/slot
4. The event originates from user interaction

### Use Callback Registry When

1. Handling server responses
2. Multiple handlers need the same response
3. You need `invoke_as_event` control

### Avoid Mixing Patterns

Don't emit a Qt signal that then registers a callback. Keep the patterns
separate to maintain clarity.

## Future Consolidation

Consider consolidating to signals-only by:

1. Creating response signals on NetworkManager
2. Connecting response signals to handlers
3. Emitting signals in _handle_response instead of callback dispatch

This would simplify the architecture to a single pattern.
