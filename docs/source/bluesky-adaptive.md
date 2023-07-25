# Bluesky-Adaptive Integration

[Bluesky-Adaptive](https://github.com/bluesky/bluesky-adaptive) is Bluesky companion package for tightly integrated
adaptive scans. Tsuchinoko may integrate with Bluesky-Adaptive by serving as the source of an `Agent`.

Running Tsuchinoko with Bluesky-Adaptive requires a `TsuchinokoAgent` between the Tsuchinoko server and the Bluesky 
RunEngine. The `TsuchinokoAgent` abstract base class provides a concise interface requiring the same components as a
Bluesky-Adaptive `Agent`.

```{eval-rst}
   
.. autoclass:: tsuchinoko.execution.bluesky_adaptive.BlueskyAdaptiveEngine
    :members:
    
.. autoclass:: tsuchinoko.execution.bluesky_adaptive.TsuchinokoAgent
    :members:
    :exclude-members: ask, tell
```