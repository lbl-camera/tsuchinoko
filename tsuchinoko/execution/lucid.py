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

    The engine can be created before the bluesky run exists.  Call
    :meth:`bind_run` once the run UID and Tiled reader are available
    (typically from a ``tsuchinoko.experiment.bind_run`` NATS message).
    """

    def __init__(self, nats_client: NATSClient,
                 lucid_prefix: str,
                 tiled_reader: TiledReader | None = None) -> None:
        self._nats_client = nats_client
        self._tiled_reader = tiled_reader
        self._lucid_prefix = lucid_prefix
        self._position: tuple = (0, 0)
        self._measured_event = threading.Event()
        self._iteration = 0

    def bind_run(self, tiled_reader: TiledReader) -> None:
        """Bind a TiledReader for an active bluesky run."""
        self._tiled_reader = tiled_reader
        logger.info("LUCIDEngine bound to run")

    def update_targets(self, targets: List[Tuple]) -> None:
        """Publish targets to NATS for LUCID to measure."""
        self._iteration += 1
        if len(targets):
            self._position = tuple(targets[-1])

        self._nats_client.publish_threadsafe("tsuchinoko.targets", {
            "targets": [list(t) for t in targets],
            "iteration": self._iteration,
        })
        logger.info(f"Published {len(targets)} targets (iteration {self._iteration})")

    def get_position(self) -> Tuple:
        return self._position

    def get_measurements(self) -> List[Tuple]:
        """Block until LUCID signals measurements ready, then read from Tiled."""
        self._measured_event.wait(timeout=10)
        self._measured_event.clear()
        if self._tiled_reader is None:
            return []
        return self._tiled_reader.read_new()

    def signal_measurements_ready(self) -> None:
        """Called by NATS callback when {prefix}.adaptive.measured arrives."""
        self._measured_event.set()
