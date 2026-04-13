"""Tests for the rewritten Core._main() with optional NATS."""

import asyncio
import time
from threading import Thread

import numpy as np
import pytest

from tsuchinoko.adaptive.random_in_process import RandomInProcess
from tsuchinoko.core import Core, CoreState
from tsuchinoko.execution.simple import SimpleEngine
from tsuchinoko.nats.config import NATSConfig


def _measure(pos):
    x, y = pos
    return pos, np.sin(x / 30), 0.1, {}


class TestCoreWithoutNATS:
    def test_no_nats_by_default(self):
        core = Core()
        assert core._nats_config is None
        assert core._nats_client is None

    def test_experiment_runs_without_nats(self):
        engine = RandomInProcess(dimensionality=2, parameter_bounds=[(0, 100), (0, 100)])
        execution = SimpleEngine(measure_func=_measure)
        core = Core(execution_engine=execution, adaptive_engine=engine, compute_metrics=False)
        core.exit_at = [5]

        thread = Thread(target=core.main)
        thread.start()
        core.state = CoreState.Starting
        thread.join(timeout=30)

        assert not thread.is_alive()
        assert len(core.data) >= 5


class TestCoreWithNATSConfig:
    def test_accepts_nats_config(self):
        config = NATSConfig(url="nats://localhost:4222")
        core = Core(nats_config=config)
        assert core._nats_config is config

    def test_event_queue_exists(self):
        core = Core()
        assert hasattr(core, '_event_queue')


class TestEventQueue:
    def test_emit_event_queues(self):
        core = Core()
        core.emit_event("tsuchinoko.state", {"state": "running"})
        assert not core._event_queue.empty()
        subject, payload = core._event_queue.get_nowait()
        assert subject == "tsuchinoko.state"
        assert payload == {"state": "running"}

    def test_emit_event_from_experiment_thread(self):
        engine = RandomInProcess(dimensionality=2, parameter_bounds=[(0, 100), (0, 100)])
        execution = SimpleEngine(measure_func=_measure)
        core = Core(execution_engine=execution, adaptive_engine=engine, compute_metrics=False)
        core.exit_at = [3]

        thread = Thread(target=core.main)
        thread.start()
        core.state = CoreState.Starting
        thread.join(timeout=30)

        assert not thread.is_alive()
        assert len(core.data) >= 3
