"""Tests for Core with TiledPublisher integration."""

import tempfile
import time
from pathlib import Path
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
    # SQL storage as well as file storage: the publisher writes an appendable
    # table, and SQLAdapter refuses a catalog offering only FileStorage.
    sql_uri = f"sqlite:///{Path(tmpdir) / 'internal.db'}"
    catalog = in_memory(writable_storage=[tmpdir, sql_uri])
    app = build_app(catalog)
    with Context.from_app(app) as ctx:
        yield ctx


@pytest.fixture
def tiled_client(tiled_context):
    return from_context(tiled_context)


class TestCoreWithTiledPublisher:
    def test_publishes_after_iterations(self, tiled_client, join_core):
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

        thread = Thread(target=core.main, daemon=True)
        thread.start()
        core.state = CoreState.Starting
        join_core(thread, core, timeout=60)

        assert len(core.data) >= 5

        adaptive = tiled_client["test_run"]["adaptive"]
        children = list(adaptive)
        iter_keys = [k for k in children if k.startswith("iter_")]
        assert len(iter_keys) >= 1

    def test_no_publisher_no_error(self, join_core):
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

        thread = Thread(target=core.main, daemon=True)
        thread.start()
        core.state = CoreState.Starting
        join_core(thread, core, timeout=30)

        assert len(core.data) >= 3
