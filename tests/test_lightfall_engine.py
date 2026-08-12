"""Tests for LightfallEngine."""

import tempfile
import threading
from unittest.mock import MagicMock

import numpy as np
import pytest
from tiled.catalog import in_memory
from tiled.client import Context, from_context
from tiled.server.app import build_app

from tiled.structures.core import Spec

from tsuchinoko.execution.lightfall import LightfallEngine
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
    run = tiled_client.create_container(key="run_001")
    # "composite" spec => CompositeClient with .read(), matching what bluesky's
    # TiledWriter creates for a real stream. A bare container has no .read().
    primary = run.create_container(
        key="primary",
        specs=[Spec("BlueskyEventStream", version="3.0"), Spec("composite")],
    )
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
def lightfall_engine(tiled_client, populated_run, mock_nats_client):
    reader = TiledReader(tiled_client, populated_run,
                         motor_names=["x_motor", "y_motor"], detector_name="detector")
    engine = LightfallEngine(nats_client=mock_nats_client, lightfall_prefix="test.lightfall",
                         tiled_reader=reader)
    return engine


def test_update_targets(lightfall_engine, mock_nats_client):
    """Publishes to 'tsuchinoko.targets' with run_uid, targets list, iteration."""
    targets = [(1.0, 2.0), (3.0, 4.0)]
    lightfall_engine.update_targets(targets)

    mock_nats_client.publish_threadsafe.assert_called_once()
    subject, payload = mock_nats_client.publish_threadsafe.call_args[0]

    assert subject == "tsuchinoko.targets"
    assert payload["targets"] == [[1.0, 2.0], [3.0, 4.0]]
    assert payload["iteration"] == 1


def test_get_position_default(mock_nats_client, tiled_client, populated_run):
    """Returns (0, 0) before any targets are published."""
    reader = TiledReader(tiled_client, populated_run,
                         motor_names=["x_motor", "y_motor"], detector_name="detector")
    engine = LightfallEngine(nats_client=mock_nats_client, lightfall_prefix="test.lightfall",
                         tiled_reader=reader)
    assert engine.get_position() == (0, 0)


def test_get_position_after_targets(lightfall_engine):
    """Returns the last target after update_targets is called."""
    targets = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
    lightfall_engine.update_targets(targets)
    assert lightfall_engine.get_position() == (5.0, 6.0)


def test_get_measurements_after_signal(lightfall_engine):
    """signal_measurements_ready() then get_measurements() returns 3 rows."""
    lightfall_engine.signal_measurements_ready()
    measurements = lightfall_engine.get_measurements()

    assert len(measurements) == 3
    positions = [m[0] for m in measurements]
    values = [m[1] for m in measurements]
    assert positions[0] == (10.0, 15.0)
    assert positions[1] == (20.0, 25.0)
    assert positions[2] == (30.0, 35.0)
    assert abs(values[0] - 0.5) < 1e-6
    assert abs(values[1] - 0.8) < 1e-6
    assert abs(values[2] - 0.3) < 1e-6


def test_get_measurements_blocks(lightfall_engine):
    """Verify get_measurements blocks until signalled."""
    results = []

    def reader():
        results.append(lightfall_engine.get_measurements())

    t = threading.Thread(target=reader, daemon=True)
    t.start()

    # Should still be blocking after 0.5s
    t.join(timeout=0.5)
    assert t.is_alive(), "get_measurements() returned too early — should be blocking"

    # Signal and verify it finishes
    lightfall_engine.signal_measurements_ready()
    t.join(timeout=5.0)
    assert not t.is_alive(), "get_measurements() did not return after signal"
    assert len(results) == 1
    assert len(results[0]) == 3
