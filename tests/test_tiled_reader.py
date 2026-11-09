"""Tests for TiledReader â€” incremental reads from a Tiled primary stream."""

from __future__ import annotations

import tempfile

import numpy as np
import pytest
from tiled.catalog import in_memory
from tiled.client import Context, from_context
from tiled.server.app import build_app

from tiled.structures.core import Spec

from tsuchinoko.tiled.reader import TiledReader


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _create_stream(run, key):
    """Create a stream node shaped like the ones bluesky's TiledWriter produces.

    A plain container has no .read(); the "composite" spec is what makes the
    client hand back a CompositeClient exposing the table facet TiledReader
    uses. Building these with a bare create_container made every reader test
    fail with "'Container' object has no attribute 'read'" against code that is
    correct for real runs. Specs match bluesky's TiledWriter.descriptor().
    """
    return run.create_container(
        key=key,
        specs=[Spec("BlueskyEventStream", version="3.0"), Spec("composite")],
    )

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
    """Run with 5 measurements: x_motor, y_motor, detector."""
    run = tiled_client.create_container(key="run_001")
    primary = _create_stream(run, "primary")
    primary.write_array(np.array([10.0, 20.0, 30.0, 40.0, 50.0]), key="x_motor")
    primary.write_array(np.array([15.0, 25.0, 35.0, 45.0, 55.0]), key="y_motor")
    primary.write_array(np.array([0.5, 0.8, 0.3, 0.9, 0.1]), key="detector")
    return "run_001"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_read_all(tiled_client, populated_run):
    """Reads all 5 measurements from a fully populated run."""
    reader = TiledReader(
        tiled_client=tiled_client,
        run_uid=populated_run,
        motor_names=["x_motor", "y_motor"],
        detector_name="detector",
    )
    measurements = reader.read_new()
    assert len(measurements) == 5


def test_read_returns_tuples(tiled_client, populated_run):
    """Each measurement is (tuple, float, float, dict)."""
    reader = TiledReader(
        tiled_client=tiled_client,
        run_uid=populated_run,
        motor_names=["x_motor", "y_motor"],
        detector_name="detector",
    )
    measurements = reader.read_new()
    for pos, val, var, metrics in measurements:
        assert isinstance(pos, tuple)
        assert isinstance(val, float)
        assert isinstance(var, float)
        assert isinstance(metrics, dict)

    # Spot-check first measurement
    pos0, val0, _var0, _m0 = measurements[0]
    assert pos0 == (10.0, 15.0)
    assert val0 == pytest.approx(0.5)


def test_incremental_read(tiled_client, populated_run):
    """Second read_new() returns empty when no new data was added."""
    reader = TiledReader(
        tiled_client=tiled_client,
        run_uid=populated_run,
        motor_names=["x_motor"],
        detector_name="detector",
    )
    first = reader.read_new()
    assert len(first) == 5

    second = reader.read_new()
    assert second == []


def test_incremental_read_after_new_data(tiled_client):
    """Reader with _rows_read=5 on a 7-row run returns only the 2 new rows.

    Tiled arrays are immutable after writing, so we simulate this by creating
    a 7-row run and fast-forwarding _rows_read to 5 to represent a reader that
    already consumed the first 5 rows.
    """
    run = tiled_client.create_container(key="run_007")
    primary = _create_stream(run, "primary")
    primary.write_array(
        np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]), key="x_motor"
    )
    primary.write_array(
        np.array([0.5, 0.8, 0.3, 0.9, 0.1, 0.6, 0.7]), key="detector"
    )

    reader = TiledReader(
        tiled_client=tiled_client,
        run_uid="run_007",
        motor_names=["x_motor"],
        detector_name="detector",
    )
    # Simulate having already read 5 rows
    reader._rows_read = 5

    new_measurements = reader.read_new()
    assert len(new_measurements) == 2

    pos0, val0, _var0, _m0 = new_measurements[0]
    assert pos0 == (60.0,)
    assert val0 == pytest.approx(0.6)

    pos1, val1, _var1, _m1 = new_measurements[1]
    assert pos1 == (70.0,)
    assert val1 == pytest.approx(0.7)

    # Rows-read counter should now be 7
    assert reader._rows_read == 7


def test_custom_variance(tiled_client, populated_run):
    """default_variance is applied to every measurement."""
    reader = TiledReader(
        tiled_client=tiled_client,
        run_uid=populated_run,
        motor_names=["x_motor"],
        detector_name="detector",
        default_variance=0.5,
    )
    measurements = reader.read_new()
    for _pos, _val, var, _m in measurements:
        assert var == pytest.approx(0.5)


def test_empty_run(tiled_client):
    """Returns empty list when the detector key is absent from primary."""
    run = tiled_client.create_container(key="run_empty")
    _primary = _create_stream(run, "primary")
    # No arrays written

    reader = TiledReader(
        tiled_client=tiled_client,
        run_uid="run_empty",
        motor_names=["x_motor"],
        detector_name="detector",
    )
    assert reader.read_new() == []
