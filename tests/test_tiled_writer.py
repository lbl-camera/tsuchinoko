"""Tests for TiledPublisher with a real in-memory Tiled catalog."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from tiled.catalog import in_memory
from tiled.client import Context, from_context
from tiled.server.app import build_app

from tsuchinoko.tiled.writer import TiledPublisher


@pytest.fixture
def tiled_context():
    """Catalog with both file storage (for zarr) and SQL storage (for tables)."""
    tmpdir = tempfile.mkdtemp()
    catalog = in_memory(writable_storage=tmpdir)

    # Register SQL writable storage so _RunWriter.create_appendable_table works
    from tiled.catalog.adapter import SQLStorage
    sql_uri = f"sqlite:///{Path(tmpdir) / 'internal.db'}"
    catalog.context.writable_storage["sql"] = SQLStorage(uri=sql_uri)

    app = build_app(catalog)
    with Context.from_app(app) as ctx:
        yield ctx


@pytest.fixture
def tiled_client(tiled_context):
    return from_context(tiled_context)


@pytest.fixture
def run_uid(tiled_client):
    """Create a bare run container (simulates what TiledWriter.start() creates)."""
    tiled_client.create_container(key="run_001")
    return "run_001"


def _make_mock_engine(dimensionality=2):
    engine = MagicMock()
    engine.dimensionality = dimensionality
    engine.optimizer = MagicMock()
    engine.optimizer.gp = True
    engine.optimizer.get_hyperparameters.return_value = np.array([100.0, 10.0, 10.0])
    engine.optimizer.posterior_mean.return_value = {"f(x)": np.random.rand(50 * 50)}
    engine.optimizer.posterior_covariance.return_value = {"v(x)": np.random.rand(50 * 50)}
    engine.optimizer.evaluate_acquisition_function.return_value = np.random.rand(50 * 50)
    engine.parameters = MagicMock()
    engine.parameters.__getitem__ = MagicMock(side_effect=lambda key: {
        ("bounds", "axis_0_min"): 0.0, ("bounds", "axis_0_max"): 100.0,
        ("bounds", "axis_1_min"): 0.0, ("bounds", "axis_1_max"): 50.0,
    }.get(key, 0.0))
    return engine


def test_stream_has_bluesky_spec(tiled_client, run_uid):
    """The adaptive stream must have BlueskyEventStream spec."""
    publisher = TiledPublisher(tiled_client, run_uid, dimensionality=2)
    publisher.write_config(_make_mock_engine())

    adaptive = tiled_client[run_uid]["adaptive"]
    spec_names = {s.name for s in adaptive.specs}
    assert "BlueskyEventStream" in spec_names


def test_stream_has_data_keys(tiled_client, run_uid):
    """Descriptor metadata should include standard data_keys."""
    publisher = TiledPublisher(tiled_client, run_uid, dimensionality=2)
    publisher.write_config(_make_mock_engine())

    meta = dict(tiled_client[run_uid]["adaptive"].metadata)
    assert "data_keys" in meta
    assert "uid" in meta
    assert "time" in meta
    dk = meta["data_keys"]
    assert "hyperparameters" in dk
    assert "targets" in dk
    assert "posterior_mean" in dk


def test_grid_shape_in_data_keys(tiled_client, run_uid):
    """data_keys should include grid_shape for posterior fields."""
    publisher = TiledPublisher(tiled_client, run_uid, dimensionality=2)
    publisher.write_config(_make_mock_engine())

    dk = dict(tiled_client[run_uid]["adaptive"].metadata)["data_keys"]
    assert dk["posterior_mean"]["grid_shape"] == [50, 50]
    assert dk["posterior_mean"]["shape"] == [2500]


def test_write_config_stores_grids_in_configuration(tiled_client, run_uid):
    """write_config() should store evaluation grids in descriptor configuration."""
    publisher = TiledPublisher(tiled_client, run_uid, dimensionality=2)
    publisher.write_config(_make_mock_engine())

    meta = dict(tiled_client[run_uid]["adaptive"].metadata)
    config = meta["configuration"]["tsuchinoko"]
    assert "evaluation_grid_x" in config["data"]
    assert "evaluation_grid_y" in config["data"]
    assert len(config["data"]["evaluation_grid_x"]) == 50


@pytest.mark.integration
def test_write_iteration(tiled_client, run_uid):
    """write_iteration() should emit an event with GP data."""
    publisher = TiledPublisher(tiled_client, run_uid, dimensionality=2)
    engine = _make_mock_engine()
    publisher.write_config(engine)

    publisher.write_iteration(1, engine, np.array([0.5, 0.8, 0.3]))

    adaptive = tiled_client[run_uid]["adaptive"]
    assert "posterior_mean" in adaptive
    assert "posterior_variance" in adaptive


@pytest.mark.integration
def test_write_multiple_iterations(tiled_client, run_uid):
    """Multiple iterations should append to the internal table."""
    publisher = TiledPublisher(tiled_client, run_uid, dimensionality=2)
    engine = _make_mock_engine()
    publisher.write_config(engine)

    for i in range(1, 4):
        publisher.write_iteration(i, engine, np.array([0.5]))

    adaptive = tiled_client[run_uid]["adaptive"]
    internal = adaptive["internal"]
    df = internal.read()
    # 3 iteration events (grids are in configuration, not events)
    assert len(df) == 3


@pytest.mark.integration
def test_high_dimensionality_skips_posterior(tiled_client):
    """D=4: hyperparameters written but no posterior or grids."""
    tiled_client.create_container(key="run_hd")
    engine = MagicMock()
    engine.optimizer = MagicMock()
    engine.optimizer.gp = True
    engine.optimizer.get_hyperparameters.return_value = np.array([100.0] * 5)

    publisher = TiledPublisher(tiled_client, "run_hd", dimensionality=4)
    publisher.write_config(engine)
    publisher.write_iteration(1, engine, np.array([0.7]))

    adaptive = tiled_client["run_hd"]["adaptive"]
    assert "posterior_mean" not in adaptive


@pytest.mark.integration
def test_acquisition_function(tiled_client, run_uid):
    """write_iteration() should write acquisition_function for D <= 3."""
    publisher = TiledPublisher(tiled_client, run_uid, dimensionality=2)
    engine = _make_mock_engine()
    publisher.write_config(engine)
    publisher.write_iteration(1, engine, np.array([0.5]))

    adaptive = tiled_client[run_uid]["adaptive"]
    assert "acquisition_function" in adaptive
