"""Tests for TiledPublisher with a real in-memory Tiled catalog."""

import tempfile
from unittest.mock import MagicMock

import numpy as np
import pytest
from tiled.catalog import in_memory
from tiled.client import Context, from_context
from tiled.server.app import build_app

from tsuchinoko.tiled.writer import TiledPublisher


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
def run_with_adaptive(tiled_client):
    tiled_client.create_container(key="run_001")
    return "run_001"


def _make_mock_engine(dimensionality=2):
    engine = MagicMock()
    engine.dimensionality = dimensionality
    engine.optimizer = MagicMock()
    engine.optimizer.gp = True  # GP is initialized
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


def test_stream_has_bluesky_spec(tiled_client, run_with_adaptive):
    """The adaptive stream must have BlueskyEventStream spec."""
    publisher = TiledPublisher(tiled_client, run_with_adaptive, dimensionality=2)
    publisher.write_config(_make_mock_engine())

    adaptive = tiled_client[run_with_adaptive]["adaptive"]
    spec_names = {s.name for s in adaptive.specs}
    assert "BlueskyEventStream" in spec_names


def test_stream_has_data_keys(tiled_client, run_with_adaptive):
    """Descriptor metadata should include standard data_keys."""
    publisher = TiledPublisher(tiled_client, run_with_adaptive, dimensionality=2)
    publisher.write_config(_make_mock_engine())

    meta = dict(tiled_client[run_with_adaptive]["adaptive"].metadata)
    assert "data_keys" in meta
    assert "uid" in meta
    assert "time" in meta
    dk = meta["data_keys"]
    assert "hyperparameters" in dk
    assert "targets" in dk
    assert "posterior_mean" in dk
    assert "evaluation_grid_x" in dk


def test_grid_shape_in_data_keys(tiled_client, run_with_adaptive):
    """data_keys should include grid_shape for posterior fields."""
    publisher = TiledPublisher(tiled_client, run_with_adaptive, dimensionality=2)
    publisher.write_config(_make_mock_engine())

    dk = dict(tiled_client[run_with_adaptive]["adaptive"].metadata)["data_keys"]
    assert dk["posterior_mean"]["grid_shape"] == [50, 50]
    assert dk["posterior_mean"]["shape"] == [50, 50]


def test_write_config(tiled_client, run_with_adaptive):
    """Creates adaptive/config/ with evaluation grids of length 50."""
    publisher = TiledPublisher(tiled_client, run_with_adaptive, dimensionality=2)
    publisher.write_config(_make_mock_engine())

    config = tiled_client[run_with_adaptive]["adaptive"]["config"]
    assert "evaluation_grid_x" in config
    assert "evaluation_grid_y" in config
    assert len(config["evaluation_grid_x"].read()) == 50
    assert len(config["evaluation_grid_y"].read()) == 50


def test_write_config_stamps_metadata(tiled_client, run_with_adaptive):
    """write_config() should stamp adaptive_engine metadata."""
    publisher = TiledPublisher(tiled_client, run_with_adaptive, dimensionality=2)
    publisher.write_config(_make_mock_engine())

    meta = tiled_client[run_with_adaptive]["adaptive"].metadata
    assert meta.get("adaptive_engine") == "tsuchinoko"


def test_write_iteration(tiled_client, run_with_adaptive):
    """Creates iter_001 with posterior_mean, posterior_variance, hyperparameters, targets."""
    publisher = TiledPublisher(tiled_client, run_with_adaptive, dimensionality=2)
    publisher.write_config(_make_mock_engine())

    targets = np.array([0.5, 0.8, 0.3])
    publisher.write_iteration(1, _make_mock_engine(), targets)

    iter_container = tiled_client[run_with_adaptive]["adaptive"]["iter_001"]
    assert "hyperparameters" in iter_container
    assert "targets" in iter_container
    assert "posterior_mean" in iter_container
    assert "posterior_variance" in iter_container


def test_write_multiple_iterations(tiled_client, run_with_adaptive):
    """Writes 3 iterations, verifies iter_001/002/003 exist."""
    publisher = TiledPublisher(tiled_client, run_with_adaptive, dimensionality=2)
    engine = _make_mock_engine()
    publisher.write_config(engine)

    for i in range(1, 4):
        publisher.write_iteration(i, engine, np.array([0.5]))

    adaptive = tiled_client[run_with_adaptive]["adaptive"]
    assert "iter_001" in adaptive
    assert "iter_002" in adaptive
    assert "iter_003" in adaptive


def test_high_dimensionality_skips_posterior(tiled_client):
    """D=4: hyperparameters present but posterior_mean absent."""
    tiled_client.create_container(key="run_004")
    engine = MagicMock()
    engine.optimizer = MagicMock()
    engine.optimizer.gp = True
    engine.optimizer.get_hyperparameters.return_value = np.array([100.0] * 5)

    publisher = TiledPublisher(tiled_client, "run_004", dimensionality=4)
    publisher.write_config(engine)
    publisher.write_iteration(1, engine, np.array([0.7]))

    iter_container = tiled_client["run_004"]["adaptive"]["iter_001"]
    assert "hyperparameters" in iter_container
    assert "posterior_mean" not in iter_container


def test_posterior_shape(tiled_client, run_with_adaptive):
    """Verifies posterior_mean is (50, 50)."""
    publisher = TiledPublisher(tiled_client, run_with_adaptive, dimensionality=2)
    engine = _make_mock_engine()
    publisher.write_config(engine)
    publisher.write_iteration(1, engine, np.array([0.5]))

    mean_arr = tiled_client[run_with_adaptive]["adaptive"]["iter_001"]["posterior_mean"].read()
    assert mean_arr.shape == (50, 50)


def test_write_iteration_includes_acquisition_function(tiled_client, run_with_adaptive):
    """write_iteration() should write acquisition_function for dim <= 3."""
    publisher = TiledPublisher(tiled_client, run_with_adaptive, dimensionality=2)
    engine = _make_mock_engine()
    publisher.write_config(engine)
    publisher.write_iteration(1, engine, np.array([0.5, 0.8]))

    iter_container = tiled_client[run_with_adaptive]["adaptive"]["iter_001"]
    assert "acquisition_function" in iter_container
    assert iter_container["acquisition_function"].read().shape == (50, 50)


def test_high_dimensionality_skips_acquisition_function(tiled_client):
    """D=4: acquisition_function should not be written."""
    tiled_client.create_container(key="run_acq")
    engine = MagicMock()
    engine.optimizer = MagicMock()
    engine.optimizer.gp = True
    engine.optimizer.get_hyperparameters.return_value = np.array([100.0] * 5)

    publisher = TiledPublisher(tiled_client, "run_acq", dimensionality=4)
    publisher.write_config(engine)
    publisher.write_iteration(1, engine, np.array([0.7]))

    iter_container = tiled_client["run_acq"]["adaptive"]["iter_001"]
    assert "acquisition_function" not in iter_container


def test_high_dimensionality_no_grids(tiled_client):
    """D=4: no evaluation grids or grid-related data_keys."""
    tiled_client.create_container(key="run_hd")
    engine = MagicMock()
    engine.optimizer = MagicMock()
    engine.optimizer.gp = True
    engine.optimizer.get_hyperparameters.return_value = np.array([100.0] * 5)

    publisher = TiledPublisher(tiled_client, "run_hd", dimensionality=4)
    publisher.write_config(engine)

    adaptive = tiled_client["run_hd"]["adaptive"]
    assert "config" not in adaptive
    dk = dict(adaptive.metadata)["data_keys"]
    assert "evaluation_grid_x" not in dk
    assert "posterior_mean" not in dk
