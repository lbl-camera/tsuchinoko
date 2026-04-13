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
    engine.parameters = MagicMock()
    engine.parameters.__getitem__ = MagicMock(side_effect=lambda key: {
        ("bounds", "axis_0_min"): 0.0, ("bounds", "axis_0_max"): 100.0,
        ("bounds", "axis_1_min"): 0.0, ("bounds", "axis_1_max"): 50.0,
    }.get(key, 0.0))
    return engine


def test_write_config(tiled_client, run_with_adaptive):
    """Creates adaptive/config/ with evaluation_grid_x and evaluation_grid_y of length 50."""
    publisher = TiledPublisher(tiled_client, run_with_adaptive, dimensionality=2)
    engine = _make_mock_engine(dimensionality=2)
    publisher.write_config(engine)

    run = tiled_client[run_with_adaptive]
    config = run["adaptive"]["config"]
    assert "evaluation_grid_x" in config
    assert "evaluation_grid_y" in config
    assert len(config["evaluation_grid_x"].read()) == 50
    assert len(config["evaluation_grid_y"].read()) == 50


def test_write_iteration(tiled_client, run_with_adaptive):
    """Creates adaptive/iter_001/ with posterior_mean, posterior_variance, hyperparameters, targets."""
    publisher = TiledPublisher(tiled_client, run_with_adaptive, dimensionality=2)
    engine = _make_mock_engine(dimensionality=2)
    publisher.write_config(engine)

    targets = np.array([0.5, 0.8, 0.3])
    publisher.write_iteration(1, engine, targets)

    run = tiled_client[run_with_adaptive]
    iter_container = run["adaptive"]["iter_001"]
    assert "hyperparameters" in iter_container
    assert "targets" in iter_container
    assert "posterior_mean" in iter_container
    assert "posterior_variance" in iter_container


def test_write_multiple_iterations(tiled_client, run_with_adaptive):
    """Writes 3 iterations, verifies iter_001/002/003 exist."""
    publisher = TiledPublisher(tiled_client, run_with_adaptive, dimensionality=2)
    engine = _make_mock_engine(dimensionality=2)
    publisher.write_config(engine)

    targets = np.array([0.5])
    for i in range(1, 4):
        publisher.write_iteration(i, engine, targets)

    run = tiled_client[run_with_adaptive]
    adaptive = run["adaptive"]
    assert "iter_001" in adaptive
    assert "iter_002" in adaptive
    assert "iter_003" in adaptive


def test_high_dimensionality_skips_posterior(tiled_client):
    """D=4: hyperparameters present but posterior_mean absent."""
    tiled_client.create_container(key="run_004")
    publisher = TiledPublisher(tiled_client, "run_004", dimensionality=4)

    # For high-D engine we only need hyperparameters mock — no bounds needed
    engine = MagicMock()
    engine.optimizer = MagicMock()
    engine.optimizer.gp = True
    engine.optimizer.get_hyperparameters.return_value = np.array([100.0, 10.0, 10.0, 10.0, 10.0])

    publisher.write_config(engine)
    targets = np.array([0.7])
    publisher.write_iteration(1, engine, targets)

    run = tiled_client["run_004"]
    iter_container = run["adaptive"]["iter_001"]
    assert "hyperparameters" in iter_container
    assert "posterior_mean" not in iter_container


def test_posterior_shape(tiled_client, run_with_adaptive):
    """Verifies posterior_mean is (50, 50)."""
    publisher = TiledPublisher(tiled_client, run_with_adaptive, dimensionality=2)
    engine = _make_mock_engine(dimensionality=2)
    publisher.write_config(engine)

    targets = np.array([0.5])
    publisher.write_iteration(1, engine, targets)

    run = tiled_client[run_with_adaptive]
    mean_arr = run["adaptive"]["iter_001"]["posterior_mean"].read()
    assert mean_arr.shape == (50, 50)
