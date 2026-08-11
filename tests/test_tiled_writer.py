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
    # Both storages go through in_memory so tiled parses each URI into the right
    # concrete Storage subclass and registers it. Constructing SQLStorage by hand
    # yielded the abstract base, whose create_adbc_connection() is a stub, so every
    # appendable-table write died with "Subclasses must implement this method."
    sql_uri = f"sqlite:///{Path(tmpdir) / 'internal.db'}"
    catalog = in_memory(writable_storage=[tmpdir, sql_uri])

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

    # One (x, y) target this iteration
    publisher.write_iteration(1, engine, np.array([[0.5, 0.8]]))

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
        publisher.write_iteration(i, engine, np.array([[0.5, 0.1]]))

    adaptive = tiled_client[run_uid]["adaptive"]
    # Tiled 0.2 serves the internal table off the base container; adaptive["internal"]
    # now raises KeyError pointing here.
    internal = adaptive.base["internal"]
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
    publisher.write_iteration(1, engine, np.array([[0.1, 0.2, 0.3, 0.4]]))

    adaptive = tiled_client["run_hd"]["adaptive"]
    assert "posterior_mean" not in adaptive


@pytest.mark.integration
def test_acquisition_function(tiled_client, run_uid):
    """write_iteration() should write acquisition_function for D <= 3."""
    publisher = TiledPublisher(tiled_client, run_uid, dimensionality=2)
    engine = _make_mock_engine()
    publisher.write_config(engine)
    publisher.write_iteration(1, engine, np.array([[0.5, 0.5]]))

    adaptive = tiled_client[run_uid]["adaptive"]
    assert "acquisition_function" in adaptive


def test_targets_data_key_declares_target_shape(tiled_client, run_uid):
    """data_keys['targets'] should record the (N_max, D) logical shape."""
    publisher = TiledPublisher(
        tiled_client, run_uid, dimensionality=2, max_targets_per_iter=3,
    )
    publisher.write_config(_make_mock_engine())

    dk = dict(tiled_client[run_uid]["adaptive"].metadata)["data_keys"]
    assert dk["targets"]["shape"] == [3 * 2]
    assert dk["targets"]["target_shape"] == [3, 2]


@pytest.mark.integration
def test_targets_stored_as_padded_flat_array(tiled_client, run_uid):
    """With N_max=3, D=2 and one target, storage is length 6 with 4 NaNs."""
    publisher = TiledPublisher(
        tiled_client, run_uid, dimensionality=2, max_targets_per_iter=3,
    )
    engine = _make_mock_engine()
    publisher.write_config(engine)
    publisher.write_iteration(1, engine, np.array([[0.5, 0.8]]))
    publisher.flush()

    adaptive = tiled_client[run_uid]["adaptive"]
    flat = np.asarray(adaptive["targets"].read())
    assert flat.shape == (1, 6)  # (n_events, N_max * D)
    # First (x, y) is the target; remaining 4 floats are NaN
    np.testing.assert_allclose(flat[0, :2], [0.5, 0.8])
    assert np.all(np.isnan(flat[0, 2:]))


@pytest.mark.integration
def test_targets_supports_variable_N_per_iteration(tiled_client, run_uid):
    """Across iterations, valid-row count can differ; padding fills the rest."""
    publisher = TiledPublisher(
        tiled_client, run_uid, dimensionality=2, max_targets_per_iter=3,
    )
    engine = _make_mock_engine()
    publisher.write_config(engine)

    publisher.write_iteration(1, engine, np.array([[0.1, 0.2]]))                  # N=1
    publisher.write_iteration(2, engine, np.array([[0.3, 0.4], [0.5, 0.6]]))      # N=2
    publisher.write_iteration(3, engine, np.array([[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]]))  # N=3
    publisher.flush()

    adaptive = tiled_client[run_uid]["adaptive"]
    flat = np.asarray(adaptive["targets"].read())
    assert flat.shape == (3, 6)
    # Iteration 1: 1 valid row, 2 NaN rows
    assert np.isnan(flat[0]).sum() == 4
    # Iteration 2: 2 valid rows, 1 NaN row
    assert np.isnan(flat[1]).sum() == 2
    # Iteration 3: full
    assert not np.isnan(flat[2]).any()


@pytest.mark.integration
def test_targets_truncates_and_warns_when_N_exceeds_max(
    tiled_client, run_uid, loguru_messages,
):
    """Writing N > N_max truncates and logs a warning."""
    publisher = TiledPublisher(
        tiled_client, run_uid, dimensionality=2, max_targets_per_iter=2,
    )
    engine = _make_mock_engine()
    publisher.write_config(engine)

    publisher.write_iteration(
        1, engine, np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
    )
    publisher.flush()

    assert any("truncating" in m.lower() for m in loguru_messages)
    adaptive = tiled_client[run_uid]["adaptive"]
    flat = np.asarray(adaptive["targets"].read())
    assert flat.shape == (1, 4)
    np.testing.assert_allclose(flat[0], [0.1, 0.2, 0.3, 0.4])


def test_empty_gp_outputs_are_skipped_not_written(tiled_client, run_uid):
    """An engine whose GP outputs come back empty (e.g. RandomInProcess or a
    not-yet-trained GP) must not crash write_iteration.

    Zero-length values written to Tiled create a (1, 0) zarr array whose
    chunk length is 0, which raises ZeroDivisionError server-side; the
    half-written node then 409s every later iteration. Empty values must be
    NaN-padded to their declared data_keys shape instead (an event may not
    simply omit a declared key: _RunWriter crashes on absent keys).
    """
    publisher = TiledPublisher(tiled_client, run_uid, dimensionality=2)
    engine = _make_mock_engine()
    engine.optimizer.get_hyperparameters.return_value = np.array([])
    engine.optimizer.posterior_mean.return_value = {"f(x)": np.array([])}
    engine.optimizer.posterior_covariance.return_value = {"v(x)": np.array([])}
    engine.optimizer.evaluate_acquisition_function.return_value = np.array([])
    publisher.write_config(engine)

    publisher.write_iteration(1, engine, np.array([[1.0, 2.0]]))
    publisher.write_iteration(2, engine, np.array([[3.0, 4.0]]))
    publisher.flush()

    adaptive = tiled_client[run_uid]["adaptive"]
    assert "targets" in adaptive.keys()
    assert np.asarray(adaptive["targets"].read()).shape[0] == 2
    # Empty GP outputs are NaN-padded to their declared shape, keeping the
    # stream schema stable without ever writing a zero-length array.
    hp = np.asarray(adaptive["hyperparameters"].read())
    assert hp.shape == (2, 20)
    assert np.isnan(hp).all()
    pm = np.asarray(adaptive["posterior_mean"].read())
    assert pm.shape == (2, 50 * 50)
    assert np.isnan(pm).all()
