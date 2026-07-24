"""Write GP outputs to a Tiled run's adaptive stream via TiledWriter.

Builds proper Bluesky documents (descriptor, event) and feeds them
through bluesky-tiled-plugins' TiledWriter, so the adaptive data is
stored in the same format as standard Bluesky runs.
"""

from __future__ import annotations

import time as _time
import uuid
from typing import Any

import numpy as np
from bluesky_tiled_plugins.writing.tiled_writer import _RunWriter
from loguru import logger


class TiledPublisher:
    """Writes GP outputs to Tiled via standard Bluesky documents.

    Emits a descriptor for the "adaptive" stream, then one event per
    iteration containing hyperparameters, targets, and (for D <= 3)
    flattened posterior grids.  The ``_RunWriter`` handles storage
    layout (internal table for scalars, zarr for large arrays).

    Grid arrays are stored flattened; the original grid shape is
    recorded in each data_key's ``grid_shape`` field for consumers.
    """

    MAX_GRID_DIMENSIONALITY = 3

    def __init__(
        self,
        tiled_client: Any,
        run_uid: str,
        dimensionality: int,
        grid_resolution: int = 50,
        max_targets_per_iter: int = 1,
    ) -> None:
        self._client = tiled_client
        self._run_uid = run_uid
        self._dimensionality = dimensionality
        self._grid_resolution = grid_resolution
        self._max_targets_per_iter = max_targets_per_iter
        self._writer: _RunWriter | None = None
        self._desc_uid: str | None = None
        self._grid_points: np.ndarray | None = None
        self._seq_num = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write_config(self, engine: Any) -> None:
        """Emit descriptor and optional grid event for the adaptive stream.

        Must be called once before any ``write_iteration`` calls.
        """
        run = self._client[self._run_uid]

        # Point a _RunWriter at the existing run (skip start doc).
        # max_array_size=0 forces every declared array data_key to zarr
        # storage; the targets key has small shape sum (e.g. [1*D]=2)
        # which would otherwise route to the internal SQL table and
        # break readers that expect adaptive[key] to be an array node.
        self._writer = _RunWriter(self._client, batch_size=1, max_array_size=0)
        self._writer.root_node = run

        # Emit descriptor
        self._desc_uid = str(uuid.uuid4())
        now = _time.time()

        # Build evaluation grids (needed for posterior computation)
        configuration = {}
        if self._dimensionality <= self.MAX_GRID_DIMENSIONALITY:
            configuration = self._build_grid_config(engine)

        self._writer.descriptor({
            "uid": self._desc_uid,
            "run_start": self._run_uid,
            "name": "adaptive",
            "time": now,
            "data_keys": self._build_data_keys(),
            "configuration": configuration,
            "object_keys": {},
            "hints": {"fields": []},
        })

        logger.info("TiledPublisher config written for run {}", self._run_uid[:8])

    def write_iteration(
        self, iteration: int, engine: Any, targets: np.ndarray
    ) -> None:
        """Emit an event with GP outputs for one iteration."""
        if self._writer is None:
            logger.warning("write_config() not called yet, skipping")
            return

        now = _time.time()
        data: dict[str, Any] = {}
        timestamps: dict[str, float] = {}

        # Hyperparameters
        try:
            hp = engine.optimizer.get_hyperparameters()
            data["hyperparameters"] = np.asarray(hp).ravel().tolist()
            timestamps["hyperparameters"] = now
        except Exception as e:
            logger.warning("Could not get hyperparameters: {}", e)

        # Targets — store as a fixed-length flat array of N_max * D
        # floats per iteration with unused rows NaN-padded.  The
        # (N_max, D) logical shape is recorded in data_keys
        # ("target_shape"), paralleling how posterior arrays declare
        # both a flat ``shape`` and a logical ``grid_shape``.
        n_max = self._max_targets_per_iter
        d = self._dimensionality
        buf = np.full((n_max, d), np.nan, dtype=float)
        if len(targets) > 0:
            arr = np.asarray(targets, dtype=float).reshape(-1, d)
            if len(arr) > n_max:
                logger.warning(
                    "Iteration {}: {} targets exceeds max_targets_per_iter={}; "
                    "truncating", iteration, len(arr), n_max,
                )
                arr = arr[:n_max]
            buf[: len(arr)] = arr
        data["targets"] = buf.ravel().tolist()
        timestamps["targets"] = now

        # Posterior grids (D <= 3 only)
        if (
            self._dimensionality <= self.MAX_GRID_DIMENSIONALITY
            and self._grid_points is not None
        ):
            self._collect_posterior(engine, data, timestamps, now)

        if not data:
            return

        # _RunWriter crashes if any declared array data_key is absent from an
        # event (empty arr_lst → min() fails), and a ZERO-LENGTH value is just
        # as fatal: it creates a (1, 0) zarr array whose chunk length is 0,
        # raising ZeroDivisionError server-side (the half-written node then
        # 409s every later iteration). So missing/empty keys are NaN-padded to
        # their declared shape, keeping the stream schema stable per event.
        for key, dk in self._build_data_keys().items():
            if len(data.get(key) or ()) == 0:
                data[key] = [float("nan")] * int(np.prod(dk["shape"]))
                timestamps[key] = now

        self._emit_event(data, timestamps, now)
        logger.info("Wrote iteration {} to Tiled (seq_num={})", iteration, self._seq_num)

    def flush(self) -> None:
        """Flush any remaining cached data in the writer."""
        if self._writer is None:
            return
        for desc_name, cache in self._writer._internal_data_cache.items():
            if cache:
                self._writer._write_internal_data(
                    cache, self._writer._desc_nodes[desc_name]
                )
                cache.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_data_keys(self) -> dict[str, dict]:
        """Build data_keys dict for the adaptive descriptor."""
        # shape sum must exceed MAX_ARRAY_SIZE (16) to route to zarr
        # rather than the SQL internal table (which can't store lists).
        data_keys: dict[str, dict] = {
            "hyperparameters": {
                "dtype": "array",
                "shape": [20],
                "source": "tsuchinoko",
                "dtype_numpy": "<f8",
            },
            "targets": {
                "dtype": "array",
                "shape": [self._max_targets_per_iter * self._dimensionality],
                "source": "tsuchinoko",
                "dtype_numpy": "<f8",
                "target_shape": [
                    self._max_targets_per_iter,
                    self._dimensionality,
                ],
            },
        }

        if self._dimensionality <= self.MAX_GRID_DIMENSIONALITY:
            res = self._grid_resolution
            flat_size = res ** self._dimensionality
            grid_shape = [res] * self._dimensionality

            for name in (
                "posterior_mean",
                "posterior_variance",
                "acquisition_function",
            ):
                data_keys[name] = {
                    "dtype": "array",
                    "shape": [flat_size],
                    "source": "tsuchinoko",
                    "dtype_numpy": "<f8",
                    "grid_shape": grid_shape,
                }

        return data_keys

    def _emit_event(
        self,
        data: dict[str, Any],
        timestamps: dict[str, float],
        now: float,
    ) -> None:
        """Emit a single Bluesky event document."""
        self._seq_num += 1
        self._writer.event({
            "uid": str(uuid.uuid4()),
            "descriptor": self._desc_uid,
            "seq_num": self._seq_num,
            "time": now,
            "data": data,
            "timestamps": timestamps,
        })

    def _build_grid_config(self, engine: Any) -> dict[str, Any]:
        """Build evaluation grids and return as descriptor configuration."""
        grids = []
        grid_data = {}
        grid_timestamps = {}
        now = _time.time()

        for i in range(self._dimensionality):
            lo = engine.parameters[("bounds", f"axis_{i}_min")]
            hi = engine.parameters[("bounds", f"axis_{i}_max")]
            grid = np.linspace(lo, hi, self._grid_resolution)
            axis = "xyz"[i]
            grid_data[f"evaluation_grid_{axis}"] = grid.tolist()
            grid_timestamps[f"evaluation_grid_{axis}"] = now
            grids.append(grid)

        mesh = np.meshgrid(*grids, indexing="ij")
        self._grid_points = np.column_stack([m.ravel() for m in mesh])

        return {
            "tsuchinoko": {
                "data": grid_data,
                "timestamps": grid_timestamps,
                "data_keys": {
                    f"evaluation_grid_{'xyz'[i]}": {
                        "dtype": "array",
                        "shape": [self._grid_resolution],
                        "source": "tsuchinoko",
                        "dtype_numpy": "<f8",
                    }
                    for i in range(self._dimensionality)
                },
            }
        }

    @staticmethod
    def _extract_array(result: Any) -> np.ndarray:
        """Extract array from gpCAM result (dict or raw array)."""
        if isinstance(result, dict):
            # Try known keys across gpCAM versions
            for key in ("f(x)", "m(x)", "m(x)_flat", "v(x)", "v(x)_flat",
                        "mean", "variance"):
                if key in result:
                    return np.asarray(result[key])
            # Fall back to first non-input value
            for key, val in result.items():
                if key not in ("x", "x_pred"):
                    return np.asarray(val)
        return np.asarray(result)

    def _collect_posterior(
        self,
        engine: Any,
        data: dict[str, Any],
        timestamps: dict[str, float],
        now: float,
    ) -> None:
        """Collect posterior mean, variance, and acquisition function."""
        try:
            if not (hasattr(engine, "optimizer") and engine.optimizer.gp is not None):
                return

            pm = engine.optimizer.posterior_mean(self._grid_points)
            data["posterior_mean"] = self._extract_array(pm).ravel().tolist()
            timestamps["posterior_mean"] = now

            pv = engine.optimizer.posterior_covariance(
                self._grid_points, variance_only=True
            )
            data["posterior_variance"] = self._extract_array(pv).ravel().tolist()
            timestamps["posterior_variance"] = now

            try:
                acq = engine.optimizer.evaluate_acquisition_function(
                    self._grid_points
                )
                data["acquisition_function"] = self._extract_array(acq).ravel().tolist()
                timestamps["acquisition_function"] = now
            except Exception as e:
                logger.warning("Could not write acquisition function: {}", e)

        except Exception as e:
            logger.warning("Could not write posterior: {}", e)
