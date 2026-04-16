"""Write GP outputs to a Tiled run's adaptive stream.

Creates a properly-specced BlueskyEventStream container with standard
descriptor metadata (data_keys, uid, time, etc.), so that the exporter
and documents() API recognise the stream.  Per-iteration GP outputs
are stored as zarr arrays inside the stream container.
"""

from __future__ import annotations

import time as _time
import uuid
from typing import Any

import numpy as np
from loguru import logger
from tiled.client.container import Container
from tiled.structures.core import Spec


class TiledPublisher:
    """Writes GP outputs to Tiled as a proper BlueskyEventStream.

    Creates an "adaptive" stream in the run with standard Bluesky
    descriptor metadata, then writes per-iteration GP outputs as
    zarr arrays inside iter_NNN sub-containers.

    Grid arrays are stored at their natural shape; the grid_shape
    is also recorded in each data_key's metadata for reference.
    """

    MAX_GRID_DIMENSIONALITY = 3

    _STREAM_SPECS = [
        Spec("BlueskyEventStream", version="3.0"),
    ]

    def __init__(
        self,
        tiled_client: Any,
        run_uid: str,
        dimensionality: int,
        grid_resolution: int = 50,
    ) -> None:
        self._client = tiled_client
        self._run_uid = run_uid
        self._dimensionality = dimensionality
        self._grid_resolution = grid_resolution
        self._adaptive = None
        self._grid_points = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write_config(self, engine: Any) -> None:
        """Create the adaptive stream and write evaluation grids.

        Must be called once before any ``write_iteration`` calls.
        """
        run = self._client[self._run_uid]

        # Build standard descriptor metadata
        data_keys = self._build_data_keys()
        desc_uid = str(uuid.uuid4())
        now = _time.time()
        metadata = {
            "uid": desc_uid,
            "time": now,
            "data_keys": data_keys,
            "configuration": {},
            "hints": {"fields": []},
            "adaptive_engine": "tsuchinoko",
        }

        # Create the stream with proper BlueskyEventStream spec.
        # Use the raw Container to allow creating nested sub-containers
        # (the spec-dispatched client may block this).
        node = run.create_container(
            key="adaptive",
            metadata=metadata,
            specs=self._STREAM_SPECS,
        )
        self._adaptive = Container(
            node.context, item=node.item, structure_clients=node.structure_clients,
        )

        # Write evaluation grids
        if self._dimensionality <= self.MAX_GRID_DIMENSIONALITY:
            self._write_grids(engine)

        logger.info("TiledPublisher config written for run {}", self._run_uid[:8])

    def write_iteration(
        self, iteration: int, engine: Any, targets: np.ndarray
    ) -> None:
        """Write GP outputs for one iteration."""
        if self._adaptive is None:
            logger.warning("write_config() not called yet, skipping")
            return

        iter_key = f"iter_{iteration:03d}"
        iter_container = self._adaptive.create_container(key=iter_key)

        # Always write hyperparameters and targets
        try:
            hp = engine.optimizer.get_hyperparameters()
            iter_container.write_array(np.asarray(hp), key="hyperparameters")
        except Exception as e:
            logger.warning("Could not write hyperparameters: {}", e)

        if len(targets) > 0:
            iter_container.write_array(np.asarray(targets), key="targets")

        # Write posterior arrays only for low-dimensional problems
        if (
            self._dimensionality <= self.MAX_GRID_DIMENSIONALITY
            and self._grid_points is not None
        ):
            self._write_posterior(engine, iter_container)

        logger.info(
            "Wrote iteration {} to Tiled (keys: {})",
            iteration,
            list(iter_container),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_data_keys(self) -> dict[str, dict]:
        """Build data_keys dict for the adaptive descriptor."""
        data_keys: dict[str, dict] = {
            "hyperparameters": {
                "dtype": "array",
                "shape": [],
                "source": "tsuchinoko",
                "dtype_numpy": "<f8",
            },
            "targets": {
                "dtype": "array",
                "shape": [],
                "source": "tsuchinoko",
                "dtype_numpy": "<f8",
            },
        }

        if self._dimensionality <= self.MAX_GRID_DIMENSIONALITY:
            res = self._grid_resolution
            grid_shape = [res] * self._dimensionality

            for i in range(self._dimensionality):
                axis = "xyz"[i]
                data_keys[f"evaluation_grid_{axis}"] = {
                    "dtype": "array",
                    "shape": [res],
                    "source": "tsuchinoko",
                    "dtype_numpy": "<f8",
                }

            for name in (
                "posterior_mean",
                "posterior_variance",
                "acquisition_function",
            ):
                data_keys[name] = {
                    "dtype": "array",
                    "shape": grid_shape,
                    "source": "tsuchinoko",
                    "dtype_numpy": "<f8",
                    "grid_shape": grid_shape,
                }

        return data_keys

    def _write_grids(self, engine: Any) -> None:
        """Build evaluation grids and write them as arrays in a config container."""
        config = self._adaptive.create_container(key="config")
        grids = []

        for i in range(self._dimensionality):
            lo = engine.parameters[("bounds", f"axis_{i}_min")]
            hi = engine.parameters[("bounds", f"axis_{i}_max")]
            grid = np.linspace(lo, hi, self._grid_resolution)
            config.write_array(grid, key=f"evaluation_grid_{'xyz'[i]}")
            grids.append(grid)

        mesh = np.meshgrid(*grids, indexing="ij")
        self._grid_points = np.column_stack([m.ravel() for m in mesh])

    def _write_posterior(self, engine: Any, container: Any) -> None:
        """Write posterior mean, variance, and acquisition function."""
        try:
            if not (hasattr(engine, "optimizer") and engine.optimizer.gp is not None):
                return

            pm = engine.optimizer.posterior_mean(self._grid_points)
            mean_vals = np.asarray(pm["f(x)"]).reshape(
                *[self._grid_resolution] * self._dimensionality
            )
            container.write_array(mean_vals, key="posterior_mean")

            pv = engine.optimizer.posterior_covariance(
                self._grid_points, variance_only=True
            )
            var_vals = np.asarray(pv["v(x)"]).reshape(
                *[self._grid_resolution] * self._dimensionality
            )
            container.write_array(var_vals, key="posterior_variance")

            try:
                acq = engine.optimizer.evaluate_acquisition_function(
                    self._grid_points
                )
                acq_vals = np.asarray(acq).reshape(
                    *[self._grid_resolution] * self._dimensionality
                )
                container.write_array(acq_vals, key="acquisition_function")
            except Exception as e:
                logger.warning("Could not write acquisition function: {}", e)

        except Exception as e:
            logger.warning("Could not write posterior: {}", e)
