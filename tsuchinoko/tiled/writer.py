"""Write GP outputs to a Tiled run's adaptive stream."""

from __future__ import annotations
from typing import Any
import numpy as np
from loguru import logger


class TiledPublisher:
    """Writes GP outputs to Tiled per-iteration sub-containers."""

    MAX_GRID_DIMENSIONALITY = 3

    def __init__(self, tiled_client: Any, run_uid: str, dimensionality: int,
                 grid_resolution: int = 50) -> None:
        self._client = tiled_client
        self._run_uid = run_uid
        self._dimensionality = dimensionality
        self._grid_resolution = grid_resolution
        self._adaptive = None
        self._grid_points = None

    def write_config(self, engine) -> None:
        """Write evaluation grid (once, at experiment start)."""
        run = self._client[self._run_uid]
        self._adaptive = run.create_container(key="adaptive")
        config = self._adaptive.create_container(key="config")

        if self._dimensionality <= self.MAX_GRID_DIMENSIONALITY:
            bounds = []
            for i in range(self._dimensionality):
                lo = engine.parameters[("bounds", f"axis_{i}_min")]
                hi = engine.parameters[("bounds", f"axis_{i}_max")]
                bounds.append((lo, hi))

            grids = []
            for i, (lo, hi) in enumerate(bounds):
                grid = np.linspace(lo, hi, self._grid_resolution)
                config.write_array(grid, key=f"evaluation_grid_{'xyz'[i]}")
                grids.append(grid)

            mesh = np.meshgrid(*grids, indexing="ij")
            self._grid_points = np.column_stack([m.ravel() for m in mesh])

        logger.info(f"TiledPublisher config written for run {self._run_uid}")

    def write_iteration(self, iteration: int, engine, targets: np.ndarray) -> None:
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
            logger.warning(f"Could not write hyperparameters: {e}")

        if len(targets) > 0:
            iter_container.write_array(np.asarray(targets), key="targets")

        # Write posterior arrays only for low-dimensional
        if self._dimensionality <= self.MAX_GRID_DIMENSIONALITY and self._grid_points is not None:
            try:
                if hasattr(engine, 'optimizer') and engine.optimizer.gp is not None:
                    pm = engine.optimizer.posterior_mean(self._grid_points)
                    mean_vals = np.asarray(pm["f(x)"]).reshape(
                        *[self._grid_resolution] * self._dimensionality
                    )
                    iter_container.write_array(mean_vals, key="posterior_mean")

                    pv = engine.optimizer.posterior_covariance(
                        self._grid_points, variance_only=True
                    )
                    var_vals = np.asarray(pv["v(x)"]).reshape(
                        *[self._grid_resolution] * self._dimensionality
                    )
                    iter_container.write_array(var_vals, key="posterior_variance")
            except Exception as e:
                logger.warning(f"Could not write posterior: {e}")

        logger.debug(f"Wrote iteration {iteration} to Tiled")
