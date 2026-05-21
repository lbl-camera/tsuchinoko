"""Read measurement data from a Tiled run's primary stream."""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger


class TiledReader:
    """Reads measurements from Tiled, tracking position for incremental reads."""

    def __init__(self, tiled_client: Any, run_uid: str, motor_names: list[str],
                 detector_name: str, default_variance: float = 1.0) -> None:
        self._client = tiled_client
        self._run_uid = run_uid
        self._motor_names = motor_names
        self._detector_name = detector_name
        self._default_variance = default_variance
        self._rows_read = 0

    def read_new(self) -> list[tuple]:
        """Read rows added since last call. Returns list of (pos, val, var, metrics)."""
        try:
            run = self._client[self._run_uid]
            if hasattr(run, "refresh"):
                run.refresh()
            primary = run["primary"]
        except (KeyError, Exception) as e:
            logger.warning(f"Cannot read primary stream: {e}")
            return []

        # Read the whole stream as one xarray.Dataset via the V3 table facet.
        # This bundles every data_var into a single HTTP round-trip and skips
        # the array endpoint's expected_shape check, which 500s whenever an
        # event lands between the client's cached shape and the actual read.
        ds = primary.read()

        if self._detector_name not in ds.data_vars:
            return []
        missing_motors = [m for m in self._motor_names if m not in ds.data_vars]
        if missing_motors:
            logger.warning(f"Motor keys missing from stream: {missing_motors}")
            return []

        detector_data = ds[self._detector_name].values
        total_rows = len(detector_data)

        if total_rows <= self._rows_read:
            return []

        motor_arrays = [
            ds[name].values[self._rows_read:total_rows] for name in self._motor_names
        ]
        values = detector_data[self._rows_read:total_rows]
        n_new = total_rows - self._rows_read

        measurements = []
        for i in range(n_new):
            position = tuple(float(arr[i]) for arr in motor_arrays)
            value = float(values[i])
            measurements.append((position, value, self._default_variance, {}))

        self._rows_read = total_rows
        logger.info(f"Read {n_new} new measurements (total: {total_rows})")
        return measurements
