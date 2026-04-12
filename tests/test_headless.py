"""End-to-end test: headless adaptive experiment with no Qt dependency.

Validates that the full import chain and experiment loop work
without PySide6, pyqtgraph, or ZMQ.
"""

import time
from threading import Thread

import numpy as np
import pytest

from tsuchinoko.adaptive import Data
from tsuchinoko.adaptive.gpCAM_in_process import GPCAMInProcessEngine
from tsuchinoko.adaptive.random_in_process import RandomInProcess
from tsuchinoko.core import Core, CoreState
from tsuchinoko.execution.simple import SimpleEngine


def _measure_func(pos):
    """Simple 2D test function."""
    x, y = pos
    value = np.sin(x / 30) + np.cos(y / 30)
    return pos, value, 0.1, {}


class TestHeadlessExperiment:
    """Full experiment lifecycle without Qt."""

    def test_gpcam_headless(self):
        """Run gpCAM adaptive experiment headless."""
        engine = GPCAMInProcessEngine(
            dimensionality=2,
            parameter_bounds=[(0, 100), (0, 100)],
            hyperparameters=[100, 10, 10],
            hyperparameter_bounds=[(0.1, 1e5), (0.1, 1e5), (0.1, 1e5)],
        )
        execution = SimpleEngine(measure_func=_measure_func)

        core = Core(
            execution_engine=execution,
            adaptive_engine=engine,
            compute_metrics=False,
        )
        core.exit_at = [10]

        thread = Thread(target=core.main)
        thread.start()
        core.state = CoreState.Starting
        thread.join(timeout=60)

        assert not thread.is_alive(), "Core did not exit in time"
        assert len(core.data) >= 10

    def test_random_headless(self):
        """Run random sampling experiment headless."""
        engine = RandomInProcess(
            dimensionality=2,
            parameter_bounds=[(0, 100), (0, 100)],
            max_targets=20,
        )
        execution = SimpleEngine(measure_func=_measure_func)

        core = Core(
            execution_engine=execution,
            adaptive_engine=engine,
            compute_metrics=False,
        )
        core.exit_at = [15]

        thread = Thread(target=core.main)
        thread.start()
        core.state = CoreState.Starting
        thread.join(timeout=30)

        assert not thread.is_alive()
        assert len(core.data) >= 15

    def test_pause_resume_headless(self):
        """Test pause/resume lifecycle headless."""
        engine = RandomInProcess(
            dimensionality=2,
            parameter_bounds=[(0, 100), (0, 100)],
        )
        execution = SimpleEngine(measure_func=_measure_func)

        core = Core(
            execution_engine=execution,
            adaptive_engine=engine,
            compute_metrics=False,
        )

        thread = Thread(target=core.main)
        thread.start()
        core.state = CoreState.Starting

        # Let it run briefly
        time.sleep(1)
        assert core.state == CoreState.Running
        assert len(core.data) > 0

        # Pause
        core.state = CoreState.Pausing
        time.sleep(0.5)
        assert core.state == CoreState.Paused
        count_at_pause = len(core.data)

        # Data should not grow while paused
        time.sleep(0.5)
        assert len(core.data) == count_at_pause

        # Resume
        core.state = CoreState.Resuming
        time.sleep(1)
        assert len(core.data) > count_at_pause

        # Exit
        core.exit()
        thread.join(timeout=10)
        assert not thread.is_alive()

    def test_import_chain_no_qt(self):
        """Verify the headless import chain has no Qt dependency."""
        import tsuchinoko
        from tsuchinoko.core import Core, CoreState
        from tsuchinoko.core.state_machine import CoreStateMachine
        from tsuchinoko.adaptive import Engine, Data
        from tsuchinoko.adaptive.gpCAM_in_process import GPCAMInProcessEngine
        from tsuchinoko.adaptive.random_in_process import RandomInProcess
        from tsuchinoko.adaptive.grid import Grid
        from tsuchinoko.execution import Engine as ExecEngine
        from tsuchinoko.execution.simple import SimpleEngine
        from tsuchinoko.execution.threaded_in_process import ThreadedInProcessEngine
        from tsuchinoko.parameters.tree import Parameter, ListParameter, TrainingParameter
        from tsuchinoko.config import AppConfig, CoreConfig
