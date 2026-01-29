"""Tests for CoreStateMachine."""

import pytest
from transitions.core import MachineError

from tsuchinoko.core import CoreState
from tsuchinoko.core.state_machine import CoreStateMachine


class TestCoreStateMachine:
    """Test state machine transitions."""

    def test_initial_state(self):
        """State machine starts in Inactive state."""
        sm = CoreStateMachine()
        assert sm.current_state == CoreState.Inactive

    def test_start_experiment(self):
        """Test starting an experiment from Inactive."""
        sm = CoreStateMachine()
        assert sm.can_start()
        sm.start()
        assert sm.current_state == CoreState.Starting
        sm.started()
        assert sm.current_state == CoreState.Running

    def test_pause_resume_cycle(self):
        """Test pausing and resuming an experiment."""
        sm = CoreStateMachine()
        sm.start()
        sm.started()
        assert sm.current_state == CoreState.Running

        # Pause
        assert sm.can_pause()
        sm.pause()
        assert sm.current_state == CoreState.Pausing
        sm.paused()
        assert sm.current_state == CoreState.Paused

        # Resume
        assert sm.can_resume()
        sm.resume()
        assert sm.current_state == CoreState.Resuming
        sm.resumed()
        assert sm.current_state == CoreState.Running

    def test_stop_from_running(self):
        """Test stopping from Running state."""
        sm = CoreStateMachine()
        sm.start()
        sm.started()
        assert sm.current_state == CoreState.Running

        assert sm.can_stop()
        sm.stop()
        assert sm.current_state == CoreState.Stopping
        sm.stopped()
        assert sm.current_state == CoreState.Inactive

    def test_stop_from_paused(self):
        """Test stopping from Paused state."""
        sm = CoreStateMachine()
        sm.start()
        sm.started()
        sm.pause()
        sm.paused()
        assert sm.current_state == CoreState.Paused

        sm.stop()
        assert sm.current_state == CoreState.Stopping
        sm.stopped()
        assert sm.current_state == CoreState.Inactive

    def test_exit_from_any_state(self):
        """Test exiting from various states."""
        # Exit from Inactive
        sm = CoreStateMachine()
        assert sm.can_exit()
        sm.exit()
        assert sm.current_state == CoreState.Exiting

        # Exit from Running
        sm = CoreStateMachine()
        sm.start()
        sm.started()
        sm.exit()
        assert sm.current_state == CoreState.Exiting

        # Exit from Paused
        sm = CoreStateMachine()
        sm.start()
        sm.started()
        sm.pause()
        sm.paused()
        sm.exit()
        assert sm.current_state == CoreState.Exiting

    def test_invalid_transition_raises(self):
        """Test that invalid transitions raise MachineError."""
        sm = CoreStateMachine()
        # Can't pause from Inactive
        with pytest.raises(MachineError):
            sm.pause()

    def test_try_transition_returns_false_on_invalid(self):
        """Test try_transition returns False for invalid transitions."""
        sm = CoreStateMachine()
        # Can't pause from Inactive
        assert not sm.try_transition('pause')
        assert sm.current_state == CoreState.Inactive

    def test_try_transition_returns_true_on_valid(self):
        """Test try_transition returns True for valid transitions."""
        sm = CoreStateMachine()
        assert sm.try_transition('start')
        assert sm.current_state == CoreState.Starting

    def test_cannot_start_when_running(self):
        """Test that start() is invalid when already running."""
        sm = CoreStateMachine()
        sm.start()
        sm.started()
        assert not sm.can_start()
        with pytest.raises(MachineError):
            sm.start()

    def test_cannot_resume_when_running(self):
        """Test that resume() is invalid when running."""
        sm = CoreStateMachine()
        sm.start()
        sm.started()
        assert not sm.can_resume()
        with pytest.raises(MachineError):
            sm.resume()

    def test_on_state_change_callback(self):
        """Test that on_state_change callback is invoked."""
        states_received = []

        def callback(state):
            states_received.append(state)

        sm = CoreStateMachine(on_state_change=callback)
        sm.start()
        sm.started()
        sm.pause()
        sm.paused()

        assert states_received == [
            CoreState.Starting,
            CoreState.Running,
            CoreState.Pausing,
            CoreState.Paused,
        ]
