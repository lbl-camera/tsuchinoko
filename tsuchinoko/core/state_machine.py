"""State machine for Tsuchinoko core experiment lifecycle.

Defines valid states and transitions for the experiment server.
Uses the `transitions` library for formal state machine semantics.
"""

from typing import Callable, Optional
from loguru import logger
from transitions import Machine
from transitions.core import MachineError

from tsuchinoko.core import CoreState


class CoreStateMachine:
    """State machine managing experiment lifecycle transitions.

    States:
        - Inactive: No experiment running, ready to start
        - Starting: Experiment thread is being initialized
        - Running: Experiment is actively collecting data
        - Pausing: Transitioning from running to paused
        - Paused: Experiment paused, can resume or stop
        - Resuming: Transitioning from paused to running
        - Stopping: Transitioning to inactive, clearing data
        - Exiting: Terminal state, application shutting down

    Note: Connecting and Restarting states exist in CoreState enum
    but are used only on the client side.
    """

    # States that the server-side state machine uses
    states = [
        CoreState.Inactive,
        CoreState.Starting,
        CoreState.Running,
        CoreState.Pausing,
        CoreState.Paused,
        CoreState.Resuming,
        CoreState.Stopping,
        CoreState.Exiting,
    ]

    def __init__(self, on_state_change: Optional[Callable[[CoreState], None]] = None):
        """Initialize the state machine.

        Args:
            on_state_change: Optional callback invoked after each state change.
        """
        self._on_state_change = on_state_change

        self.machine = Machine(
            model=self,
            states=self.states,
            initial=CoreState.Inactive,
            auto_transitions=False,
            send_event=True,
        )

        # Define valid transitions
        # Format: trigger, source, dest, [conditions], [unless], [before], [after]

        # Starting an experiment
        self.machine.add_transition(
            trigger='start',
            source=CoreState.Inactive,
            dest=CoreState.Starting,
            after='_log_transition',
        )
        self.machine.add_transition(
            trigger='started',
            source=CoreState.Starting,
            dest=CoreState.Running,
            after='_log_transition',
        )

        # Pausing
        self.machine.add_transition(
            trigger='pause',
            source=CoreState.Running,
            dest=CoreState.Pausing,
            after='_log_transition',
        )
        self.machine.add_transition(
            trigger='paused',
            source=CoreState.Pausing,
            dest=CoreState.Paused,
            after='_log_transition',
        )

        # Resuming from pause
        self.machine.add_transition(
            trigger='resume',
            source=CoreState.Paused,
            dest=CoreState.Resuming,
            after='_log_transition',
        )
        self.machine.add_transition(
            trigger='resumed',
            source=CoreState.Resuming,
            dest=CoreState.Running,
            after='_log_transition',
        )

        # Stopping - can stop from multiple states
        self.machine.add_transition(
            trigger='stop',
            source=[CoreState.Running, CoreState.Starting, CoreState.Pausing,
                    CoreState.Paused, CoreState.Resuming],
            dest=CoreState.Stopping,
            after='_log_transition',
        )
        self.machine.add_transition(
            trigger='stopped',
            source=CoreState.Stopping,
            dest=CoreState.Inactive,
            after='_log_transition',
        )

        # Exiting - can exit from any state except Exiting
        self.machine.add_transition(
            trigger='exit',
            source=[CoreState.Inactive, CoreState.Starting, CoreState.Running,
                    CoreState.Pausing, CoreState.Paused, CoreState.Resuming,
                    CoreState.Stopping],
            dest=CoreState.Exiting,
            after='_log_transition',
        )

    def _log_transition(self, event):
        """Log state transitions."""
        logger.info(f'State transition: {event.transition.source} → {event.transition.dest}')
        if self._on_state_change:
            # event.transition.dest is the CoreState enum value
            self._on_state_change(self.state)

    @property
    def current_state(self) -> CoreState:
        """Get the current state."""
        return self.state

    def can_start(self) -> bool:
        """Check if start transition is valid from current state."""
        return self.state == CoreState.Inactive

    def can_pause(self) -> bool:
        """Check if pause transition is valid from current state."""
        return self.state == CoreState.Running

    def can_resume(self) -> bool:
        """Check if resume transition is valid from current state."""
        return self.state == CoreState.Paused

    def can_stop(self) -> bool:
        """Check if stop transition is valid from current state."""
        return self.state in [CoreState.Running, CoreState.Starting,
                              CoreState.Pausing, CoreState.Paused, CoreState.Resuming]

    def can_exit(self) -> bool:
        """Check if exit transition is valid from current state."""
        return self.state != CoreState.Exiting

    def try_transition(self, trigger: str) -> bool:
        """Attempt a transition, returning False if invalid instead of raising.

        Args:
            trigger: Name of the transition trigger (e.g., 'start', 'pause')

        Returns:
            True if transition succeeded, False if invalid from current state.
        """
        try:
            getattr(self, trigger)()
            return True
        except MachineError:
            logger.warning(f'Invalid transition "{trigger}" from state {self.state}')
            return False
