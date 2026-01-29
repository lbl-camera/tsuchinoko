"""Application context for widget access without singleton metaclass.

This module provides a clean alternative to the Singleton metaclass pattern,
allowing widgets to be accessed globally while maintaining testability.
"""

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from tsuchinoko.widgets.displays import Configuration, StateManager, GraphManager


class ApplicationContext:
    """Holds references to application-wide widgets.

    This replaces the Singleton metaclass pattern with an explicit context
    that can be configured during testing.

    Usage:
        # In MainWindow initialization
        context = ApplicationContext()
        context.configuration = Configuration()
        context.state_manager = StateManager()
        context.graph_manager = GraphManager()
        ApplicationContext.set_current(context)

        # Elsewhere in the app
        config = ApplicationContext.current().configuration

    For testing:
        # Create mock context
        test_context = ApplicationContext()
        test_context.configuration = MockConfiguration()
        ApplicationContext.set_current(test_context)
    """

    _current: Optional['ApplicationContext'] = None

    def __init__(self):
        self._configuration: Optional['Configuration'] = None
        self._state_manager: Optional['StateManager'] = None
        self._graph_manager: Optional['GraphManager'] = None

    @classmethod
    def current(cls) -> 'ApplicationContext':
        """Get the current application context.

        Returns:
            The current ApplicationContext instance.

        Raises:
            RuntimeError: If no context has been set.
        """
        if cls._current is None:
            # Auto-create a default context for backward compatibility
            cls._current = cls()
        return cls._current

    @classmethod
    def set_current(cls, context: Optional['ApplicationContext']):
        """Set the current application context.

        Args:
            context: The context to set as current, or None to clear.
        """
        cls._current = context

    @classmethod
    def reset(cls):
        """Reset the context to None. Useful for testing."""
        cls._current = None

    @property
    def configuration(self) -> Optional['Configuration']:
        """Get the Configuration widget."""
        return self._configuration

    @configuration.setter
    def configuration(self, value: 'Configuration'):
        """Set the Configuration widget."""
        self._configuration = value

    @property
    def state_manager(self) -> Optional['StateManager']:
        """Get the StateManager widget."""
        return self._state_manager

    @state_manager.setter
    def state_manager(self, value: 'StateManager'):
        """Set the StateManager widget."""
        self._state_manager = value

    @property
    def graph_manager(self) -> Optional['GraphManager']:
        """Get the GraphManager widget."""
        return self._graph_manager

    @graph_manager.setter
    def graph_manager(self, value: 'GraphManager'):
        """Set the GraphManager widget."""
        self._graph_manager = value


def get_configuration() -> Optional['Configuration']:
    """Get the current Configuration widget.

    Returns:
        The Configuration widget, or None if not set.
    """
    return ApplicationContext.current().configuration


def get_state_manager() -> Optional['StateManager']:
    """Get the current StateManager widget.

    Returns:
        The StateManager widget, or None if not set.
    """
    return ApplicationContext.current().state_manager


def get_graph_manager() -> Optional['GraphManager']:
    """Get the current GraphManager widget.

    Returns:
        The GraphManager widget, or None if not set.
    """
    return ApplicationContext.current().graph_manager
