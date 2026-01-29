"""Tests for ApplicationContext."""

import pytest
from unittest.mock import Mock

from tsuchinoko.widgets.context import (
    ApplicationContext,
    get_configuration,
    get_state_manager,
    get_graph_manager,
)


class TestApplicationContext:
    """Test ApplicationContext functionality."""

    def setup_method(self):
        """Reset context before each test."""
        ApplicationContext.reset()

    def teardown_method(self):
        """Reset context after each test."""
        ApplicationContext.reset()

    def test_current_returns_same_instance(self):
        """Current should return the same instance."""
        ctx1 = ApplicationContext.current()
        ctx2 = ApplicationContext.current()
        assert ctx1 is ctx2

    def test_set_current(self):
        """Can set a custom context."""
        custom_ctx = ApplicationContext()
        ApplicationContext.set_current(custom_ctx)
        assert ApplicationContext.current() is custom_ctx

    def test_reset_clears_context(self):
        """Reset should clear the current context."""
        ctx1 = ApplicationContext.current()
        ApplicationContext.reset()
        ctx2 = ApplicationContext.current()
        assert ctx1 is not ctx2

    def test_configuration_property(self):
        """Can set and get configuration."""
        ctx = ApplicationContext()
        mock_config = Mock()
        ctx.configuration = mock_config
        assert ctx.configuration is mock_config

    def test_state_manager_property(self):
        """Can set and get state manager."""
        ctx = ApplicationContext()
        mock_sm = Mock()
        ctx.state_manager = mock_sm
        assert ctx.state_manager is mock_sm

    def test_graph_manager_property(self):
        """Can set and get graph manager."""
        ctx = ApplicationContext()
        mock_gm = Mock()
        ctx.graph_manager = mock_gm
        assert ctx.graph_manager is mock_gm

    def test_get_configuration_helper(self):
        """get_configuration returns configuration from current context."""
        mock_config = Mock()
        ctx = ApplicationContext()
        ctx.configuration = mock_config
        ApplicationContext.set_current(ctx)
        assert get_configuration() is mock_config

    def test_get_state_manager_helper(self):
        """get_state_manager returns state manager from current context."""
        mock_sm = Mock()
        ctx = ApplicationContext()
        ctx.state_manager = mock_sm
        ApplicationContext.set_current(ctx)
        assert get_state_manager() is mock_sm

    def test_get_graph_manager_helper(self):
        """get_graph_manager returns graph manager from current context."""
        mock_gm = Mock()
        ctx = ApplicationContext()
        ctx.graph_manager = mock_gm
        ApplicationContext.set_current(ctx)
        assert get_graph_manager() is mock_gm

    def test_helper_returns_none_when_not_set(self):
        """Helpers return None when widgets not set."""
        ctx = ApplicationContext()
        ApplicationContext.set_current(ctx)
        assert get_configuration() is None
        assert get_state_manager() is None
        assert get_graph_manager() is None
