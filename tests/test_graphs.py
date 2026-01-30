# tests/test_graphs.py
"""Tests for graphs module."""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from tsuchinoko.graphs import Graph, Location, ComputeMode, RenderMode


class TestGraphBase:
    """Tests for Graph base class."""

    def test_graph_has_id(self):
        """Test Graph has a unique ID."""
        g1 = Graph()
        g2 = Graph()

        assert g1.id is not None
        assert g2.id is not None
        assert g1.id != g2.id

    def test_graph_id_is_uuid_string(self):
        """Test Graph ID is a valid UUID string."""
        g = Graph()
        # UUID strings have 36 characters with 4 hyphens
        assert len(g.id) == 36
        assert g.id.count('-') == 4

    def test_graph_default_compute_with(self):
        """Test Graph default compute_with is Client."""
        g = Graph()
        assert g.compute_with == Location.Client

    def test_graph_default_compute_mode(self):
        """Test Graph default compute_mode is Blocking."""
        g = Graph()
        assert g.compute_mode == ComputeMode.Blocking

    def test_graph_default_render_mode(self):
        """Test Graph default render_mode is Blocking."""
        g = Graph()
        assert g.render_mode == RenderMode.Blocking

    def test_graph_inequality_different_ids(self):
        """Test Graph instances with different IDs are not equal (eq=False means identity check)."""
        g1 = Graph()
        g2 = Graph()
        # With eq=False, comparison falls back to identity
        assert g1 != g2
        assert g1 == g1

    def test_graph_widget_args_default(self):
        """Test Graph widget_args defaults to empty tuple."""
        g = Graph()
        assert g.widget_args == tuple()

    def test_graph_widget_kwargs_default(self):
        """Test Graph widget_kwargs defaults to empty dict."""
        g = Graph()
        assert g.widget_kwargs == {}

    def test_graph_compute_method_exists(self):
        """Test Graph has compute method."""
        g = Graph()
        assert hasattr(g, 'compute')
        assert callable(g.compute)

    def test_graph_update_method_exists(self):
        """Test Graph has update method."""
        g = Graph()
        assert hasattr(g, 'update')
        assert callable(g.update)

    def test_graph_make_widget_method_exists(self):
        """Test Graph has make_widget method."""
        g = Graph()
        assert hasattr(g, 'make_widget')
        assert callable(g.make_widget)


class TestGraphEnums:
    """Tests for Graph-related enums."""

    def test_location_enum_values(self):
        """Test Location enum has expected values."""
        assert hasattr(Location, 'Client')
        assert hasattr(Location, 'Core')
        assert hasattr(Location, 'ExecutionEngine')
        assert hasattr(Location, 'AdaptiveEngine')

    def test_compute_mode_enum_values(self):
        """Test ComputeMode enum has expected values."""
        assert hasattr(ComputeMode, 'Blocking')
        assert hasattr(ComputeMode, 'Threaded')

    def test_render_mode_enum_values(self):
        """Test RenderMode enum has expected values."""
        assert hasattr(RenderMode, 'Blocking')
        assert hasattr(RenderMode, 'Background')
