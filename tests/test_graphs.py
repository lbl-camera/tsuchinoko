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


from tsuchinoko.graphs.common import Cloud, Score, Variance
from tsuchinoko.adaptive import Data
from tsuchinoko.widgets.context import ApplicationContext
from tsuchinoko.widgets.graph_widgets import CloudWidget


@pytest.fixture
def sample_data():
    """Create sample Data for graph tests with non-collinear points."""
    data = Data(dimensionality=2)
    # Use non-collinear points to avoid Delaunay triangulation errors
    # Create a simple grid pattern
    positions = [
        (0, 0), (50, 0), (100, 0),
        (0, 50), (50, 50), (100, 50),
        (0, 100), (50, 100), (100, 100),
        (25, 25)  # Add point off-grid for variety
    ]
    for i, (x, y) in enumerate(positions):
        value = float(i)
        variance = 0.1
        data.inject_new([((x, y), value, variance, {'metric1': i * 2})])
    return data


@pytest.fixture
def mock_configuration():
    """Create a mock configuration with bounds parameter."""
    mock_config = MagicMock()
    # Mock the parameter.child('bounds') to return proper bound values
    bounds_mock = MagicMock()
    bounds_mock.__getitem__ = MagicMock(side_effect=lambda key: {
        'axis_0_min': 0, 'axis_0_max': 100,
        'axis_1_min': 0, 'axis_1_max': 100
    }.get(key, 0))
    mock_config.parameter.child.return_value = bounds_mock

    ctx = ApplicationContext()
    ctx.configuration = mock_config
    ApplicationContext.set_current(ctx)
    yield mock_config
    ApplicationContext.reset()


class TestCloudGraph:
    """Tests for Cloud graph class."""

    def test_cloud_class_attributes(self):
        """Test Cloud has correct class-level attributes."""
        # Cloud uses ClassVar for data_key and accumulates
        assert Cloud.data_key is None
        assert Cloud.accumulates is True

    def test_cloud_widget_args_set_in_post_init(self):
        """Test Cloud sets widget_args in __post_init__ from class vars."""
        # Score is a concrete Cloud subclass with defined data_key
        g = Score()
        assert g.widget_args == ('scores', True)

    def test_cloud_inherits_from_graph(self):
        """Test Cloud is a Graph subclass."""
        g = Score()  # Use Score as concrete subclass
        assert isinstance(g, Graph)
        assert isinstance(g, Cloud)


class TestScoreGraph:
    """Tests for Score graph class (subclass of Cloud)."""

    def test_score_default_data_key(self):
        """Test Score has default data_key='scores'."""
        g = Score()
        assert g.data_key == 'scores'

    def test_score_default_name(self):
        """Test Score has default name='Score'."""
        g = Score()
        assert g.name == 'Score'

    def test_score_accumulates_true(self):
        """Test Score accumulates is True (inherited from Cloud)."""
        g = Score()
        assert g.accumulates is True

    def test_score_widget_args(self):
        """Test Score widget_args are set correctly."""
        g = Score()
        assert g.widget_args == ('scores', True)

    def test_score_make_widget(self, qtbot, mock_configuration):
        """Test Score creates a widget."""
        g = Score()
        widget = g.make_widget()
        assert widget is not None
        qtbot.addWidget(widget)

    def test_score_update(self, qtbot, mock_configuration, sample_data):
        """Test Score update method."""
        g = Score()
        widget = g.make_widget()
        qtbot.addWidget(widget)

        # Should not raise
        g.update(widget, sample_data, slice(0, None))


class TestVarianceGraph:
    """Tests for Variance graph class (subclass of Cloud)."""

    def test_variance_default_data_key(self):
        """Test Variance has default data_key='variances'."""
        g = Variance()
        assert g.data_key == 'variances'

    def test_variance_default_name(self):
        """Test Variance has default name='Variance'."""
        g = Variance()
        assert g.name == 'Variance'

    def test_variance_accumulates_true(self):
        """Test Variance accumulates is True (inherited from Cloud)."""
        g = Variance()
        assert g.accumulates is True

    def test_variance_widget_args(self):
        """Test Variance widget_args are set correctly."""
        g = Variance()
        assert g.widget_args == ('variances', True)

    def test_variance_make_widget(self, qtbot, mock_configuration):
        """Test Variance creates a widget."""
        g = Variance()
        widget = g.make_widget()
        assert widget is not None
        qtbot.addWidget(widget)

    def test_variance_update(self, qtbot, mock_configuration, sample_data):
        """Test Variance update method."""
        g = Variance()
        widget = g.make_widget()
        qtbot.addWidget(widget)

        # Should not raise
        g.update(widget, sample_data, slice(0, None))
