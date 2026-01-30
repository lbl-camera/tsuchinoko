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


from tsuchinoko.graphs.common import Cloud, Score, Variance, Scatter, Plot
from tsuchinoko.adaptive import Data
from tsuchinoko.widgets.context import ApplicationContext
from tsuchinoko.widgets.graph_widgets import CloudWidget
from pyqtgraph import PlotWidget


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


class TestScatter:
    """Tests for Scatter graph class.

    Scatter creates scatter plots with KMeans clustering for visualization.
    It requires x_key and y_key to specify which data fields to use for
    x and y coordinates, and n_clusters for the KMeans clustering.
    """

    def test_scatter_inherits_from_graph(self):
        """Test Scatter is a Graph subclass."""
        g = Scatter(x_key='scores', y_key='variances', n_clusters=2)
        assert isinstance(g, Graph)

    def test_scatter_widget_class(self):
        """Test Scatter uses PlotWidget as widget_class."""
        assert Scatter.widget_class == PlotWidget

    def test_scatter_init_stores_parameters(self):
        """Test Scatter __init__ stores x_key, y_key, n_clusters."""
        g = Scatter(x_key='scores', y_key='variances', n_clusters=3, kmeans_kwargs={'n_init': 10})
        assert g.x_key == 'scores'
        assert g.y_key == 'variances'
        assert g.n_clusters == 3
        assert g.kmeans_kwargs == {'n_init': 10}

    def test_scatter_init_default_kmeans_kwargs(self):
        """Test Scatter __init__ with default kmeans_kwargs."""
        g = Scatter(x_key='scores', y_key='variances', n_clusters=2)
        assert g.kmeans_kwargs is None

    def test_scatter_has_unique_id(self):
        """Test Scatter instances have unique IDs."""
        g1 = Scatter(x_key='scores', y_key='variances', n_clusters=2)
        g2 = Scatter(x_key='scores', y_key='variances', n_clusters=2)
        assert g1.id != g2.id

    def test_scatter_make_widget(self, qtbot):
        """Test Scatter creates a PlotWidget."""
        g = Scatter(x_key='scores', y_key='variances', n_clusters=2)
        widget = g.make_widget()
        assert widget is not None
        assert isinstance(widget, PlotWidget)
        qtbot.addWidget(widget)

    @pytest.mark.xfail(
        reason="Scatter.update uses np.dstack which creates 3D array, but KMeans requires 2D. "
               "Should use np.column_stack instead."
    )
    def test_scatter_update_with_sufficient_data(self, qtbot):
        """Test Scatter update method with sufficient data for KMeans.

        KMeans requires data shape to be valid. The data needs to have
        enough points and proper dimensionality for clustering.

        Note: This test is expected to fail because Scatter.update uses
        np.dstack((x, y)) which creates a 3D array, but KMeans requires
        a 2D array. The fix would be to use np.column_stack((x, y)) instead.
        """
        # Create data with enough points for 2 clusters
        data = Data(dimensionality=2)
        # Create 20 points in two distinct clusters
        for i in range(10):
            data.inject_new([((i, i), float(i), 0.1, {})])  # Cluster 1: diagonal
        for i in range(10):
            data.inject_new([((i + 50, 100 - i), float(i + 10), 0.1, {})])  # Cluster 2: off diagonal

        g = Scatter(x_key='scores', y_key='variances', n_clusters=2, kmeans_kwargs={'n_init': 'auto'})
        widget = g.make_widget()
        qtbot.addWidget(widget)

        # Should not raise
        g.update(widget, data, slice(0, None))

        # After update, widget should have scatter plot items (one per cluster)
        items = widget.getPlotItem().items
        assert len(items) == 2  # One ScatterPlotItem per cluster


class TestPlot:
    """Tests for Plot graph class (creates line plots).

    Plot is a dataclass that uses ClassVar for data_key, meaning
    subclasses define their own data_key at class level. The
    label_key is an InitVar used only in __post_init__.
    """

    def test_plot_class_data_key_is_classvar(self):
        """Test Plot.data_key is a ClassVar (None by default)."""
        assert Plot.data_key is None

    def test_plot_widget_class(self):
        """Test Plot uses PlotGraphWidget as widget_class."""
        from tsuchinoko.graphs.common import PlotGraphWidget
        assert Plot.widget_class == PlotGraphWidget

    def test_plot_default_accumulates(self):
        """Test Plot accumulates defaults to False."""
        g = Plot()
        assert g.accumulates is False

    def test_plot_with_accumulates(self):
        """Test Plot with accumulates=True."""
        g = Plot(accumulates=True)
        assert g.accumulates is True

    def test_plot_inherits_from_graph(self):
        """Test Plot is a Graph subclass."""
        g = Plot()
        assert isinstance(g, Graph)

    def test_plot_make_widget(self, qtbot):
        """Test Plot creates a PlotGraphWidget."""
        from tsuchinoko.graphs.common import PlotGraphWidget
        g = Plot()
        widget = g.make_widget()
        assert widget is not None
        assert isinstance(widget, PlotGraphWidget)
        qtbot.addWidget(widget)

    def test_plot_make_widget_stores_label_key(self, qtbot):
        """Test Plot stores label_key in widget_kwargs via __post_init__."""
        g = Plot(label_key='test_label')
        assert g.widget_kwargs['label_key'] == 'test_label'
        widget = g.make_widget()
        qtbot.addWidget(widget)

    def test_plot_has_unique_id(self):
        """Test Plot instances have unique IDs."""
        g1 = Plot()
        g2 = Plot()
        assert g1.id != g2.id


class TestScorePlotSubclass:
    """Tests for a Plot subclass pattern.

    Since Plot uses ClassVar for data_key, we test how subclasses
    would define their own data_key.
    """

    def test_creating_plot_subclass(self):
        """Test creating a Plot subclass with custom data_key."""
        from dataclasses import dataclass
        from typing import ClassVar

        @dataclass(eq=False)
        class ScorePlot(Plot):
            data_key: ClassVar[str] = 'scores'

        g = ScorePlot()
        assert g.data_key == 'scores'
        assert isinstance(g, Plot)
        assert isinstance(g, Graph)
