"""Tests for CloudWidget in graph_widgets module."""
import pytest
import numpy as np
from PySide6.QtCore import Qt
from pyqtgraph import InfiniteLine, PlotWidget

from tsuchinoko.widgets.graph_widgets import CloudWidget
from tsuchinoko.graphics_items.clouditem import CloudItem
from tsuchinoko.graphics_items.mixins import ClickRequesterPlot
from tsuchinoko.adaptive import Data


@pytest.fixture
def cloud_widget(qtbot):
    """Create a CloudWidget for testing."""
    widget = CloudWidget(data_key='scores', accumulates=True)
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def non_accumulating_widget(qtbot):
    """Create a non-accumulating CloudWidget for testing."""
    widget = CloudWidget(data_key='variances', accumulates=False)
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def sample_data():
    """Create sample data with positions, scores, and variances."""
    data = Data(dimensionality=2)
    for i in range(10):
        data.inject_new([((i * 10, i * 10), float(i), 0.1 * i, {})])
    return data


class TestCloudWidgetInit:
    """Tests for CloudWidget initialization."""

    def test_init_data_key(self, cloud_widget):
        """Test widget stores data_key."""
        assert cloud_widget.data_key == 'scores'

    def test_init_accumulates(self, cloud_widget):
        """Test widget stores accumulates flag."""
        assert cloud_widget.accumulates is True

    def test_init_non_accumulating(self, non_accumulating_widget):
        """Test widget stores accumulates=False."""
        assert non_accumulating_widget.accumulates is False
        assert non_accumulating_widget.data_key == 'variances'

    def test_init_has_graph(self, cloud_widget):
        """Test widget has a graph plot of correct type."""
        assert cloud_widget.graph is not None
        assert isinstance(cloud_widget.graph, ClickRequesterPlot)

    def test_init_has_timeline(self, cloud_widget):
        """Test widget has a timeline (InfiniteLine)."""
        assert cloud_widget.timeline is not None
        assert isinstance(cloud_widget.timeline, InfiniteLine)

    def test_init_has_timeline_plot(self, cloud_widget):
        """Test widget has a timeline_plot (PlotWidget)."""
        assert cloud_widget.timeline_plot is not None
        assert isinstance(cloud_widget.timeline_plot, PlotWidget)

    def test_init_has_cloud(self, cloud_widget):
        """Test widget has a cloud item of correct type."""
        assert cloud_widget.cloud is not None
        assert isinstance(cloud_widget.cloud, CloudItem)

    def test_init_has_arrows(self, cloud_widget):
        """Test widget has max_arrow and last_arrow."""
        assert cloud_widget.max_arrow is not None
        assert cloud_widget.last_arrow is not None

    def test_init_empty_cache(self, cloud_widget):
        """Test widget starts with empty cache."""
        assert cloud_widget.cache == dict()

    def test_init_timeline_movable(self, cloud_widget):
        """Test timeline is movable."""
        assert cloud_widget.timeline.movable is True

    def test_init_timeline_plot_height(self, cloud_widget):
        """Test timeline plot has fixed height."""
        assert cloud_widget.timeline_plot.maximumHeight() == 40

    def test_init_output_selector_hidden(self, cloud_widget):
        """Test output selector is initially hidden."""
        assert cloud_widget.output_selector.isHidden() is True


class TestCloudWidgetCache:
    """Tests for CloudWidget cache functionality."""

    def test_empty_cache_returns_none_for_nframes(self, cloud_widget):
        """Test nframes returns None when cache is empty."""
        assert cloud_widget.nframes() is None

    def test_timeline_changed_with_empty_cache(self, cloud_widget):
        """Test timeline_changed does nothing with empty cache."""
        # Should not raise an exception
        cloud_widget.timeline_changed()


class TestCloudWidgetPlayback:
    """Tests for CloudWidget playback functionality."""

    def test_initial_play_rate(self, cloud_widget):
        """Test initial play rate is zero."""
        assert cloud_widget.play_rate == 0

    def test_initial_fps(self, cloud_widget):
        """Test initial fps is 1 Hz."""
        assert cloud_widget.fps == 1

    def test_play_timer_not_active(self, cloud_widget):
        """Test play timer is not active initially."""
        assert cloud_widget.play_timer.isActive() is False

    def test_play_zero_stops_timer(self, cloud_widget):
        """Test play(0) stops the timer."""
        cloud_widget.play(0)
        assert cloud_widget.play_timer.isActive() is False
        assert cloud_widget.play_rate == 0

    def test_keys_pressed_empty(self, cloud_widget):
        """Test keysPressed dict is initially empty."""
        assert cloud_widget.keysPressed == {}
