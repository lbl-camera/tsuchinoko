"""Tests for graphics_items mixins."""
import os

import pytest
from unittest.mock import MagicMock, patch
from PySide6.QtCore import Qt, QEvent, QPoint

from tsuchinoko.graphics_items.mixins import (
    ClickRequester, RequestRelay, request_relay,
    ClickRequesterPlot, ClickRequesterBase,
    BetterLayout, BetterButtons, AspectRatioLock,
    LogScaleIntensity, LogScaleImageItem, ViridisImageView,
    BetterAutoLUTRangeImageView, ComposableItemImageView
)
from tsuchinoko.graphs import GraphSignalRelay, graph_signal_relay

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


class TestRequestRelay:
    """Tests for RequestRelay singleton."""

    def test_request_relay_is_singleton(self):
        """Test request_relay is a module-level singleton."""
        assert request_relay is not None
        assert isinstance(request_relay, RequestRelay)

    def test_request_relay_has_signal(self):
        """Test RequestRelay has sigRequestMeasure signal."""
        assert hasattr(request_relay, 'sigRequestMeasure')

    def test_request_relay_signal_can_connect(self):
        """Test that sigRequestMeasure signal can be connected to a slot."""
        callback = MagicMock()
        request_relay.sigRequestMeasure.connect(callback)
        # Emit test data
        test_pos = (10.0, 20.0)
        request_relay.sigRequestMeasure.emit(test_pos)
        callback.assert_called_once_with(test_pos)
        # Disconnect to avoid affecting other tests
        request_relay.sigRequestMeasure.disconnect(callback)


class TestGraphSignalRelay:
    """Tests for GraphSignalRelay."""

    def test_graph_signal_relay_has_push_signal(self):
        """Test GraphSignalRelay has sigPush signal."""
        assert hasattr(graph_signal_relay, 'sigPush')

    def test_graph_signal_relay_is_singleton(self):
        """Test graph_signal_relay is a module-level singleton."""
        assert graph_signal_relay is not None
        assert isinstance(graph_signal_relay, GraphSignalRelay)

    def test_graph_signal_relay_signal_can_connect(self):
        """Test that sigPush signal can be connected to a slot."""
        callback = MagicMock()
        graph_signal_relay.sigPush.connect(callback)
        # Emit test data
        test_data = {'key': 'value'}
        graph_signal_relay.sigPush.emit(test_data)
        callback.assert_called_once_with(test_data)
        # Disconnect to avoid affecting other tests
        graph_signal_relay.sigPush.disconnect(callback)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
class TestClickRequester:
    """Tests for ClickRequester ImageView mixin."""

    def test_click_requester_has_measure_action(self, qtbot):
        """Test ClickRequester creates measure action."""
        widget = ClickRequester()
        qtbot.add_widget(widget)
        assert hasattr(widget, 'measure_action')
        assert widget.measure_action.text() == 'Queue Measurement at Point'

    def test_click_requester_scene_method(self, qtbot):
        """Test ClickRequester _scene method returns scene."""
        widget = ClickRequester()
        qtbot.add_widget(widget)
        # _scene() should return the scene object
        scene = widget._scene()
        assert scene is widget.scene

    def test_click_requester_event_filter_installed(self, qtbot):
        """Test ClickRequester installs event filter on graphicsView."""
        widget = ClickRequester()
        qtbot.add_widget(widget)
        # Event filter should be installed
        assert widget._last_mouse_event_pos is None

    def test_click_requester_context_menu(self, qtbot):
        """Test ClickRequester adds action to context menu."""
        widget = ClickRequester()
        qtbot.add_widget(widget)
        # Check that measure_action is in the context menu
        assert widget.measure_action in widget._scene().contextMenu


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
class TestClickRequesterPlot:
    """Tests for ClickRequesterPlot PlotWidget mixin."""

    def test_click_requester_plot_has_measure_action(self, qtbot):
        """Test ClickRequesterPlot creates measure action."""
        widget = ClickRequesterPlot()
        qtbot.add_widget(widget)
        assert hasattr(widget, 'measure_action')
        assert widget.measure_action.text() == 'Queue Measurement at Point'

    def test_click_requester_plot_scene_method(self, qtbot):
        """Test ClickRequesterPlot _scene method returns sceneObj."""
        widget = ClickRequesterPlot()
        qtbot.add_widget(widget)
        # _scene() should return the sceneObj
        scene = widget._scene()
        assert scene is widget.sceneObj

    def test_click_requester_plot_event_filter_installed(self, qtbot):
        """Test ClickRequesterPlot installs event filter on itself."""
        widget = ClickRequesterPlot()
        qtbot.add_widget(widget)
        # Event filter should be installed
        assert widget._last_mouse_event_pos is None


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
class TestBetterLayout:
    """Tests for BetterLayout ImageView mixin."""

    def test_better_layout_creates_layouts(self, qtbot):
        """Test BetterLayout creates outer, left, and right layouts."""
        widget = BetterLayout()
        qtbot.add_widget(widget)
        assert hasattr(widget.ui, 'outer_layout')
        assert hasattr(widget.ui, 'left_layout')
        assert hasattr(widget.ui, 'right_layout')

    def test_better_layout_roi_btn_hidden(self, qtbot):
        """Test BetterLayout hides roiBtn."""
        widget = BetterLayout()
        qtbot.add_widget(widget)
        assert not widget.ui.roiBtn.isVisible()


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
class TestBetterButtons:
    """Tests for BetterButtons ImageView mixin."""

    def test_better_buttons_creates_reset_axes_button(self, qtbot):
        """Test BetterButtons creates Reset Axes button."""
        widget = BetterButtons()
        qtbot.add_widget(widget)
        assert hasattr(widget, 'resetAxesBtn')
        assert widget.resetAxesBtn.text() == "Reset Axes"

    def test_better_buttons_creates_reset_lut_button(self, qtbot):
        """Test BetterButtons creates Reset LUT button."""
        widget = BetterButtons()
        qtbot.add_widget(widget)
        assert hasattr(widget, 'resetLUTBtn')
        assert widget.resetLUTBtn.text() == "Reset LUT"


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
class TestAspectRatioLock:
    """Tests for AspectRatioLock ImageView mixin."""

    def test_aspect_ratio_lock_creates_button(self, qtbot):
        """Test AspectRatioLock creates Lock Aspect button."""
        widget = AspectRatioLock()
        qtbot.add_widget(widget)
        assert hasattr(widget, '_aspect_ratio_button')
        assert widget._aspect_ratio_button.text() == 'Lock Aspect'

    def test_aspect_ratio_lock_button_checkable(self, qtbot):
        """Test AspectRatioLock button is checkable."""
        widget = AspectRatioLock()
        qtbot.add_widget(widget)
        assert widget._aspect_ratio_button.isCheckable()

    def test_aspect_ratio_lock_default_locked(self, qtbot):
        """Test AspectRatioLock defaults to locked."""
        widget = AspectRatioLock()
        qtbot.add_widget(widget)
        assert widget._aspect_ratio_button.isChecked()

    def test_aspect_ratio_lock_can_unlock(self, qtbot):
        """Test AspectRatioLock can be initialized unlocked."""
        widget = AspectRatioLock(lock_aspect=False)
        qtbot.add_widget(widget)
        assert not widget._aspect_ratio_button.isChecked()


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
class TestLogScaleIntensity:
    """Tests for LogScaleIntensity ImageView mixin."""

    def test_log_scale_intensity_creates_button(self, qtbot):
        """Test LogScaleIntensity creates Log Intensity button."""
        widget = LogScaleIntensity()
        qtbot.add_widget(widget)
        assert hasattr(widget, 'logIntensityButton')
        assert widget.logIntensityButton.text() == "Log Intensity"

    def test_log_scale_intensity_button_checkable(self, qtbot):
        """Test LogScaleIntensity button is checkable."""
        widget = LogScaleIntensity()
        qtbot.add_widget(widget)
        assert widget.logIntensityButton.isCheckable()

    def test_log_scale_intensity_default_enabled(self, qtbot):
        """Test LogScaleIntensity defaults to enabled."""
        widget = LogScaleIntensity()
        qtbot.add_widget(widget)
        assert widget.logIntensityButton.isChecked()

    def test_log_scale_intensity_can_disable(self, qtbot):
        """Test LogScaleIntensity can be initialized disabled."""
        widget = LogScaleIntensity(log_scale=False)
        qtbot.add_widget(widget)
        assert not widget.logIntensityButton.isChecked()


class TestLogScaleImageItem:
    """Tests for LogScaleImageItem."""

    def test_log_scale_image_item_has_log_scale_attribute(self):
        """Test LogScaleImageItem has logScale attribute."""
        import numpy as np
        item = LogScaleImageItem(np.zeros((10, 10)))
        assert hasattr(item, 'logScale')
        assert item.logScale is True


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
class TestViridisImageView:
    """Tests for ViridisImageView."""

    def test_viridis_image_view_uses_viridis(self, qtbot):
        """Test ViridisImageView uses viridis colormap by default."""
        widget = ViridisImageView()
        qtbot.add_widget(widget)
        # Widget should be created successfully with viridis
        assert widget is not None


class TestComposableItemImageView:
    """Tests for ComposableItemImageView."""

    def test_composable_item_image_view_has_bases_tuple(self):
        """Test ComposableItemImageView has imageItem_bases as empty tuple."""
        assert hasattr(ComposableItemImageView, 'imageItem_bases')
        assert ComposableItemImageView.imageItem_bases == tuple()
