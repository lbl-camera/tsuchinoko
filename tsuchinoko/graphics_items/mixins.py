from functools import cached_property

import numpy as np
from pyqtgraph import ImageView, GraphicsObject, PlotWidget
from qtpy.QtCore import QEvent, QRect, Qt, QPoint, Signal, QObject
from qtpy.QtGui import QPainterPath, QMouseEvent
from qtpy.QtWidgets import QMenu, QAction, QApplication


class RequestRelay(QObject):
    sigRequestMeasure = Signal(tuple)


request_relay = RequestRelay()


class ClickRequester(ImageView):
    def __init__(self, *args, **kwargs):
        super(ClickRequester, self).__init__(*args, **kwargs)

        self.measure_action = QAction('Queue Measurement at Point')
        self.measure_action.triggered.connect(self.emit_measure_request)
        self.scene.contextMenu.append(self.measure_action)
        self._last_mouse_event_pos = None
        self.ui.graphicsView.installEventFilter(self)

    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.Type.MouseButtonPress:
            if ev.button() == Qt.MouseButton.RightButton:
                self._last_mouse_event_pos = ev.pos()

        return False



    def emit_measure_request(self, *_):
        app_pos = self._last_mouse_event_pos
        # map to local pos
        local_pos = self.view.vb.mapSceneToView(app_pos)
        request_relay.sigRequestMeasure.emit(local_pos.toTuple())


class ClickRequesterPlot(PlotWidget):
    def __init__(self, *args, **kwargs):
        super(ClickRequesterPlot, self).__init__(*args, **kwargs)

        self.measure_action = QAction('Queue Measurement at Point')
        self.measure_action.triggered.connect(self.emit_measure_request)
        self.sceneObj.contextMenu.append(self.measure_action)
        self._last_mouse_event_pos = None
        self.installEventFilter(self)

    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.Type.MouseButtonPress:
            if ev.button() == Qt.MouseButton.RightButton:
                self._last_mouse_event_pos = ev.pos()

        return False

    def emit_measure_request(self, *_):
        app_pos = self._last_mouse_event_pos
        # map to local pos
        local_pos = self.plotItem.vb.mapSceneToView(app_pos)
        request_relay.sigRequestMeasure.emit(local_pos.toTuple())
