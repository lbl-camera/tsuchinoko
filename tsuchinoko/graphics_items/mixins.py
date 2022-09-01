from pyqtgraph import ImageView, PlotWidget
from qtpy.QtCore import Signal, QObject, QEvent, Qt
from qtpy.QtWidgets import QAction

# TODO: map imageitems into target coordinate domain


class RequestRelay(QObject):
    sigRequestMeasure = Signal(tuple)


request_relay = RequestRelay()


class ClickRequesterBase:
    def __init__(self, *args, **kwargs):
        super(ClickRequesterBase, self).__init__(*args, **kwargs)

        self.measure_action = QAction('Queue Measurement at Point')
        self.measure_action.triggered.connect(self.emit_measure_request)
        self._scene().contextMenu.append(self.measure_action)
        self._last_mouse_event_pos = None
        self._install_filter()

    def _install_filter(self):
        ...

    def _scene(self):
        ...

    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.Type.MouseButtonPress:
            if ev.button() == Qt.MouseButton.RightButton:
                self._last_mouse_event_pos = ev.pos()

        return False


class ClickRequester(ClickRequesterBase, ImageView):
    def _scene(self):
        return self.scene

    def _install_filter(self):
        self.ui.graphicsView.installEventFilter(self)

    def emit_measure_request(self, *_):
        app_pos = self._last_mouse_event_pos
        # map to local pos
        local_pos = self.view.vb.mapSceneToView(app_pos)
        request_relay.sigRequestMeasure.emit(local_pos.toTuple())


class ClickRequesterPlot(ClickRequesterBase, PlotWidget):
    def _install_filter(self):
        self.installEventFilter(self)

    def _scene(self):
        return self.sceneObj

    def emit_measure_request(self, *_):
        app_pos = self._last_mouse_event_pos
        # map to local pos
        local_pos = self.plotItem.vb.mapSceneToView(app_pos)
        request_relay.sigRequestMeasure.emit(local_pos.toTuple())
