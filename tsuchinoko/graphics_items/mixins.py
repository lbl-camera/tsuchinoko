from pyqtgraph import ImageView, PlotWidget, RectROI
from qtpy.QtCore import Signal, QObject, QEvent, Qt, QSignalBlocker
from qtpy.QtWidgets import QAction, QApplication, QWidget

from tsuchinoko.widgets.displays import Configuration


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


class DomainROI(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        bounds = [tuple(Configuration().parameter.child('bounds')[f'axis_{i}_{limit}']
                   for limit in ['min', 'max'])
                  for i in range(2)]
        self._domain_roi = RectROI((bounds[0][0],bounds[1][0]),
                                    (bounds[0][1]-bounds[0][0], bounds[1][1]-bounds[1][0]))
        self._domain_roi.sigRegionChangeFinished.connect(self.update_bounds)
        self.getView().addItem(self._domain_roi)
        bounds_param = Configuration().parameter.child('bounds')
        for child in bounds_param.children():
            child.sigValueChanged.connect(self.update_roi)

    def update_bounds(self):
        bounds_param = Configuration().parameter.child('bounds')
        for child in bounds_param.children():
            child.sigValueChanged.disconnect(self.update_roi)
        bounds_param[f'axis_0_min'] = self._domain_roi.pos().x()
        bounds_param[f'axis_1_min'] = self._domain_roi.pos().y()
        bounds_param[f'axis_0_max'] = self._domain_roi.pos().x() + self._domain_roi.size().x()
        bounds_param[f'axis_1_max'] = self._domain_roi.pos().y() + self._domain_roi.size().y()
        for child in bounds_param.children():
            child.sigValueChanged.connect(self.update_roi)

    def update_roi(self):
        bounds_param = Configuration().parameter.child('bounds')
        blocker = QSignalBlocker(self._domain_roi)
        self._domain_roi.setPos(bounds_param[f'axis_0_min'], bounds_param[f'axis_1_min'], update=False)
        self._domain_roi.setSize(bounds_param[f'axis_0_max'] - bounds_param[f'axis_0_min'],
                                 bounds_param[f'axis_1_max'] - bounds_param[f'axis_1_min'])
