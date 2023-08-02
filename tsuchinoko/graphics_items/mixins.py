from typing import Callable

import numpy as np
from pyqtgraph import ImageView, PlotWidget, RectROI, ImageItem, PlotItem
from qtpy.QtCore import QPointF, Signal, QObject, QEvent, Qt, QSignalBlocker
from qtpy.QtWidgets import QAction, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QSizePolicy
from pyqtgraph import functions as fn, debug, Point

from tsuchinoko.widgets.displays import Configuration


# TODO: map imageitems into target coordinate domain


class RequestRelay(QObject):
    sigRequestMeasure = Signal(tuple)


request_relay = RequestRelay()


class BetterLayout(ImageView):
    # Replaces awkward gridlayout with more structured v/hboxlayouts, and removes useless buttons
    def __init__(self, *args, **kwargs):
        super(BetterLayout, self).__init__(*args, **kwargs)

        # Shrink LUT
        self.getHistogramWidget().setMinimumWidth(10)

        self._reset_layout()
        self._set_layout()

    def _set_layout(self, layout=None):
        # Replace the layout
        QWidget().setLayout(self.ui.layoutWidget.layout())
        if layout is not None:
            self.ui.layoutWidget.setLayout(layout)
        else:
            self.ui.layoutWidget.setLayout(self.ui.outer_layout)

    def _reset_layout(self):
        self.ui.outer_layout = QHBoxLayout()
        self.ui.left_layout = QVBoxLayout()
        self.ui.right_layout = QVBoxLayout()
        self.ui.outer_layout.addLayout(self.ui.left_layout)
        self.ui.outer_layout.addLayout(self.ui.right_layout)
        for layout in [self.ui.outer_layout, self.ui.left_layout, self.ui.right_layout]:
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)

        self.ui.left_layout.addWidget(self.ui.graphicsView)
        self.ui.right_layout.addWidget(self.ui.histogram)

        # Must keep the roiBtn around; ImageView expects to be able to check its state
        self.ui.roiBtn.setParent(self)
        self.ui.roiBtn.hide()


class BetterButtons(BetterLayout):
    def __init__(self, *args, **kwargs):
        super(BetterButtons, self).__init__(*args, **kwargs)

        # Setup axes reset button
        self.resetAxesBtn = QPushButton("Reset Axes")
        sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.resetAxesBtn.sizePolicy().hasHeightForWidth())
        self.resetAxesBtn.setSizePolicy(sizePolicy)
        self.resetAxesBtn.setObjectName("resetAxes")
        self.ui.right_layout.addWidget(self.resetAxesBtn)
        self.resetAxesBtn.clicked.connect(self.autoRange)

        # Setup LUT reset button
        self.resetLUTBtn = QPushButton("Reset LUT")
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.resetLUTBtn.sizePolicy().hasHeightForWidth())
        # self.resetLUTBtn.setSizePolicy(sizePolicy)
        # self.resetLUTBtn.setObjectName("resetLUTBtn")
        self.ui.right_layout.addWidget(self.resetLUTBtn)
        self.resetLUTBtn.clicked.connect(self.autoLevels)


class AspectRatioLock(BetterLayout):
    def __init__(self, *args, lock_aspect=True, **kwargs):
        super().__init__(*args, **kwargs)
        self._aspect_ratio_button = QPushButton('Lock Aspect')
        self._aspect_ratio_button.setCheckable(True)
        self.ui.right_layout.addWidget(self._aspect_ratio_button)
        self._aspect_ratio_button.setChecked(lock_aspect)
        self._aspect_ratio_button.toggled.connect(self.toggle_lock_aspect)

    def toggle_lock_aspect(self):
        if self._aspect_ratio_button.isChecked():
            self.view.setAspectLocked(True)
        else:
            self.view.setAspectLocked(False)


class ComposableItemImageView(ImageView):
    """
    Used to compose together different image view mixins that may use different ItemImage subclasses.
    See LogScaleIntensity, LogScaleImageItem, ImageViewHistogramOverflowFIx, ImageItemHistorgramOverflowFix.
    Note that any imageItem named argument passed into the ImageView mixins above will discard the item and instead
    create a composition of imageItem_bases with their respective ImageItem class.
    """

    imageItem_bases = tuple()


class LogScaleImageItem(ImageItem):
    def __init__(self, *args, **kwargs):
        super(LogScaleImageItem, self).__init__(*args, **kwargs)
        self.logScale = True

    def render(self):
        # Convert data to QImage for display.

        profile = debug.Profiler()
        if self.image is None or self.image.size == 0:
            return
        if isinstance(self.lut, Callable):
            lut = self.lut(self.image)
        else:
            lut = self.lut

        if self.logScale:
            image = self.image + 1
            with np.errstate(invalid="ignore", divide='ignore'):
                image = image.astype(float)
                np.log(image, where=image >= 0, out=image)  # map to 0-255
        else:
            image = self.image

        if self.autoDownsample:
            # reduce dimensions of image based on screen resolution
            o = self.mapToDevice(QPointF(0, 0))
            x = self.mapToDevice(QPointF(1, 0))
            y = self.mapToDevice(QPointF(0, 1))
            w = Point(x - o).length()
            h = Point(y - o).length()
            if w == 0 or h == 0:
                self.qimage = None
                return
            xds = max(1, int(1.0 / w))
            yds = max(1, int(1.0 / h))
            axes = [1, 0] if self.axisOrder == "row-major" else [0, 1]
            image = fn.downsample(image, xds, axis=axes[0])
            image = fn.downsample(image, yds, axis=axes[1])
            self._lastDownsample = (xds, yds)
        else:
            pass

        # if the image data is a small int, then we can combine levels + lut
        # into a single lut for better performance
        levels = self.levels
        if levels is not None and levels.ndim == 1 and image.dtype in (np.ubyte, np.uint16):
            if self._effectiveLut is None:
                eflsize = 2 ** (image.itemsize * 8)
                ind = np.arange(eflsize)
                minlev, maxlev = levels
                levdiff = maxlev - minlev
                levdiff = 1 if levdiff == 0 else levdiff  # don't allow division by 0
                if lut is None:
                    efflut = fn.rescaleData(ind, scale=255.0 / levdiff, offset=minlev, dtype=np.ubyte)
                else:
                    lutdtype = np.min_scalar_type(lut.shape[0] - 1)
                    efflut = fn.rescaleData(
                        ind,
                        scale=(lut.shape[0] - 1) / levdiff,
                        offset=minlev,
                        dtype=lutdtype,
                        clip=(0, lut.shape[0] - 1)
                    )
                    efflut = lut[efflut]

                self._effectiveLut = efflut
            lut = self._effectiveLut
            levels = None

        # Assume images are in column-major order for backward compatibility
        # (most images are in row-major order)

        if self.axisOrder == "col-major":
            image = image.transpose((1, 0, 2)[: image.ndim])

        if self.logScale:
            with np.errstate(invalid="ignore"):
                levels = np.log(np.add(levels, 1))
            levels[0] = np.nanmax([levels[0], 0])

        argb, alpha = fn.makeARGB(image, lut=lut, levels=levels)
        self.qimage = fn.makeQImage(argb, alpha, transpose=False)


class LogScaleIntensity(BetterLayout, ComposableItemImageView):
    def __init__(self, *args, log_scale=True, **kwargs):
        # Composes a new type consisting of any ImageItem types in imageItem_bases with this classes's helper ImageItem
        # class (LogScaleImageItem)
        self.imageItem_bases += (LogScaleImageItem,)
        imageItem = type("DynamicImageItem", tuple(self.imageItem_bases), {})(np.zeros((1, 1)))
        if "imageItem" in kwargs:
            del kwargs["imageItem"]
        super(LogScaleIntensity, self).__init__(imageItem=imageItem, *args, **kwargs)

        self.logScale = log_scale

        # Setup log scale button
        self.logIntensityButton = QPushButton("Log Intensity")
        sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.logIntensityButton.sizePolicy().hasHeightForWidth())
        self.logIntensityButton.setSizePolicy(sizePolicy)
        self.logIntensityButton.setObjectName("logIntensity")
        self.ui.right_layout.addWidget(self.logIntensityButton)
        self.logIntensityButton.setCheckable(True)
        self.setLogScale(self.logScale)
        self.logIntensityButton.clicked.connect(self._setLogScale)

    def _setLogScale(self, value):
        self.imageItem.logScale = value
        self.imageItem.qimage = None
        self.imageItem.update()
        self.getHistogramWidget().region.setBounds([0 if value else None, None])

    def setLogScale(self, value):
        self._setLogScale(value)
        self.logIntensityButton.setChecked(value)


class ViridisImageView(ImageView):
    def __init__(self, *args, **kwargs):
        super(ViridisImageView, self).__init__(*args, **kwargs)

        # Use Viridis by default
        self.setPredefinedGradient("viridis")


class BetterAutoLUTRangeImageView(ImageView):
    def __init__(self, *args, **kwargs):
        super(BetterAutoLUTRangeImageView, self).__init__(*args, **kwargs)

    def quickMinMax(self, data):
        """
        Estimate the min/max values of *data* by subsampling. MODIFIED TO USE THE 99TH PERCENTILE instead of max.
        """
        if data is None:
            return 0, 0
        ax = np.argmax(data.shape)
        sl = [slice(None)] * data.ndim
        sl[ax] = slice(None, None, max(1, int(data.size // 1e6)))
        for axis_sl in sl:
            data = data[axis_sl]
        return [(
            np.nanpercentile(np.where(data > np.nanmin(data), data, np.nanmax(data)), 2),
            np.nanpercentile(np.where(data < np.nanmax(data), data, np.nanmin(data)), 98),
        )]


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
        ev.ignore()

        return False


class YInvert(ImageView):
    def __init__(self, *args, invert_y=False, **kwargs):
        if 'view' in kwargs:
            raise ValueError(f'Setting view is incompatible with this widget ({type(self)}')
        graph = PlotItem()
        super().__init__(*args, view=graph, **kwargs)
        graph.vb.invertY(invert_y)


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
        self._domain_roi = RectROI((bounds[0][0], bounds[1][0]),
                                   (bounds[0][1] - bounds[0][0], bounds[1][1] - bounds[1][0]))
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
