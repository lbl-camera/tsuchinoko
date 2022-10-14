import sys
import warnings
from collections.abc import Callable

import numpy as np

import pyqtgraph as pg
from numpy import math
from pyqtgraph import debug, getConfigOption, functions as fn, colormap, ColorMap, getCupy, rescaleData, applyLookupTable
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
from pyqtgraph.graphicsItems.ScatterPlotItem import SymbolAtlas, ScatterPlotItem
from scipy.spatial import Delaunay

pg.setConfigOption('useOpenGL', True)
pg.setConfigOption('enableExperimental', True)


HAVE_OPENGL = hasattr(QtWidgets, 'QOpenGLWidget')


class CloudItem(pg.GraphicsObject):
    """
    An opengl point cloud visualization utilizing Delaunay Triangulation. Hardware acceleration is required.
    """

    sigDataChanged = sigImageChanged = QtCore.Signal()
    sigClicked = QtCore.Signal(object, object)
    sigPointsClicked = QtCore.Signal(object, object, object)
    sigPointsHovered = QtCore.Signal(object, object, object)

    def __init__(self, **kwargs):
        super(CloudItem, self).__init__(kwargs.get('parent', None))
        self.levels = None
        self.clear()

        self.scatter = ScatterPlotItem(**kwargs)
        self.scatter.setParentItem(self)
        self.scatter.sigClicked.connect(self.scatterClicked)
        self.scatter.sigHovered.connect(self.scatterHovered)

        self.metaData = {}
        self.opts = {
            'name': None,
            'antialias': getConfigOption('antialias'),
            'mouseWidth': 8, # width of shape responding to mouse click
            'compositionMode': None,
            'pen': None,
            'hoverable': False,
            'tip': None,
        }

        # self.setClickable(kwargs.get('clickable', False))

        if 'x' in kwargs:
            self.setData(**kwargs)

    def setData(self, **kwargs):
        """
        =============== =================================================================
        **Arguments:**
        x, y            (numpy arrays) Data positions to display
        c               (numpy array) Data values to display
        pen             Pen to use when drawing. Any single argument accepted by
                        :func:`mkPen <pyqtgraph.mkPen>` is allowed.
        antialias       (bool) Whether to use antialiasing when drawing. This
                        is disabled by default because it decreases performance.
        compositionMode See :func:`setCompositionMode
                        <pyqtgraph.PlotCurveItem.setCompositionMode>`.
        colorMap :      pg.ColorMap, default pg.colormap.get('viridis')
                        Colormap used to map the z value to colors.
        *hoverable*            If True, sigHovered is emitted with a list of hovered points, a tool tip is shown containing
                        information about them, and an optional separate style for them is used. Default is False.
        *tip*           A string-valued function of a spot's (x, y, data) values. Set to None to prevent a tool tip
                        from being shown.
        *hoverSymbol*   A single symbol to use for hovered spots. Set to None to keep symbol unchanged. Default is None.
        *hoverSize*     A single size to use for hovered spots. Set to -1 to keep size unchanged. Default is -1.
        *hoverPen*      A single pen to use for hovered spots. Set to None to keep pen unchanged. Default is None.
        *hoverBrush*    A single brush to use for hovered spots. Set to None to keep brush unchanged. Default is None.
        =============== =================================================================

        **Notes on performance:**

        Line widths greater than 1 pixel affect the performance as discussed in
        the documentation of :class:`PlotDataItem <pyqtgraph.PlotDataItem>`.
        """
        if 'autoLevels' not in kwargs:
            kwargs['autoLevels'] = True
        self.updateData(**kwargs)

    def updateData(self, **kwargs):
        self.clear()
        self.scatter.clear()
        self.extendData(**kwargs)

    def extendData(self, x, y, c, autoLevels=False, **kwargs):
        profiler = debug.Profiler()

        if 'compositionMode' in kwargs:
            self.setCompositionMode(kwargs['compositionMode'])
        if 'hoverable' in kwargs:
            self.opts['hoverable'] = bool(kwargs['hoverable'])
        if 'tip' in kwargs:
            self.opts['tip'] = kwargs['tip']

        data = {'x': x, 'y': y, 'c': c}

        for k, datum in data.items():
            if isinstance(datum, (list, tuple)):
                data[k] = datum = np.array(datum)
            if not isinstance(datum, np.ndarray) or datum.ndim > 1:
                raise Exception("Plot data must be 1D ndarray.")
            if datum.dtype.kind == 'c':
                raise Exception("Can not plot complex data types.")

        profiler("data checks")

        self.xData = np.append(self.xData, data['x'])
        self.yData = np.append(self.yData, data['y'])
        self.cData = np.append(self.cData, data['c'])

        if len(self.xData) > 4:
            if not self.delaunay:
                points = np.column_stack([self.xData, self.yData])
                self.delaunay = Delaunay(points, incremental=True)
            else:
                points = np.column_stack([data['x'], data['y']])
                self.delaunay.add_points(points)

            self.simplices = self.delaunay.simplices

        self.invalidateBounds()
        self.prepareGeometryChange()
        self.informViewBoundsChanged()

        profiler('copy')

        self._mouseShape = None

        if 'name' in kwargs:
            self.opts['name'] = kwargs['name']
        if 'pen' in kwargs:
            self.setPen(kwargs['pen'])
        if 'antialias' in kwargs:
            self.opts['antialias'] = kwargs['antialias']
        ## if symbol pen/brush are given with no previously set symbol, then assume symbol is 'o'
        if 'symbol' not in kwargs and ('symbolPen' in kwargs or 'symbolBrush' in kwargs or 'symbolSize' in kwargs):
            if self.opts['symbol'] is None:
                kwargs['symbol'] = 'o'
        if autoLevels is None:
            if 'levels' in kwargs:
                autoLevels = False
            else:
                autoLevels = True
        if autoLevels or self.levels is None:
            mn, mx = self.cData.min(), self.cData.max()
            # mn and mx can still be NaN if the data is all-NaN
            if mn == mx or np.isnan(mn) or np.isnan(mx):
                mn = 0
                mx = 255
            kwargs['levels'] = [mn, mx]
        if 'levels' in kwargs:
            self.setLevels(kwargs['levels'])

        self.scatter.addPoints(x=x, y=y, **kwargs)

        profiler('set')
        self.update()
        profiler('update')
        self.sigDataChanged.emit()
        profiler('emit')

    def implements(self, interface=None):
        ints = ['plotData']
        if interface is None:
            return ints
        return interface in ints

    def name(self):
        """ Returns the name that represents this item in the legend. """
        return self.opts.get('name', None)

    def setAlpha(self, alpha, auto):
        if self.opts['alphaHint'] == alpha and self.opts['alphaMode'] == auto:
            return
        self.opts['alphaHint'] = alpha
        self.opts['alphaMode'] = auto
        self.setOpacity(alpha)

    def setPen(self, *args, **kargs):
        """
        Sets the pen used to draw lines between points.
        The argument can be a :class:`QtGui.QPen` or any combination of arguments accepted by
        :func:`pyqtgraph.mkPen() <pyqtgraph.mkPen>`.
        """
        pen = fn.mkPen(*args, **kargs)
        self.opts['pen'] = pen
        #self.curve.setPen(pen)
        #for c in self.curves:
        #c.setPen(pen)
        #self.update()
        self.updateItems(styleUpdate=True)

    def setSymbol(self, symbol):
        """ `symbol` can be any string recognized by
        :class:`ScatterPlotItem <pyqtgraph.ScatterPlotItem>` or a list that
        specifies a symbol for each point.
        """
        if self.opts['symbol'] == symbol:
            return
        self.opts['symbol'] = symbol
        #self.scatter.setSymbol(symbol)
        self.updateItems(styleUpdate=True)

    def setSymbolPen(self, *args, **kargs):
        """
        Sets the :class:`QtGui.QPen` used to draw symbol outlines.
        See :func:`mkPen() <pyqtgraph.functions.mkPen>`) for arguments.
        """
        pen = fn.mkPen(*args, **kargs)
        if self.opts['symbolPen'] == pen:
            return
        self.opts['symbolPen'] = pen
        #self.scatter.setSymbolPen(pen)
        self.updateItems(styleUpdate=True)

    def setSymbolBrush(self, *args, **kargs):
        """
        Sets the :class:`QtGui.QBrush` used to fill symbols.
        See :func:`mkBrush() <pyqtgraph.functions.mkBrush>`) for arguments.
        """
        brush = fn.mkBrush(*args, **kargs)
        if self.opts['symbolBrush'] == brush:
            return
        self.opts['symbolBrush'] = brush
        #self.scatter.setSymbolBrush(brush)
        self.updateItems(styleUpdate=True)

    def setSymbolSize(self, size):
        """
        Sets the symbol size.
        """
        if self.opts['symbolSize'] == size:
            return
        self.opts['symbolSize'] = size
        #self.scatter.setSymbolSize(symbolSize)
        self.updateItems(styleUpdate=True)

    @debug.warnOnException  ## raising an exception here causes crash
    def paint(self, p, opt, widget):
        profiler = debug.Profiler()
        if self.xData is None or len(self.xData) == 0:
            return

        if getConfigOption('enableExperimental'):
            if HAVE_OPENGL and isinstance(widget, QtWidgets.QOpenGLWidget):
                self.paintGL(p, opt, widget)
                return
        raise RuntimeError('OpenGL and experimental mode must be enabled to use CloudItem')

    def paintGL(self, p, opt, widget):
        profiler = debug.Profiler()
        cData = self.cData
        levels = self.levels

        # Request a lookup table if this image has only one channel
        self.lut = self._ensure_proper_substrate(self.lut, np)
        if isinstance(self.lut, Callable):
            lut = self._ensure_proper_substrate(self.lut(self.cData), np)
        else:
            lut = self.lut

        if lut is None:
            lut = np.tile(range(256), [3, 1]).T

        if lut is not None:
            scale = lut.shape[0]
        else:
            scale = 255.

        if levels is not None:
            minVal, maxVal = levels
        else:
            minVal, maxVal = cData.min(), cData.max()

        if minVal == maxVal:
            maxVal = np.nextafter(maxVal, 2*maxVal)
        rng = maxVal-minVal
        rng = 1 if rng == 0 else rng

        cData = rescaleData(cData, scale / rng, minVal, dtype=cData.dtype)

        profiler('apply levels')

        cData = applyLookupTable(cData, lut).astype(np.ubyte)

        profiler('apply lut')


        p.beginNativePainting()
        import OpenGL.GL as gl

        if sys.platform == 'win32':
            # If Qt is built to dynamically load OpenGL, then the projection and
            # modelview matrices are not setup.
            # https://doc.qt.io/qt-6/windows-graphics.html
            # https://code.woboq.org/qt6/qtbase/src/opengl/qopenglpaintengine.cpp.html
            # Technically, we could enable it for all platforms, but for now, just
            # enable it where it is required, i.e. Windows
            gl.glMatrixMode(gl.GL_PROJECTION)
            gl.glLoadIdentity()
            gl.glOrtho(0, widget.width(), widget.height(), 0, -999999, 999999)
            gl.glMatrixMode(gl.GL_MODELVIEW)
            mat = QtGui.QMatrix4x4(self.sceneTransform())
            gl.glLoadMatrixf(np.array(mat.data(), dtype=np.float32))

        ## set clipping viewport
        view = self.getViewBox()
        if view is not None:
            rect = view.mapRectToItem(self, view.boundingRect())
            #gl.glViewport(int(rect.x()), int(rect.y()), int(rect.width()), int(rect.height()))

            #gl.glTranslate(-rect.x(), -rect.y(), 0)

            gl.glEnable(gl.GL_STENCIL_TEST)
            gl.glColorMask(gl.GL_FALSE, gl.GL_FALSE, gl.GL_FALSE, gl.GL_FALSE) # disable drawing to frame buffer
            gl.glDepthMask(gl.GL_FALSE)  # disable drawing to depth buffer
            gl.glStencilFunc(gl.GL_NEVER, 1, 0xFF)
            gl.glStencilOp(gl.GL_REPLACE, gl.GL_KEEP, gl.GL_KEEP)

            ## draw stencil pattern
            gl.glStencilMask(0xFF)
            gl.glClear(gl.GL_STENCIL_BUFFER_BIT)
            gl.glBegin(gl.GL_TRIANGLES)
            gl.glVertex2f(rect.x(), rect.y())
            gl.glVertex2f(rect.x()+rect.width(), rect.y())
            gl.glVertex2f(rect.x(), rect.y()+rect.height())
            gl.glVertex2f(rect.x()+rect.width(), rect.y()+rect.height())
            gl.glVertex2f(rect.x()+rect.width(), rect.y())
            gl.glVertex2f(rect.x(), rect.y()+rect.height())
            gl.glEnd()

            gl.glColorMask(gl.GL_TRUE, gl.GL_TRUE, gl.GL_TRUE, gl.GL_TRUE)
            gl.glDepthMask(gl.GL_TRUE)
            gl.glStencilMask(0x00)
            gl.glStencilFunc(gl.GL_EQUAL, 1, 0xFF)

        try:
            # c = self.cData - min(self.cData)
            # c = np.nan_to_num(c / max(c) * (len(self._lut)-1))
            # lut_c = lut[cData.astype(np.int_)]
            gl.glBegin(gl.GL_TRIANGLES)
            for i, (i, j, k) in enumerate(self.simplices):
                ci, cj, ck = cData[[i, j, k]]
                gl.glColor3ub(*ci)
                gl.glVertex3f(self.xData[i], self.yData[i], 0)
                gl.glColor3ub(*cj)
                gl.glVertex3f(self.xData[j], self.yData[j], 0)
                gl.glColor3ub(*ck)
                gl.glVertex3f(self.xData[k], self.yData[k], 0)
            gl.glEnd()
        finally:
            p.endNativePainting()

    def clear(self):
        self.xData = np.array([])  ## raw values
        self.yData = np.array([])
        self.cData = np.array([])
        self.lut = None
        self._mouseShape = None
        self._mouseBounds = None
        self._boundsCache = [None, None]
        self._boundingRect = None
        self.fragmentAtlas = SymbolAtlas()
        self.delaunay = None
        self.simplices = []
        self._effectiveLut = None

    def boundingRect(self):
        if self._boundingRect is None:
            (xmn, xmx) = self.dataBounds(ax=0)
            if xmn is None or xmx is None:
                return QtCore.QRectF()
            (ymn, ymx) = self.dataBounds(ax=1)
            if ymn is None or ymx is None:
                return QtCore.QRectF()

            px = py = 0.0
            pxPad = self.pixelPadding()
            if pxPad > 0:
                # determine length of pixel in local x, y directions
                px, py = self.pixelVectors()
                try:
                    px = 0 if px is None else px.length()
                except OverflowError:
                    px = 0
                try:
                    py = 0 if py is None else py.length()
                except OverflowError:
                    py = 0

                # return bounds expanded by pixel size
                px *= pxPad
                py *= pxPad
            #px += self._maxSpotWidth * 0.5
            #py += self._maxSpotWidth * 0.5
            self._boundingRect = QtCore.QRectF(xmn-px, ymn-py, (2*px)+xmx-xmn, (2*py)+ymx-ymn)

        return self._boundingRect

    def dataBounds(self, ax, frac=1.0, orthoRange=None):
        ## Need this to run as fast as possible.
        ## check cache first:
        cache = self._boundsCache[ax]
        if cache is not None and cache[0] == (frac, orthoRange):
            return cache[1]

        if self.xData is None or len(self.xData) == 0:
            return (None, None)

        if ax == 0:
            d = self.xData
            d2 = self.yData
        elif ax == 1:
            d = self.yData
            d2 = self.xData
        else:
            raise ValueError("Invalid axis value")

        ## If an orthogonal range is specified, mask the data now
        if orthoRange is not None:
            mask = (d2 >= orthoRange[0]) * (d2 <= orthoRange[1])
            d = d[mask]
            #d2 = d2[mask]

        if len(d) == 0:
            return (None, None)

        ## Get min/max (or percentiles) of the requested data range
        if frac >= 1.0:
            # include complete data range
            # first try faster nanmin/max function, then cut out infs if needed.
            with warnings.catch_warnings():
                # All-NaN data is acceptable; Explicit numpy warning is not needed.
                warnings.simplefilter("ignore")
                b = (np.nanmin(d), np.nanmax(d))
            if math.isinf(b[0]) or math.isinf(b[1]):
                mask = np.isfinite(d)
                d = d[mask]
                if len(d) == 0:
                    return (None, None)
                b = (d.min(), d.max())

        elif frac <= 0.0:
            raise Exception("Value for parameter 'frac' must be > 0. (got %s)" % str(frac))
        else:
            # include a percentile of data range
            mask = np.isfinite(d)
            d = d[mask]
            if len(d) == 0:
                return (None, None)
            b = np.percentile(d, [50 * (1 - frac), 50 * (1 + frac)])

        ## Add pen width only if it is non-cosmetic.
        # pen = self.opts['pen']
        # spen = self.opts['shadowPen']
        # if not pen.isCosmetic():
        #     b = (b[0] - pen.widthF()*0.7072, b[1] + pen.widthF()*0.7072)
        # if spen is not None and not spen.isCosmetic() and spen.style() != QtCore.Qt.PenStyle.NoPen:
        #     b = (b[0] - spen.widthF()*0.7072, b[1] + spen.widthF()*0.7072)

        self._boundsCache[ax] = [(frac, orthoRange), b]
        return b

    def pixelPadding(self):
        # pen = self.opts['pen']
        # spen = self.opts['shadowPen']
        w = 0
        # if pen.isCosmetic():
        #     w += pen.widthF()*0.7072
        # if spen is not None and spen.isCosmetic() and spen.style() != QtCore.Qt.PenStyle.NoPen:
        #     w = max(w, spen.widthF()*0.7072)
        # if self.clickable:
        #     w = max(w, self.opts['mouseWidth']//2 + 1)
        return w

    def invalidateBounds(self):
        self._boundingRect = None
        self._boundsCache = [None, None]

    def scatterClicked(self, plt, points, ev):
        self.sigClicked.emit(self, ev)
        self.sigPointsClicked.emit(self, points, ev)

    def scatterHovered(self, plt, points, ev):
        self.sigPointsHovered.emit(self, points, ev)

    def setLookupTable(self, lut, update=True):
        """
        Sets lookup table ``lut`` to use for false color display of a monochrome image. See :func:`makeARGB <pyqtgraph.makeARGB>` for more
        information on how this is used. Optionally, `lut` can be a callable that accepts the current image as an
        argument and returns the lookup table to use.

        Ordinarily, this table is supplied by a :class:`~pyqtgraph.HistogramLUTItem`,
        :class:`~pyqtgraph.GradientEditorItem` or :class:`~pyqtgraph.ColorBarItem`.

        Setting ``update = False`` avoids an immediate image update.
        """
        if lut is not self.lut:
            lut = self._ensure_proper_substrate(lut, np)
            self.lut = lut
            self._effectiveLut = None
            if update:
                self.update()

    def setLevels(self, levels, update=True):
        """
        Sets image scaling levels.
        See :func:`makeARGB <pyqtgraph.makeARGB>` for more details on how levels are applied.

        Parameters
        ----------
            levels: list_like
                - ``[blackLevel, whiteLevel]``
                  sets black and white levels for monochrome data and can be used with a lookup table.
                - ``[[minR, maxR], [minG, maxG], [minB, maxB]]``
                  sets individual scaling for RGB values. Not compatible with lookup tables.
            update: bool, optional
                Controls if image immediately updates to reflect the new levels.
        """

        if levels is not None:
            levels = np.asarray(levels)
        self.levels = levels
        self._effectiveLut = None
        if update:
            self.update()

    @staticmethod
    def _ensure_proper_substrate(data, substrate):
        if data is None or isinstance(data, Callable) or isinstance(data, substrate.ndarray):
            return data
        cupy = getCupy()
        if substrate == cupy and not isinstance(data, cupy.ndarray):
            data = cupy.asarray(data)
        elif substrate == np:
            if cupy is not None and isinstance(data, cupy.ndarray):
                data = data.get()
            else:
                data = np.asarray(data)
        return data

    def getHistogram(self, bins='auto', step='auto', perChannel=False, targetImageSize=200,
                     targetHistogramSize=500, **kwds):
        """
        Returns `x` and `y` arrays containing the histogram values for the current image.
        For an explanation of the return format, see :func:`numpy.histogram()`.

        The `step` argument causes pixels to be skipped when computing the histogram to save time.
        If `step` is 'auto', then a step is chosen such that the analyzed data has
        dimensions approximating `targetImageSize` for each axis.

        The `bins` argument and any extra keyword arguments are passed to
        :func:`self.xp.histogram()`. If `bins` is `auto`, a bin number is automatically
        chosen based on the image characteristics:

          * Integer images will have approximately `targetHistogramSize` bins,
            with each bin having an integer width.
          * All other types will have `targetHistogramSize` bins.

        If `perChannel` is `True`, then a histogram is computed for each channel,
        and the output is a list of the results.
        """
        # This method is also used when automatically computing levels.
        if self.cData is None or len(self.cData) == 0:
            return None, None
        if step == 'auto':
            step = max(1, int(np.ceil(len(self.cData) / targetImageSize)))
        stepData = self.cData[::step]

        if isinstance(bins, str) and bins == 'auto':
            mn = np.nanmin(stepData).item()
            mx = np.nanmax(stepData).item()
            if mx == mn:
                # degenerate image, arange will fail
                mx += 1
            if np.isnan(mn) or np.isnan(mx):
                # the data are all-nan
                return None, None
            if stepData.dtype.kind in "ui":
                # For integer data, we select the bins carefully to avoid aliasing
                step = int(np.ceil((mx - mn) / targetHistogramSize))
                bins = []
                if step > 0.0:
                    bins = np.arange(mn, mx + 1.01 * step, step, dtype=int)
            else:
                # for float data, let numpy select the bins.
                bins = np.linspace(mn, mx, targetHistogramSize)

            if len(bins) == 0:
                bins = np.asarray((mn, mx))

        kwds['bins'] = bins

        cp = getCupy()
        if perChannel:
            hist = []
            for i in range(stepData.shape[-1]):
                stepChan = stepData[..., i]
                stepChan = stepChan[np.isfinite(stepChan)]
                h = np.histogram(stepChan, **kwds)
                if cp:
                    hist.append((cp.asnumpy(h[1][:-1]), cp.asnumpy(h[0])))
                else:
                    hist.append((h[1][:-1], h[0]))
            return hist
        else:
            stepData = stepData[np.isfinite(stepData)]
            hist = np.histogram(stepData, **kwds)
            if cp:
                return cp.asnumpy(hist[1][:-1]), cp.asnumpy(hist[0])
            else:
                return hist[1][:-1], hist[0]
