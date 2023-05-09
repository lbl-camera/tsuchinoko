from dataclasses import InitVar, dataclass, field
from functools import lru_cache, partial
from itertools import count
from typing import Tuple, ClassVar, List

import numpy as np
from loguru import logger
from pyqtgraph import PlotItem, PlotWidget, TableWidget, mkColor, intColor, PlotDataItem, mkPen, mkBrush, colormap, \
    ScatterPlotItem
from qtpy.QtWidgets import QFormLayout, QWidget, QComboBox, QLabel, QVBoxLayout
from qtpy.QtCore import Qt, QSignalBlocker, Signal, QRectF

from tsuchinoko.graphics_items.mixins import ClickRequester, DomainROI
from tsuchinoko.graphs import Graph, Location, graph_signal_relay
from tsuchinoko.widgets.displays import Configuration
from tsuchinoko.widgets.graph_widgets import CloudWidget
import sklearn

from tsuchinoko.widgets.simple import DoubleSlider, ValueDoubleSlider


@dataclass(eq=False)
class Scatter(Graph):
    widget_class = PlotWidget

    def __init__(self, x_key: str, y_key: str, n_clusters: int, kmeans_kwargs: dict = None, name: str = 'Scatter'):
        self.x_key = x_key
        self.y_key = y_key
        self.n_clusters = n_clusters
        self.kmeans_kwargs = kmeans_kwargs
        super().__init__(name)

    def update(self, widget, data, update_slice: slice):
        with data.r_lock():
            x = data[self.x_key].copy()
            y = data[self.y_key].copy()

        widget.clear()

        kmeans = sklearn.cluster.KMeans(self.n_clusters, **self.kmeans_kwargs or {})
        kmeans.fit(np.dstack((x, y)))
        labels = kmeans.labels_
        for i in range(self.n_clusters):
            xi = x[labels == i]
            yi = y[labels == i]
            widget.addItem(ScatterPlotItem(x=xi, y=yi, brush=mkBrush(color=intColor(i, values=self.n_clusters))))


@dataclass(eq=False)
class Table(Graph):
    widget_class = TableWidget
    widget_kwargs = {'sortable': False}
    data_keys: Tuple[str] = tuple()
    name = 'Table'

    def update(self, widget, data, update_slice: slice):
        # data = data[update_slice]

        with data.r_lock():
            x = data.positions.copy()
            v = data.variances.copy()
            y = data.scores.copy()

            extra_fields = {data_key: data[data_key].copy() for data_key in self.data_keys}

        lengths = len(v), len(x), len(y), *map(len, extra_fields.values())
        min_length = min(lengths)
        if not np.all(np.array(lengths) == min_length):
            logger.warning(f'Ragged arrays passed to cloud item with lengths (v, x, y): {lengths}')
            x = x[:min_length]
            y = y[:min_length]
            v = v[:min_length]
            extra_fields = {k: v[:min_length] for k, v in extra_fields.items()}

        values = np.array([x, y, v, *extra_fields.values()], dtype=object)

        names = ['Position', 'Value', 'Variance'] + list(extra_fields.keys())

        rows = range(update_slice.start, len(x))
        table = [{name: value[i] for name, value in zip(names, values)} for i in rows]

        if update_slice.start == 0:
            widget.setData(table)
        else:
            for row, table_row in zip(rows, table):
                widget.setRow(row, list(table_row.values()))


@dataclass(eq=False)  # TODO: Should this be a dataclass?
class ImageViewBlend(DomainROI, ClickRequester):
    def __init__(self, *args, invert_y=False, **kwargs):
        graph = PlotItem()
        super().__init__(*args, view=graph, **kwargs)
        graph.vb.invertY(invert_y)


@dataclass(eq=False)
class Image(Graph):
    widget_class = ImageViewBlend
    data_key: ClassVar[str] = None
    accumulates: ClassVar[bool] = False
    invert_y = False

    def __post_init__(self):
        if not self.name and self.data_key:
            self.name = self.data_key

    def update(self, widget, data, update_slice: slice):
        with data.r_lock():
            v = data[self.data_key].copy()
        if self.accumulates:
            raise NotImplemented('Accumulation in Image graphs not implemented yet')
        else:
            if getattr(v, 'ndim', None) in [2, 3]:
                bounds = [tuple(Configuration().parameter.child('bounds')[f'axis_{i}_{limit}']
                                for limit in ['min', 'max'])
                          for i in range(2)]
                rect = QRectF(bounds[0][0], bounds[1][0], bounds[0][1]-bounds[0][0], bounds[1][1]-bounds[1][0])
                widget.imageItem.setImage(v, autoLevels=widget.imageItem.image is None, rect=rect)


@dataclass(eq=False)
class Cloud(Graph):
    data_key: ClassVar[str] = None
    accumulates: ClassVar[bool] = True
    widget_class = type('CloudBlend', (CloudWidget, DomainROI), {})
    widget_args: Tuple = tuple()

    def __post_init__(self):
        self.widget_args = (self.data_key, self.accumulates)

    def update(self, widget, data, update_slice: slice):
        widget.update_data(data, update_slice)


class PlotGraphWidget(PlotWidget):
    def __init__(self, *args, label_key=None, **kwargs):
        super().__init__(*args, **kwargs)
        if label_key:
            self.getPlotItem().addLegend()


@dataclass(eq=False)
class Plot(Graph):
    data_key: str = None
    widget_class = PlotGraphWidget
    label_key: InitVar[str] = None
    accumulates: bool = False

    def __post_init__(self, label_key):
        self.widget_kwargs['label_key'] = label_key

    def update(self, widget, data, update_slice: slice):
        with data.r_lock():
            v = data[self.data_key].copy()
        if self.accumulates:
            widget.plot(np.asarray(v), clear=True, label=self.label_key)
        else:
            widget.plot(np.asarray(v), clear=True, label=self.label_key)


@dataclass(eq=False)
class MultiPlot(Plot):
    pen_key:str = None

    @staticmethod
    def get_color(i, count):
        if count < 9:
            color = mkColor(i)
        else:
            color = intColor(i, hues=count, minHue=180, maxHue=300)
        return color

    def colorize(self, widget, data):
        plot_data_items = list(filter(lambda item: isinstance(item, PlotDataItem), widget.getPlotItem().items))
        count = len(plot_data_items)

        for i, item in enumerate(plot_data_items):
            if isinstance(item, PlotDataItem):
                color = self.get_color(i, count)
                item.setPen(color)
                item.setSymbolBrush(color)
                item.setSymbolPen('w')

    def update(self, widget, data: 'Data', update_slice: slice):
        if update_slice.start == 0:
            widget.getPlotItem().clear()

        with data.r_lock():
            v = data[self.data_key].copy()
            labels = data[self.label_key].copy()
            if self.pen_key is not None:
                pens = data[self.pen_key].copy()

        for i, label, plot_data in zip(count(update_slice.start), labels[update_slice], v[update_slice]):
            kwargs = {}
            if self.pen_key is not None:
                kwargs['pen'] = mkPen(pens[i])
            widget.plot(plot_data, name=label, **kwargs)

        if self.pen_key is None:
            self.colorize(widget, data)


@dataclass(eq=False)
class DynamicColorMultiPlot(MultiPlot):
    def __init__(self, color_scalar_key, *args, colormap_name='CET-L17', **kwargs):
        super().__init__(*args, **kwargs)
        self.colormap = colormap.get(colormap_name)
        self.color_scalar_key = color_scalar_key
        self.item_colors = []

    def update(self, widget, data: 'Data', update_slice: slice):
        with data.r_lock():
            c = data[self.color_scalar_key].copy()
        c_min = np.min(c)
        c_max = np.max(c)
        scaled_c = np.interp(c, (c_min, c_max), (0, 1))
        self.item_colors = list(map(self.colormap.map, scaled_c))

        super().update(widget, data, update_slice)

    def get_color(self, i, count):
        return self.item_colors[i]


@dataclass(eq=False)
class Variance(Cloud):
    data_key = 'variances'
    name = 'Variance'

    def compute(self, data, engine):
        pass  # This is free


@dataclass(eq=False)
class Score(Cloud):
    data_key = 'scores'
    name = 'Score'

    def compute(self, data, engine):
        pass  # This is free


@dataclass(eq=False)
class GPCamPosteriorCovariance(Image):
    shape = (50, 50)
    data_key = 'Posterior Covariance'
    invert_y = True

    def compute(self, data, engine: 'GPCamInProcessEngine'):
        with data.r_lock():  # quickly grab positions within lock before passing to optimizer
            positions = np.asarray(data.positions.copy())

        # compute posterior covariance without lock
        result_dict = engine.optimizer.posterior_covariance(positions)

        # assign to data object with lock
        with data.w_lock():
            data.states[self.data_key] = result_dict['S(x)']


@dataclass(eq=False)
class GPCamAcquisitionFunction(Image):
    compute_with = Location.AdaptiveEngine
    shape = (50, 50)
    data_key = 'Acquisition Function'

    def compute(self, data, engine: 'GPCAMInProcessEngine'):
        from tsuchinoko.adaptive.gpCAM_in_process import acquisition_functions  # avoid circular import

        bounds = tuple(tuple(engine.parameters[('bounds', f'axis_{i}_{edge}')]
                             for edge in ['min', 'max'])
                       for i in range(engine.dimensionality))

        grid_positions = image_grid(bounds, self.shape)

        # calculate acquisition function
        acquisition_function_value = engine.optimizer.evaluate_acquisition_function(grid_positions,
                                                                                    acquisition_function=
                                                                                    acquisition_functions[
                                                                                        engine.parameters[
                                                                                            'acquisition_function']])

        try:
            acquisition_function_value = acquisition_function_value.reshape(*self.shape)
        except (ValueError, AttributeError):
            acquisition_function_value = np.array([[0]])

        # assign to data object with lock
        with data.w_lock():
            data.states[self.data_key] = acquisition_function_value


@dataclass(eq=False)
class GPCamPosteriorMean(Image):
    compute_with = Location.AdaptiveEngine
    shape = (50, 50)
    data_key = 'Posterior Mean'

    def compute(self, data, engine: 'GPCAMInProcessEngine'):
        bounds = ((engine.parameters[('bounds', f'axis_{i}_{edge}')]
                   for edge in ['min', 'max'])
                  for i in range(engine.dimensionality))

        grid_positions = image_grid(bounds, self.shape)

        # calculate acquisition function
        posterior_mean_value = engine.optimizer.posterior_mean(grid_positions)['f(x)'].reshape(*self.shape)

        # assign to data object with lock
        with data.w_lock():
            data.states['Posterior Mean'] = posterior_mean_value


@lru_cache(maxsize=10)
def image_grid(bounds, shape):
    return np.asarray(np.meshgrid(*(np.linspace(*bound, num=bins) for bins, bound in zip(shape, bounds)))).T.reshape(-1,
                                                                                                                     2)


class SliceImageWidget(QWidget):
    sigSliceChanged = Signal(int, float)  # axis, value
    sigImageAxesChanged = Signal(int, int)  # x and y image axes

    def __init__(self, dimensions: int, bounds:List[List[float]], invert_y=False):
        super().__init__()
        dim_indices = list(range(1, dimensions + 1))
        self.x_dimension_selector = QComboBox()
        self.y_dimension_selector = QComboBox()
        self.x_dimension_selector.addItems(map(str, dim_indices))
        self.y_dimension_selector.addItems(map(str, dim_indices))
        self.x_dimension_selector.setCurrentIndex(0)
        self.y_dimension_selector.setCurrentIndex(1)
        self.x_dimension_selector.currentIndexChanged.connect(partial(self.dimension_selected, axis='x'))
        self.y_dimension_selector.currentIndexChanged.connect(partial(self.dimension_selected, axis='y'))

        self.image_view = ImageViewBlend(invert_y=invert_y)

        self.dims_layout = QFormLayout()
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.image_view)
        self.layout().addLayout(self.dims_layout)

        self.dims_layout.addRow("X dimension:", self.x_dimension_selector)
        self.dims_layout.addRow("Y dimension:", self.y_dimension_selector)
        self.dimension_sliders = []
        for i in range(dimensions):
            slider = ValueDoubleSlider(Qt.Horizontal)
            slider.sigFloatValueChanged.connect(partial(self.sigSliceChanged.emit, i))
            slider.setMinimum(bounds[i][0])
            slider.setMaximum(bounds[i][1])
            slider.setInterval((bounds[i][1] - bounds[i][0]) / 100)
            self.dimension_sliders.append(slider)
        for i, slider in enumerate(self.dimension_sliders[2:]):
            label = QLabel(f'Dimension {i + 3} slice:')
            self.dims_layout.addRow(label, slider)

    def dimension_selected(self, index, axis):
        dim_indices = set(range(self.x_dimension_selector.count()))
        dim_indices.remove(index)
        if axis == 'x':
            if self.y_dimension_selector.currentIndex() == index:
                blocker = QSignalBlocker(self.y_dimension_selector)
                self.y_dimension_selector.setCurrentIndex(dim_indices.pop())
            else:
                dim_indices.remove(self.y_dimension_selector.currentIndex())
        if axis == 'y':
            if self.x_dimension_selector.currentIndex() == index:
                blocker = QSignalBlocker(self.x_dimension_selector)
                self.x_dimension_selector.setCurrentIndex(dim_indices.pop())
            else:
                dim_indices.remove(self.x_dimension_selector.currentIndex())

        # clear layout
        for i in reversed(range(self.dims_layout.count())):
            self.dims_layout.itemAt(i).widget().setParent(None)

        self.dims_layout.addRow("X dimension:", self.x_dimension_selector)
        self.dims_layout.addRow("Y dimension:", self.y_dimension_selector)
        for i in dim_indices:
            label = QLabel(f'Dimension {i + 1} slice:')
            self.dims_layout.addRow(label, self.dimension_sliders[i])

        self.sigImageAxesChanged.emit(self.x_dimension_selector.currentIndex(), self.y_dimension_selector.currentIndex())


@dataclass(eq=False)
class SliceImageGraph(Graph):
    dimensions: int = 3
    widget_class = SliceImageWidget
    bounds: List[List[float]] = field(default=None)
    data_key: ClassVar[str] = None
    invert_y:bool = False
    accumulates: ClassVar[bool] = False

    def __post_init__(self):
        self.dimension_sliders = []
        self.slices = [dim[0] for i, dim in enumerate(self.bounds)]
        self.image_axes = {'x': 0, 'y': 1}
        self.widget_kwargs['dimensions'] = self.dimensions
        self.widget_kwargs['bounds'] = self.bounds
        if not self.name and self.data_key:
            self.name = self.data_key

    def make_widget(self):
        widget = super().make_widget()
        widget.sigSliceChanged.connect(self.set_slice)
        widget.sigImageAxesChanged.connect(self.set_image_axes)
        return widget

    def set_slice(self, i, v):
        self.slices[i] = v
        graph_signal_relay.sigPush.emit(self)

    def set_image_axes(self, x, y):
        self.image_axes['x'] = x
        self.image_axes['y'] = y
        graph_signal_relay.sigPush.emit(self)

    def update(self, widget, data: 'Data', *args):
        Image.update(self, widget.image_view, data, *args)


@lru_cache(maxsize=10)
def slice_grid(bounds, shape, image_axes, slices):
    ticks = [*(np.linspace(*bounds[i],
                           num=shape[0] if image_axes[0] == i else shape[1])
               if i in image_axes else
               slices[i] for i in range(len(bounds)))]
    return np.asarray(np.meshgrid(*ticks)).T.reshape(-1, len(bounds))


class HighDimensionalityGPCamPosteriorMean(SliceImageGraph):
    compute_with = Location.AdaptiveEngine
    shape = (50, 50)
    data_key = 'Posterior Mean'

    def compute(self, data, engine: 'GPCAMInProcessEngine'):
        bounds = [tuple([engine.parameters[('bounds', f'axis_{i}_{edge}')]
                   for edge in ['min', 'max']])
                  for i in range(engine.dimensionality)]

        image_axes = list(self.image_axes.values())
        for i, value in enumerate(self.slices):
            if i not in image_axes:
                bounds[i] = (value, value)

        # TODO: calculate positions based on axis order and slices
        grid_positions = slice_grid(tuple(bounds), self.shape, tuple(image_axes), tuple(self.slices))

        # calculate acquisition function
        posterior_mean_value = engine.optimizer.posterior_mean(grid_positions)['f(x)'].reshape(*self.shape)

        # transpose if necessary
        if self.image_axes['x'] > self.image_axes['y']:
            posterior_mean_value = posterior_mean_value.T

        # assign to data object with lock
        with data.w_lock():
            data.states['Posterior Mean'] = posterior_mean_value


if __name__ == '__main__':
    from pyqtgraph import mkQApp

    app = mkQApp()
    graph = SliceImageGraph(5, [[0, 1]] * 5, data_key='blah')
    graph.make_widget()
    widget = graph.widget
    widget.show()

    app.exec_()
