from functools import lru_cache
from typing import Tuple

import numpy as np
from loguru import logger
from pyqtgraph import PlotItem, PlotWidget, TableWidget, mkColor, intColor, PlotDataItem, mkPen

from tsuchinoko.graphics_items.mixins import ClickRequester
from tsuchinoko.graphs import Graph, Location
from tsuchinoko.widgets.graph_widgets import CloudWidget


class Table(Graph):
    def __init__(self, data_keys: Tuple[str] = None, name: str = 'Table'):
        super(Table, self).__init__(name)
        self.data_keys = data_keys or tuple()

    def make_widget(self):
        self.widget = TableWidget(sortable=False)
        return self.widget

    def update(self, data, update_slice: slice):
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

        values = np.array([x, y, v, *extra_fields.values()])

        names = ['Position', 'Value', 'Variance'] + list(extra_fields.keys())

        rows = range(update_slice.start, len(x))
        table = [{name: value[i] for name, value in zip(names, values)} for i in rows]

        if update_slice.start == 0:
            self.widget.setData(table)
        else:
            for row, table_row in zip(rows, table):
                self.widget.setRow(row, list(table_row.values()))


class ImageViewBlend(ClickRequester):
    pass


class Image(Graph):
    def __init__(self,
                 data_key, name: str = None,
                 accumulates: bool = False,
                 invert_y=False,
                 widget_kwargs: dict = None):
        self.data_key = data_key
        self.accumulates = accumulates
        self.widget_kwargs = widget_kwargs or dict()
        self.invert_y = invert_y
        super(Image, self).__init__(name=name or data_key)

    def make_widget(self):
        graph = PlotItem()
        self.widget = ImageViewBlend(view=graph, **self.widget_kwargs)
        graph.vb.invertY(self.invert_y)  # imageview forces invertY; this resets it
        return self.widget

    def update(self, data, update_slice: slice):
        with data.r_lock():
            v = data[self.data_key].copy()
        if self.accumulates:
            raise NotImplemented('Accumulation in Image graphs not implemented yet')
        else:
            if getattr(v, 'ndim', None) in [2, 3]:
                self.widget.imageItem.setImage(v, autoLevels=self.widget.imageItem.image is None)


class Cloud(Graph):
    def __init__(self, data_key, name: str = None, accumulates: bool = True):
        self.data_key = data_key
        self.accumulates = accumulates

        super(Cloud, self).__init__(name=name)

    def make_widget(self):
        self.widget = CloudWidget(self.data_key, accumulates=self.accumulates)
        return self.widget

    def update(self, data, update_slice: slice):
        self.widget.update_data(data, update_slice)


class Plot(Graph):
    def __init__(self, data_key, name: str = None, label_key=None, accumulates: bool = False, widget_kwargs=None):
        self.data_key = data_key
        self.accumulates = accumulates
        self.widget_kwargs = widget_kwargs or dict()
        self.label_key = label_key
        super(Plot, self).__init__(name=name or data_key)

    def make_widget(self):
        self.widget = PlotWidget(**self.widget_kwargs)
        if self.label_key:
            self.widget.getPlotItem().addLegend()
        return self.widget

    def update(self, data, update_slice: slice):
        with data.r_lock():
            v = data[self.data_key].copy()
        if self.accumulates:
            self.widget.plot(np.asarray(v), clear=True, label=self.label_key)
        else:
            self.widget.plot(np.asarray(v), clear=True, label=self.label_key)
            

class MultiPlot(Plot):
    def __init__(self, *args, pen_key=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pen_key = pen_key

    @staticmethod
    def get_color(i, count):
        if count < 9:
            color = mkColor(i)
        else:
            color = intColor(i, hues=count, minHue=180, maxHue=300)
        return color

    def colorize(self, data):
        plot_data_items = list(filter(lambda item: isinstance(item, PlotDataItem), self.widget.getPlotItem().items))
        count = len(plot_data_items)

        for i, item in enumerate(plot_data_items):
            if isinstance(item, PlotDataItem):
                color = self.get_color(i, count)
                item.setPen(color)
                item.setSymbolBrush(color)
                item.setSymbolPen('w')

    def update(self, data: 'Data', update_slice:slice):
        if update_slice.start == 0:
            self.widget.getPlotItem().clear()

        with data.r_lock():
            v = data[self.data_key].copy()
            labels = data[self.label_key].copy()
            if self.pen_key is not None:
                pens = data[self.pen_key].copy()

        for i, label, plot_data in zip(count(update_slice.start), labels[update_slice], v[update_slice]):
            kwargs = {}
            if self.pen_key is not None:
                kwargs['pen'] = mkPen(pens[i])
            self.widget.plot(plot_data, name=label, **kwargs)

        if self.pen_key is None:
            self.colorize(data)


class DynamicColorMultiPlot(MultiPlot):
    def __init__(self, color_scalar_key, *args, colormap_name='CET-L17', **kwargs):
        super().__init__(*args, **kwargs)
        self.colormap = colormap.get(colormap_name)
        self.color_scalar_key = color_scalar_key
        self.item_colors = []

    def update(self, data: 'Data', update_slice:slice):
        with data.r_lock():
            c = data[self.color_scalar_key].copy()
        c_min = np.min(c)
        c_max = np.max(c)
        scaled_c = np.interp(c, (c_min, c_max), (0, 1))
        self.item_colors = list(map(self.colormap.map, scaled_c))

        super().update(data, update_slice)

    def get_color(self, i, count):
        return self.item_colors[i]


class Variance(Cloud):
    def __init__(self):
        super(Variance, self).__init__(data_key='variances', name='Variance')

    def compute(self, data, engine):
        pass  # This is free


class Score(Cloud):
    def __init__(self):
        super(Score, self).__init__(data_key='scores', name='Score')

    def compute(self, data, engine):
        pass  # This is free


class GPCamPosteriorCovariance(Image):
    def __init__(self, shape=(50, 50)):
        self.shape = shape
        super(GPCamPosteriorCovariance, self).__init__(data_key='Posterior Covariance', invert_y=True)

    def compute(self, data, engine: 'GPCamInProcessEngine'):
        with data.r_lock():  # quickly grab positions within lock before passing to optimizer
            positions = np.asarray(data.positions.copy())

        # compute posterior covariance without lock
        result_dict = engine.optimizer.posterior_covariance(positions)

        # assign to data object with lock
        with data.w_lock():
            data.states[self.data_key] = result_dict['S(x)']


class GPCamAcquisitionFunction(Image):
    compute_with = Location.AdaptiveEngine

    def __init__(self, shape=(50, 50)):
        self.shape = shape
        super(GPCamAcquisitionFunction, self).__init__(data_key='Acquisition Function')

    def compute(self, data, engine: 'GPCAMInProcessEngine'):
        from tsuchinoko.adaptive.gpCAM_in_process import acquisition_functions  # avoid circular import

        bounds = tuple(tuple(engine.parameters[('bounds', f'axis_{i}_{edge}')]
                             for edge in ['min', 'max'])
                       for i in range(engine.dimensionality))

        grid_positions = image_grid(bounds, self.shape)

        # calculate acquisition function
        acquisition_function_value = engine.optimizer.evaluate_acquisition_function(grid_positions,
                                                                                    acquisition_function=acquisition_functions[engine.parameters['acquisition_function']])

        try:
            acquisition_function_value = acquisition_function_value.reshape(*self.shape)
        except (ValueError, AttributeError):
            acquisition_function_value = np.array([[0]])

        # assign to data object with lock
        with data.w_lock():
            data.states[self.data_key] = acquisition_function_value


class GPCamPosteriorMean(Image):
    compute_with = Location.AdaptiveEngine

    def __init__(self, shape=(50, 50)):
        self.shape = shape
        super(GPCamPosteriorMean, self).__init__(data_key='Posterior Mean')

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
    return np.asarray(np.meshgrid(*(np.linspace(*bound, num=bins) for bins, bound in zip(shape, bounds)))).T.reshape(-1, 2)
