from functools import lru_cache
from typing import Tuple

import numpy as np
from loguru import logger
from pyqtgraph import HistogramLUTWidget, mkBrush, mkPen, PlotItem, PlotWidget, TableWidget
from qtpy.QtWidgets import QWidget, QHBoxLayout

from tsuchinoko.graphics_items.clouditem import CloudItem
from tsuchinoko.graphics_items.indicatoritem import BetterCurveArrow
from tsuchinoko.graphics_items.mixins import ClickRequesterPlot, ClickRequester
from tsuchinoko.graphs import Graph, Location


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
    def __init__(self, data_key, name: str = None, accumulates: bool = False):
        self.data_key = data_key
        self.accumulates = accumulates
        super(Image, self).__init__(name=name or data_key)

    def make_widget(self):
        graph = PlotItem()
        self.widget = ImageViewBlend(view=graph)
        graph.vb.invertY(False)  # imageview forces invertY; this resets it
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
        graph = ClickRequesterPlot()
        # scatter = ScatterPlotItem(name='scatter', x=[0], y=[0], size=10, pen=mkPen(None), brush=mkBrush(255, 255, 255, 120))
        self.cloud = CloudItem(name='scatter', size=10)
        histlut = HistogramLUTWidget()
        histlut.setImageItem(self.cloud)

        self.widget = QWidget()
        self.widget.setLayout(QHBoxLayout())

        self.widget.layout().addWidget(graph)
        self.widget.layout().addWidget(histlut)

        graph.addItem(self.cloud)

        # Hard-coded to show max
        self.max_arrow = BetterCurveArrow(self.cloud.scatter, brush=mkBrush('r'))
        self.last_arrow = BetterCurveArrow(self.cloud.scatter, brush=mkBrush('w'))
        # text = TextItem()

        return self.widget

    def update(self, data, update_slice: slice):
        # require_clear = False

        with data.r_lock():
            v = data[self.data_key].copy()

            x, y = zip(*data.positions)

        lengths = len(v), len(x), len(y)

        if not np.all(np.array(lengths) == min(lengths)):
            logger.warning(f'Ragged arrays passed to cloud item with lengths (v, x, y): {lengths}')
            x = x[:min(lengths)]
            y = y[:min(lengths)]
            v = v[:min(lengths)]

        if not len(x):
            return

        # c = [255 * i / len(x) for i in range(len(x))]
        max_index = np.argmax(v)
        last_data_size = min(update_slice.start, len(self.cloud.cData))

        if last_data_size == 0:
            action = self.cloud.setData
        elif not self.accumulates:
            action = self.cloud.updateData
        else:
            action = self.cloud.extendData
            x = x[last_data_size + 1:]
            y = y[last_data_size + 1:]
            v = v[last_data_size + 1:]

        action(x=x,
               y=y,
               c=v,
               data=v,
               # size=5,
               hoverable=True,
               # hoverSymbol='s',
               # hoverSize=6,
               hoverPen=mkPen('b', width=2),
               # hoverBrush=mkBrush('g'),
               )
        # scatter.setData(
        #     [{'pos': (xi, yi),
        #       'size': (vi - min(v)) / (max(v) - min(v)) * 20 + 2 if max(v) != min(v) else 20,
        #       'brush': mkBrush(color=mkColor(255, 255, 255)) if i == len(x) - 1 else mkBrush(
        #           color=mkColor(255 - c, c, 0)),
        #       'symbol': '+' if i == len(x) - 1 else 'o'}
        #      for i, (xi, yi, vi, c) in enumerate(zip(x, y, v, c))])

        self.max_arrow.setIndex(max_index)
        self.last_arrow.setIndex(len(self.cloud.cData) - 1)
        # text.setText(f'Max: {v[max_index]:.2f} ({x[max_index]:.2f}, {y[max_index]:.2f})')
        # text.setPos(x[max_index], y[max_index])


class Plot(Graph):
    def __init__(self, data_key, name: str = None, accumulates: bool = False, widget_kwargs=None):
        self.data_key = data_key
        self.accumulates = accumulates
        self.widget_kwargs = widget_kwargs or dict()
        super(Plot, self).__init__(name=name or data_key)

    def make_widget(self):
        self.widget = PlotWidget(**self.widget_kwargs)
        return self.widget

    def update(self, data, update_slice: slice):
        with data.r_lock():
            v = data[self.data_key].copy()
        if self.accumulates:
            self.widget.plot(np.asarray(v), clear=True)
        else:
            self.widget.plot(np.asarray(v), clear=True)


class GPCamVariance(Cloud):
    def __init__(self):
        super(GPCamVariance, self).__init__(data_key='variances', name='Variance')

    def compute(self, data, engine):
        pass  # This is free from gpCAM


class GPCamScore(Cloud):
    def __init__(self):
        super(GPCamScore, self).__init__(data_key='scores', name='Score')

    def compute(self, data, engine):
        pass  # This is free from gpCAM


class GPCamPosteriorCovariance(Image):
    def __init__(self, shape=(50, 50)):
        self.shape = shape
        super(GPCamPosteriorCovariance, self).__init__(data_key='Posterior Covariance')

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
