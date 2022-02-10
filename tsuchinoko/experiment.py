import time
from functools import partial

from bluesky.plan_stubs import checkpoint, stage
from bluesky.preprocessors import run_decorator
import numpy as np
from gpcam.autonomous_experimenter import AutonomousExperimenterGP
import pyqtgraph as pg

from tsuchinoko.utils.threads import invoke_in_main_thread
from tsuchinoko.widgets.displays import RunEngineControls, GraphManager


class Experiment():
    def __init__(self, graph_manager: GraphManager):
        super(Experiment, self).__init__()

        self._graph_items = dict()
        self.graph_manager = graph_manager

    def plan(self):
        raise NotImplementedError

    def update_graphs(self, *args, **kwargs):
        raise NotImplementedError

    def init_graph(self, name, indicator=None):
        if name not in self.graph_manager.graphs:
            graph = pg.PlotWidget()
            scatter = pg.ScatterPlotItem(x=[0], y=[0], size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 120))
            self._graph_items[name]={'scatter': scatter}
            graph.addItem(scatter)
            if indicator:
                arrow = pg.CurveArrow(scatter)
                text = pg.TextItem()
                graph.addItem(arrow)
                graph.addItem(text)
                self._graph_items[name]['arrow'] = arrow
                self._graph_items[name]['text'] = text

            self.graph_manager.register_graph(name, graph)

    def _update_graph(self, name, x, y, v, indicator='maxvalue'):
        self.init_graph(name, indicator)

        c = [255 * i / len(x) for i in range(len(x))]
        max_index = np.argmax(v)
        self._graph_items[name]['scatter'].setData(
            [{'pos': (xi, yi),
              'size': vi / max(v) * 20 + 5,
              'brush': pg.mkBrush(color=pg.mkColor(255, 255, 255)) if i == len(x) - 1 else pg.mkBrush(
                  color=pg.mkColor(255 - c, c, 0)),
              'symbol': '+' if i == len(x) - 1 else 'o'}
             for i, (xi, yi, vi, c) in enumerate(zip(x, y, v, c))])
        self._graph_items[name]['arrow'].setIndex(max_index)
        self._graph_items[name]['text'].setText(f'Max: {v[max_index]:.2f} ({x[max_index]:.2f}, {y[max_index]:.2f})')
        self._graph_items[name]['text'].setPos(x[max_index], y[max_index])


class GPExperiment(Experiment):

    delayed_data = None
    def delay_measure(self, data):
        self.delayed_data = data
        # for entry in data:
        #     entry['value'] = 0
        return data

    @run_decorator(md={})
    def plan(self,
             motors,
             bounds,
             other_devices,
             hyperparameters,
             hyperparameter_bounds,
             N,
             **kwargs):

        ##let's initialize the autonomous experimenter ...
        experiment = AutonomousExperimenterGP(bounds,
                                              self.delay_measure,
                                              hyperparameters,
                                              hyperparameter_bounds,
                                              init_dataset_size=10,
                                              acq_func=self.acq_func,
                                              append_data_after_send=True)
        experiment.instrument_func = partial(self.instrument_func,
                                             **kwargs)
        devices = motors + other_devices
        for device in devices:
            yield from stage(device)

        experiment.data.dataset = (yield from experiment.instrument_func(experiment.data.dataset))

        # TODO: handle yielding of init dataset actions

        # ...train...
        experiment.train()

        n = 0
        while n < N:
            cycle_start = time.time()
            res = experiment.gp_optimizer.ask(
                n=1,
                acquisition_function=experiment.acq_func,
                cost_function=experiment.cost_func,
                dask_client=experiment.acq_func_opt_dask_client)
            next_measurement_points = res["x"]
            post_var = experiment.gp_optimizer.posterior_covariance(next_measurement_points)["v(x)"]

            info = [{"hyperparameters": experiment.gp_optimizer.hyperparameters,
                     "posterior std": np.sqrt(post_var[j])} for j in range(len(next_measurement_points))]
            new_data = experiment.data.inject_arrays(next_measurement_points, info=info)
            new_data = yield from experiment.instrument_func(new_data)
            experiment.data.dataset = experiment.data.dataset + new_data
            experiment.x, experiment.y, experiment.v, experiment.t, experiment.c, vp = experiment.extract_data()
            experiment.tell(experiment.x, experiment.y, experiment.v, vp)
            if (len(experiment.data.dataset) % 10) == 0:
                experiment.train(pop_size=10, tol=1e-6, max_iter=20, method="global")
            invoke_in_main_thread(self.update_graphs, experiment)
            # show_result(experiment)
            yield from checkpoint()
            RunEngineControls().cycle_time.setText(f'{time.time()-cycle_start:.1f} s')

            n += 1

    def instrument_func(self, *args, **kwargs):
        raise NotImplementedError

    def acq_func(self, *args, **kwargs):
        raise NotImplementedError

    def update_graphs(self, experiment):
        x = [acq['position'][0] for acq in experiment.data.dataset]
        y = [acq['position'][1] for acq in experiment.data.dataset]

        for metric_name in experiment.data.dataset[-1].get('metrics', {}):
            self._update_graph(metric_name, x, y, [acq['metrics'][metric_name] for acq in experiment.data.dataset])

        self._update_graph('score', x, y, [acq['value'] for acq in experiment.data.dataset])

