from functools import partial

import numpy as np
from PySide2.QtCore import QTimer, Qt
from PySide2.QtWidgets import QApplication, QSlider, QWidget, QHBoxLayout
from caproto.sync.client import write, read
from gpcam.autonomous_experimenter import AutonomousExperimenterGP


def simple(N):
    def acq_func(x, gp):
        m = gp.posterior_mean(x)["f(x)"]
        v = gp.posterior_covariance(x)["v(x)"]
        print('val:', m, v)
        return m + 3 * np.sqrt(v)

    def instrument(data):
        for entry in data:
            position = entry['position']
            # move to position
            for i, value in enumerate(position):
                write('test:optic_' + str(i + 1), value)

            # get value
            value = read('test:scalar_sensor').data[0]

            # circularity = ...
            entry['value'] = value  # + circularity
            entry['variance'] = 1e-1
        return data

    ##set up your parameter space
    parameters = np.array([[0, 3],
                           [0, 3],
                           # [0, 4],
                           # [0, 5]
                           ])

    ##set up some hyperparameters, if you have no idea, set them to 1 and make the training bounds large
    init_hyperparameters = np.array([1, 1, 1])
    hyperparameter_bounds = np.array([[.1, 1],  # bounds square of 'value' range
                                      [0.01, 10],
                                      [0.01, 10]])

    ##let's initialize the autonomous experimenter ...
    my_ae = AutonomousExperimenterGP(parameters,
                                     instrument,
                                     init_hyperparameters,
                                     hyperparameter_bounds,
                                     init_dataset_size=10,
                                     # acq_func='ucb',
                                     acq_func=acq_func,
                                     run_every_iteration=show_result)
    # ...train...
    my_ae.train()

    # ...and run. That's it. You successfully executed an autonomous experiment.
    # my_ae.go(N=N)

    return my_ae


def run(experiment: AutonomousExperimenterGP):
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
    experiment.data.dataset = experiment.data.dataset + experiment.instrument_func(new_data)
    experiment.x, experiment.y, experiment.v, experiment.t, experiment.c, vp = experiment.extract_data()
    experiment.tell(experiment.x, experiment.y, experiment.v, vp)
    if len(experiment.data.dataset) % 10:
        experiment.train(pop_size=10, tol=1e-6, max_iter=20, method="global")
    show_result(experiment)


def show_result(experiment):
    global scatter
    x = [acq['position'][0] for acq in experiment.data.dataset]
    y = [acq['position'][1] for acq in experiment.data.dataset]
    v = [acq['value'] for acq in experiment.data.dataset]
    c = [255 * i / len(experiment.data.dataset) for i in range(len(experiment.data.dataset))]
    max_index = np.argmax(v)
    scatter.setData(
        [{'pos': (xi, yi),
          'size': vi / max(v) * 20 + 5,
          'brush': pg.mkBrush(color=pg.mkColor(255, 255, 255)) if i == len(x) - 1 else pg.mkBrush(color=pg.mkColor(255 - c, c, 0)),
          'symbol': '+' if i == len(x) - 1 else 'o'}
         for i, (xi, yi, vi, c) in enumerate(zip(x, y, v, c))])
    arrow.setIndex(max_index)
    text.setText(f'Max: ({x[max_index]:.2f}, {y[max_index]:.2f})')
    text.setPos(x[max_index], y[max_index])


if __name__ == '__main__':
    import pyqtgraph as pg

    qapp = QApplication([])

    w = QWidget()
    w.setLayout(QHBoxLayout())
    iv = pg.PlotWidget()
    scatter = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 120))
    arrow = pg.CurveArrow(scatter)
    text = pg.TextItem()
    slider = QSlider(orientation=Qt.Vertical)
    slider.setMinimum(-100)
    slider.setMaximum(100)

    w.layout().addWidget(iv)
    w.layout().addWidget(slider)

    iv.addItem(scatter)
    iv.addItem(arrow)
    iv.addItem(text)
    w.show()
    N = 1000

    experiment = simple(N)

    timer = QTimer()
    timer.timeout.connect(partial(run, experiment))
    timer.start(.01)

    qapp.exec_()
