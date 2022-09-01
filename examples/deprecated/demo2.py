import logging
import time

import numpy as np
from PySide2.QtWidgets import QApplication
from bluesky.plan_stubs import mov, checkpoint, trigger_and_read
from ophyd.sim import SynAxis, SynSignal
from scipy.stats import multivariate_normal

from tsuchinoko.experiment import GPExperiment
from tsuchinoko.widgets.displays import RunEngineControls, Configuration
from tsuchinoko.widgets.mainwindow import MainWindow


class AlignmentExperiment(GPExperiment):
    bounds = np.array([[-.1, .1],
                       [-.1, .1]])
    hyperparameters = np.array([1, 1, 1])
    hyperparameter_bounds = np.array([[.1, 1],  # bounds square of 'value' range
                                      [0.01, 10],
                                      [0.01, 10]])
    max_N = 1e9

    # Devices
    motors = [motor1 := SynAxis(name='motor1', labels={'motors'}), motor2 := SynAxis(name='motor2', labels={'motors'})]

    # inject aliases to mock EpicsMotor
    for device in [motor1, motor2]:
        device.user_readback = device.readback

    def measure_monitor(self):
        beam_center = self.motor1.readback.get(), self.motor2.readback.get()
        beam_stddev = self.motor1.readback.get() ** 2 + .5, self.motor2.readback.get() ** 2 + .5

        return 10 * multivariate_normal.pdf((0, 0),
                                            beam_center,
                                            np.diag(np.asarray(beam_stddev) ** 2))  # * np.random.rand()*1e-1

    def __init__(self, *args, **kwargs):
        super(AlignmentExperiment, self).__init__(*args, **kwargs)

        self.monitor = SynSignal(name='monitor',
                                 labels={'monitor'},
                                 func=self.measure_monitor)

        self.beam_stddev_x = SynSignal(name='beam_stddev_x',
                                       labels={'beam_stddev_x'},
                                       func=lambda: self.motor1.readback.get() ** 2 + .5)
        self.beam_stddev_y = SynSignal(name='beam_stddev_y',
                                       labels={'beam_stddev_y'},
                                       func=lambda: self.motor2.readback.get() ** 2 + .5)

    def acq_func(self, x, gp):
        m = gp.posterior_mean(x)["f(x)"]
        v = gp.posterior_covariance(x)["v(x)"]
        # print('val:', m, v)
        return m + (Configuration().posterior_weight_factor.value()) * np.sqrt(v)

    def instrument_func(self, data, monitor, motor1, motor2):
        for entry in data:
            yield from checkpoint()
            cycle_start = time.time()

            next_motor_position = entry['position']
            yield from mov(motor1, next_motor_position[0], motor2, next_motor_position[1])

            ret = (yield from trigger_and_read([self.monitor, self.beam_stddev_x, self.beam_stddev_y], name='primary'))
            amplitude = ret['monitor']['value']
            stddev_x = ret['beam_stddev_x']['value']
            stddev_y = ret['beam_stddev_y']['value']

            # get value
            metric_factors = [Configuration().mean_weight.value(), Configuration().stddev_x_weight.value(), Configuration().stddev_y_weight.value()]
            metric_vec = np.asarray([amplitude, 1 / stddev_x, 1 / stddev_y]) * np.asarray(metric_factors)
            logging.info(msg=f'metrics: {metric_vec}')
            entry['value'] = np.linalg.norm(metric_vec)
            entry['variance'] = 0.0005 ** 2  # TODO: use the variance from the fit
            entry['metrics'] = dict()
            entry['metrics']['amplitude'] = amplitude
            entry['metrics']['X stddev'] = stddev_x
            entry['metrics']['Y stddev'] = stddev_y
            entry['metrics']['amplitude weighted'] = metric_vec[0]
            entry['metrics']['X stddev weighted (inverted)'] = metric_vec[1]
            entry['metrics']['Y stddev weighted (inverted)'] = metric_vec[2]

            RunEngineControls().measurement_time.setText(f'{time.time() - cycle_start:.1f} s')
        return data

    @property
    def plan(self):
        yield from super(AlignmentExperiment, self).plan(self.motors,
                                                         self.bounds,
                                                         [self.monitor, self.beam_stddev_x, self.beam_stddev_y],
                                                         self.hyperparameters,
                                                         self.hyperparameter_bounds,
                                                         self.max_N,
                                                         monitor=self.monitor,
                                                         motor1=self.motor1,
                                                         motor2=self.motor2,
                                                         )


if __name__ == '__main__':
    import pyqtgraph as pg

    qapp = QApplication([])

    main_window = MainWindow()

    main_window.experiment = AlignmentExperiment(main_window.graph_manager_widget)

    main_window.show()

    iv = pg.PlotWidget()
    scatter = pg.ScatterPlotItem(x=[0], y=[0], size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 120))
    arrow = pg.CurveArrow(scatter)
    text = pg.TextItem()
    iv.addItem(scatter)
    iv.addItem(arrow)
    iv.addItem(text)

    exit(qapp.exec_())
