import time
from functools import partial
import logging

import numpy as np
import qdarkstyle
from PySide2.QtWidgets import QFormLayout, QDoubleSpinBox, QCheckBox, QPushButton, QSpacerItem, QLabel, QSizePolicy
from qtpy.QtCore import QTimer, Qt
from qtpy.QtWidgets import QApplication, QSlider, QWidget, QHBoxLayout
from bluesky.preprocessors import run_decorator
from gpcam.autonomous_experimenter import AutonomousExperimenterGP
# from caproto.sync.client import write, read
from bluesky.plan_stubs import mov, read, checkpoint, create, stage
from bluesky.plans import scan, tune_centroid
from ophyd import Device, Component, SignalRO, EpicsSignalRO, EpicsMotor
from ophyd.sim import SynAxis, SynGauss, DirectImage, SynSignalRO, SynSignal
from scipy.stats import multivariate_normal, norm

from tsuchinoko.plan_stubs import tune_centroid_and_fit
from tsuchinoko.utils.threads import QThreadFuture, invoke_in_main_thread
from tsuchinoko.utils import runengine


# TODO: goal values: FWHM 5 microns
# TODO: backlash: none
# TODO: positioner precision:
# TODO: photodiode accuracy: .0005 nA
# TODO: ring mode: 2-bunch mode has stable current
# TODO: measurement time: integrated over .5s
# TODO: mirror movement speed: fast
# TODO: device names, units: diode in nA
# TODO: limits
# TODO: test device control
# TODO:
# BCS701:DetectorDiodeCurrent:ai-AI
# BCS701:PinholeX; microns ; +- 14500 Y; +-6000 X; accurate to 10 nm
# BCS701:PinholeY
# BCS701:M112HorizAngle
# BCS701:M112VertAngle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        # logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)


def acq_func(x, gp):
    m = gp.posterior_mean(x)["f(x)"]
    v = gp.posterior_covariance(x)["v(x)"]
    # print('val:', m, v)
    return m + (posterior_weight_factor.value()) * np.sqrt(v)


def measure_quality(data, monitor, motor1, motor2, pinhole1, pinhole1min, pinhole1max, pinhole2, pinhole2min, pinhole2max,
                    pinhole_min_step):
    for entry in data:
        cycle_start = time.time()
        position = entry['position']
        # move to position
        logging.info(msg=f'Moving to position: {position}')
        yield from mov(motor1, position[0])
        yield from mov(motor2, position[1])

        # move pinhole to centroid and fit along the way
        logging.info(msg=f'Looking for centroid with pinhole...')
        x_fit = (yield from tune_centroid_and_fit([monitor], 'monitor', pinhole1, pinhole1min, pinhole1max, pinhole_min_step, snake=True, name='x_fit', expected_spot_size=1))
        y_fit = (yield from tune_centroid_and_fit([monitor], 'monitor', pinhole2, pinhole2min, pinhole2max, pinhole_min_step, snake=True, name='y_fit', expected_spot_size=1, debug=debug_fit.isChecked()))
        x_fit = (yield from tune_centroid_and_fit([monitor], 'monitor', pinhole1, pinhole1min, pinhole1max, pinhole_min_step, snake=True, name='x_fit', expected_spot_size=1, debug=debug_fit.isChecked()))
        logging.info(msg=f'Centroid estimated at: {pinhole1.readback.get()}, {pinhole2.readback.get()}')
        logging.info(msg=f'X fit:\n    amplitude: {x_fit.amplitude.value:.3f}\n    stddev: {x_fit.stddev.value:.3f}')
        logging.info(msg=f'Y fit:\n    amplitude: {y_fit.amplitude.value:.3f}\n    stddev: {y_fit.stddev.value:.3f}')

        # TODO: check error between x_amp, y_amp, and measured amp

        # move to max

        # measure

        # get value
        metric_factors = [mean_weight.value(), stddev_x_weight.value(), stddev_y_weight.value()]
        metric_vec = np.asarray([y_fit.amplitude.value, 1/x_fit.stddev.value, 1/y_fit.stddev.value]) * np.asarray(metric_factors)
        print('metrics:', *metric_vec)
        entry['value'] = np.linalg.norm(metric_vec)
        entry['variance'] = measure_variance
        measurement_time.setText(f'{time.time()-cycle_start:.1f} s')
    return data


@run_decorator(md={})
def alignment_plan(monitor, motor1, min1, max1, motor2, min2, max2, pinhole1, pinhole1min, pinhole1max, pinhole2, pinhole2min,
                   pinhole2max, pinhole_min_step, N, **kwargs):
    # set up parameter space
    parameters = np.array([[min1, max1],
                           [min2, max2],
                           ])

    ##set up some hyperparameters, if you have no idea, set them to 1 and make the training bounds large
    init_hyperparameters = np.array([1, 1, 1])
    hyperparameter_bounds = np.array([[.1, 1],  # bounds square of 'value' range
                                      [0.01, 10],
                                      [0.01, 10]])

    delayed_data = None
    def delay_measure(data):
        global delayed_data
        delayed_data = data
        # for entry in data:
        #     entry['value'] = 0
        return data

    ##let's initialize the autonomous experimenter ...
    experiment = AutonomousExperimenterGP(parameters,
                                          delay_measure,
                                          init_hyperparameters,
                                          hyperparameter_bounds,
                                          init_dataset_size=10,
                                          # acq_func='ucb',
                                          acq_func=acq_func,
                                          run_every_iteration=show_result,
                                          append_data_after_send=True)
    experiment.instrument_func = partial(measure_quality,
                                         monitor=monitor,
                                         motor1=motor1,
                                         motor2=motor2,
                                         pinhole1=pinhole1,
                                         pinhole1min=pinhole1min,
                                         pinhole1max=pinhole1max,
                                         pinhole2=pinhole2,
                                         pinhole2min=pinhole2min,
                                         pinhole2max=pinhole2max,
                                         pinhole_min_step=pinhole_min_step)
    devices = [monitor, motor1, motor2, pinhole1, pinhole2]
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
        invoke_in_main_thread(show_result, experiment)
        # show_result(experiment)
        yield from checkpoint()
        cycle_time.setText(f'{time.time()-cycle_start:.1f} s')

        n += 1


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
          'brush': pg.mkBrush(color=pg.mkColor(255, 255, 255)) if i == len(x) - 1 else pg.mkBrush(
              color=pg.mkColor(255 - c, c, 0)),
          'symbol': '+' if i == len(x) - 1 else 'o'}
         for i, (xi, yi, vi, c) in enumerate(zip(x, y, v, c))])
    arrow.setIndex(max_index)
    text.setText(f'Max: ({x[max_index]:.2f}, {y[max_index]:.2f})')
    text.setPos(x[max_index], y[max_index])


if __name__ == '__main__':
    import pyqtgraph as pg

    min1 = -3
    max1 = 3
    min2 = -3
    max2 = 3
    pinhole1min = -3
    pinhole1max = 3
    pinhole2min = -3
    pinhole2max = 3
    pinhole_min_step = .01

    # min1 = -9.5 # hard limit -13.9
    # max1 = -6.5 # hard limit 1.9
    # min2 = -3 # hard limit -17.4
    # max2 = 3 # hard limit 2.2
    # pinhole1min = -6000
    # pinhole1max = 6000
    # pinhole2min = -14500
    # pinhole2max = 14500
    # pinhole_min_step = .5  # 10% of reasonable FWHM
    mean_weight_default, stddev_x_weight_default, stddev_y_weight_default = [1e2, 1e-1, 1e-1]
    posterior_weight_factor_default = 3
    measure_variance = 0.0005**2  # TODO: Is this a unitless or unit-full value? a percentage?

    motor1 = SynAxis(name='motor1', labels={'motors'})
    motor2 = SynAxis(name='motor2', labels={'motors'})
    pinhole1 = SynAxis(name='pinhole1', labels={'motors'})
    pinhole2 = SynAxis(name='pinhole2', labels={'motors'})

    # motor1 = EpicsMotor(prefix='BCS701:M112HorizAngle', name='motor1')
    # motor2 = EpicsMotor(prefix='BCS701:M112VertAngle', name='motor2')
    # pinhole1 = EpicsMotor(prefix='BCS701:PinholeX', name='pinhole1')
    # pinhole1 = EpicsMotor(prefix='BCS701:PinholeY', name='pinhole1')

    def measure_monitor():
        beam_center = motor1.readback.get(), motor2.readback.get()
        beam_stddev = motor1.readback.get() ** 2 + .5, motor2.readback.get() ** 2 + .5
        pinhole = pinhole1.readback.get(), pinhole2.readback.get()
        # print(beam_center, beam_stddev, pinhole)

        return multivariate_normal.pdf(pinhole,
                                       beam_center,
                                       np.diag(np.asarray(beam_stddev) ** 2))# * np.random.rand()*1e-1

    monitor = SynSignal(name='monitor',
                        labels={'monitor'},
                        func=measure_monitor)

    # monitor = EpicsSignalRO(name='monitor', prefix='BCS701:DetectorDiodeCurrent:ai-AI')

    qapp = QApplication([])
    dark_stylesheet = qdarkstyle.load_stylesheet()
    qapp.setStyleSheet(dark_stylesheet)

    w = QWidget()
    outer_layout = QHBoxLayout()
    w.setLayout(outer_layout)
    outer_layout.setContentsMargins(0, 0, 0, 0)
    iv = pg.PlotWidget()
    scatter = pg.ScatterPlotItem(x=[0], y=[0], size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 120))
    arrow = pg.CurveArrow(scatter)
    text = pg.TextItem()
    posterior_weight_factor = QDoubleSpinBox()
    posterior_weight_factor.setValue(posterior_weight_factor_default)
    mean_weight = QDoubleSpinBox()
    mean_weight.setValue(mean_weight_default)
    stddev_x_weight = QDoubleSpinBox()
    stddev_x_weight.setValue(stddev_y_weight_default)
    stddev_y_weight = QDoubleSpinBox()
    stddev_y_weight.setValue(stddev_y_weight_default)
    debug_fit = QCheckBox()
    start = QPushButton('Start')
    pause = QPushButton('Pause')
    resume = QPushButton('Resume')
    pause.hide()
    resume.hide()
    measurement_time = QLabel('...')
    cycle_time = QLabel('...')

    w.layout().addWidget(iv)
    form_layout = QFormLayout()
    form_layout.addRow('Post. Mean/Covariance Weighting', posterior_weight_factor)
    form_layout.addRow('Mean weight', mean_weight)
    form_layout.addRow('Std. Dev. X weight', stddev_x_weight)
    form_layout.addRow('Std. Dev. Y weight', stddev_y_weight)
    form_layout.addRow('Debug centroid/tune/fit', debug_fit)
    form_layout.addWidget(start)
    form_layout.addWidget(pause)
    form_layout.addWidget(resume)
    form_layout.addItem(QSpacerItem(1, 1, vData=QSizePolicy.Expanding))
    form_layout.addRow('Measurement Time:', measurement_time)
    form_layout.addRow('Cycle Time:', cycle_time)
    w.layout().addLayout(form_layout)

    iv.addItem(scatter)
    iv.addItem(arrow)
    iv.addItem(text)
    w.show()
    N = 1000

    RE = runengine.get_run_engine()

    plan = alignment_plan(monitor, motor1, min1, max1, motor2, min2, max2, pinhole1, pinhole1min, pinhole1max, pinhole2,
                          pinhole2min, pinhole2max, pinhole_min_step, 10000)

    def start_plan():
        RE(plan)
        start.hide()
        pause.show()

    def pause_plan():
        RE.pause()
        resume.show()
        pause.hide()

    def resume_plan():
        RE.resume()
        resume.hide()
        pause.show()

    start.clicked.connect(start_plan)
    pause.clicked.connect(pause_plan)
    resume.clicked.connect(resume_plan)

    qapp.exec_()
