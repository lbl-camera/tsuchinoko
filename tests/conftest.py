from pathlib import Path
from threading import Thread

import pytest
import numpy as np
from PIL import Image
from loguru import logger
from pytest import fixture
from pytest_lazy_fixtures import lf as lazy_fixture
from scipy import ndimage

from tsuchinoko.adaptive.gpCAM_in_process import GPCAMInProcessEngine
from tsuchinoko.adaptive.random_in_process import RandomInProcess
from tsuchinoko.core import CoreState, Core
from tsuchinoko.execution.simple import SimpleEngine
from tsuchinoko.execution.threaded_in_process import ThreadedInProcessEngine


@fixture
def join_core():
    """Join a Core.main thread, failing with the real reason if it stays alive.

    experiment_loop pauses the Core on any exception and pushes it onto
    _exception_queue, and Core.main then spins in Paused forever, so a broken
    engine otherwise surfaces only as "the thread is still alive".
    """
    def _join(thread, core, timeout):
        thread.join(timeout=timeout)
        if not thread.is_alive():
            return

        exceptions = []
        while not core._exception_queue.empty():
            exceptions.append(core._exception_queue.get())
        detail = ('; '.join(repr(e) for e in exceptions) if exceptions
                  else 'no exception was queued')
        raise AssertionError(
            f'Core did not exit within {timeout}s (state={core.state}, '
            f'completed_iterations={core.data._completed_iterations}): {detail}')

    return _join


@fixture
def loguru_messages():
    """Collect loguru records emitted during a test.

    pytest's caplog only sees the stdlib logging module, and tsuchinoko logs
    through loguru, so caplog.records is always empty for our own warnings.
    """
    messages = []
    sink_id = logger.add(messages.append, level='DEBUG', format='{message}')
    yield messages
    logger.remove(sink_id)


@fixture
def image_data():
    image = np.flipud(
        np.asarray(Image.open(Path(__file__).parent.parent / 'tsuchinoko' / 'examples' / 'sombrero_pug.jpg')))
    luminosity = np.average(image, axis=2)
    return luminosity


@fixture
def image_func(image_data):
    def bilinear_sample(pos):
        return pos, ndimage.map_coordinates(image_data, [[pos[1]], [pos[0]]], order=1)[0], 1, {}
    return bilinear_sample


@fixture
def simple_execution_engine(image_func):
    return SimpleEngine(measure_func=image_func)


@fixture
def gpcam_engine(image_data):
    return GPCAMInProcessEngine(dimensionality=2,
                                parameter_bounds=[(0, image_data.shape[1]),
                                                  (0, image_data.shape[0])],
                                hyperparameters=[255, 100, 100],
                                hyperparameter_bounds=[(0, 1e5), (0, 1e5), (0, 1e5)])


@fixture
def random_engine(image_data):
    return RandomInProcess(dimensionality=2,
                           parameter_bounds=[(0, image_data.shape[1]),
                                             (0, image_data.shape[0])],
                           max_targets=100)


@fixture
def bluesky_execution_engine(image_func):
    pytest.importorskip('PySide6')
    from bluesky.plan_stubs import checkpoint, mov, trigger_and_read
    from ophyd import Device
    from ophyd.sim import SynAxis, SynSignal, Cpt
    from tsuchinoko.execution.bluesky_in_process import BlueskyInProcessEngine
    from tsuchinoko.utils.runengine import get_run_engine

    class PointDetector(Device):
        motor1 = Cpt(SynAxis, name='motor1')
        motor2 = Cpt(SynAxis, name='motor2')
        value = Cpt(SynSignal, name='value')

        def __init__(self, name):
            super(PointDetector, self).__init__(name=name)
            self.value.sim_set_func(self.get_value)

        def get_value(self):
            return image_func([int(self.motor2.position), int(self.motor1.position)])

        def trigger(self, *args, **kwargs):
            return self.value.trigger(*args, **kwargs)

    point_detector = PointDetector('point_detector')

    def measure_target(target):
        yield from checkpoint()
        yield from mov(point_detector.motor1, target[0], point_detector.motor2, target[1])
        ret = (yield from trigger_and_read([point_detector]))
        return ret[point_detector.value.name]['value'], 2

    def get_position():
        yield from checkpoint()
        return point_detector.motor1.position, point_detector.motor2.position

    execution = BlueskyInProcessEngine(measure_target, get_position)
    yield execution

    logger.info('starting bluesky engine teardown')
    RE = get_run_engine()
    RE.RE.halt()
    RE.process_queue_thread.requestInterruption()
    RE.process_queue_thread.wait()
    logger.info('bluesky engine teardown finished')


@fixture
def threaded_execution_engine(image_func):
    execution = ThreadedInProcessEngine(image_func)
    yield execution
    execution.exiting = True
    execution.measure_thread.join()


@fixture(params=[lazy_fixture('random_engine'),
                 lazy_fixture('gpcam_engine')])
def adaptive_test_engines(simple_execution_engine, request):
    adaptive_engine = request.param
    return adaptive_engine, simple_execution_engine


@fixture(params=[lazy_fixture('threaded_execution_engine'),
                 lazy_fixture('bluesky_execution_engine')])
def execution_test_engines(random_engine, request):
    execution_engine = request.param
    return random_engine, execution_engine


@fixture(params=[lazy_fixture('execution_test_engines'),
                 lazy_fixture('adaptive_test_engines')])
def core(request):
    adaptive_engine, execution_engine = request.param
    logger.info('starting setup')
    core = Core()
    core.set_adaptive_engine(adaptive_engine)
    core.set_execution_engine(execution_engine)
    server_thread = Thread(target=core.main, daemon=True)
    server_thread.start()
    core.state = CoreState.Starting
    logger.info('setup complete')

    yield core

    core.exit()
    server_thread.join(timeout=10)
    logger.info('teardown complete')
