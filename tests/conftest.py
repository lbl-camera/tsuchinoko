from pathlib import Path
from threading import Thread
import dataclasses
import typing

import pytest
import numpy as np
from PIL import Image
from bluesky.plan_stubs import checkpoint, mov, trigger_and_read
from loguru import logger
from ophyd import Device
from ophyd.sim import SynAxis, SynSignal, Cpt
from pytest import fixture
from scipy import ndimage

from tsuchinoko.adaptive.gpCAM_in_process import GPCAMInProcessEngine
from tsuchinoko.adaptive.random_in_process import RandomInProcess
from tsuchinoko.core import CoreState, ZMQCore
from tsuchinoko.execution.bluesky_in_process import BlueskyInProcessEngine
from tsuchinoko.execution.simple import SimpleEngine
from tsuchinoko.execution.threaded_in_process import ThreadedInProcessEngine
from tsuchinoko.utils.runengine import get_run_engine
# Disable logging to console when running tests
# NOTE: it seems there is a bug between loguru and pytest; pytest tries to log to a tempfile, but closes it when finished
# NOTE: if loguru has a backlog of messages
# logger.remove()



@dataclasses.dataclass
class LazyFixture:
    """Lazy fixture dataclass."""

    name: str


def lazy_fixture(name: str) -> LazyFixture:
    """Mark a fixture as lazy."""
    return LazyFixture(name)


def is_lazy_fixture(value: object) -> bool:
    """Check whether a value is a lazy fixture."""
    return isinstance(value, LazyFixture)


def pytest_make_parametrize_id(
        config: pytest.Config,
        val: object,
        argname: str,
) -> str | None:
    """Inject lazy fixture parametrized id.

    Reference:
    - https://bit.ly/48Off6r

    Args:
        config (pytest.Config): pytest configuration.
        value (object): fixture value.
        argname (str): automatic parameter name.

    Returns:
        str: new parameter id.
    """
    if is_lazy_fixture(val):
        return typing.cast(LazyFixture, val).name
    return None


@pytest.hookimpl(tryfirst=True)
def pytest_fixture_setup(
        fixturedef: pytest.FixtureDef,
        request: pytest.FixtureRequest,
) -> object | None:
    """Lazy fixture setup hook.

    This hook will never take over a fixture setup but just simply will
    try to resolve recursively any lazy fixture found in request.param.

    Reference:
    - https://bit.ly/3SyvsXJ

    Args:
        fixturedef (pytest.FixtureDef): fixture definition object.
        request (pytest.FixtureRequest): fixture request object.

    Returns:
        object | None: fixture value or None otherwise.
    """
    if hasattr(request, "param") and request.param:
        request.param = _resolve_lazy_fixture(request.param, request)
    return None


def _resolve_lazy_fixture(__val: object, request: pytest.FixtureRequest) -> object:
    """Lazy fixture resolver.

    Args:
        __val (object): fixture value object.
        request (pytest.FixtureRequest): pytest fixture request object.

    Returns:
        object: resolved fixture value.
    """
    if isinstance(__val, list | tuple):
        return tuple(_resolve_lazy_fixture(v, request) for v in __val)
    if isinstance(__val, typing.Mapping):
        return {k: _resolve_lazy_fixture(v, request) for k, v in __val.items()}
    if not is_lazy_fixture(__val):
        return __val
    lazy_obj = typing.cast(LazyFixture, __val)
    return request.getfixturevalue(lazy_obj.name)


@fixture
def image_data():
    # Load data from a jpg image to be used as a luminosity map
    image = np.flipud(
        np.asarray(Image.open(Path(__file__).parent.parent / 'tsuchinoko' / 'examples' / 'sombrero_pug.jpg')))
    luminosity = np.average(image, axis=2)
    return luminosity


@fixture
def image_func(image_data):
    # Bilinear sampling will be used to effectively smooth pixel edges in source data
    def bilinear_sample(pos):
        return pos, ndimage.map_coordinates(image_data, [[pos[1]], [pos[0]]], order=1)[0], 1, {}

    return bilinear_sample


@fixture
def simple_execution_engine(image_func):
    execution = SimpleEngine(measure_func=image_func)
    return execution


@fixture
def gpcam_engine(image_data):
    # Define a gpCAM adaptive engine with initial parameters
    adaptive = GPCAMInProcessEngine(dimensionality=2,
                                    parameter_bounds=[(0, image_data.shape[1]),
                                                      (0, image_data.shape[0])],
                                    hyperparameters=[255, 100, 100],
                                    hyperparameter_bounds=[(0, 1e5),
                                                           (0, 1e5),
                                                           (0, 1e5)])
    return adaptive


@fixture
def random_engine(image_data):
    adaptive = RandomInProcess(dimensionality=2,
                               parameter_bounds=[(0, image_data.shape[1]),
                                                 (0, image_data.shape[0])],
                               max_targets=100
                               )  # this engine is FAST, so best to limit it
    return adaptive


@fixture
def bluesky_execution_engine(image_func):
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

    # Instantiate a simulated device
    point_detector = PointDetector('point_detector')

    # Define a Bluesky Plan component for performing measurements at targets.
    # Note that this returns the measured value and variance
    def measure_target(target):
        yield from checkpoint()
        yield from mov(point_detector.motor1, target[0], point_detector.motor2, target[1])
        ret = (yield from trigger_and_read([point_detector]))
        return ret[point_detector.value.name]['value'], 2  # variance of 1

    # Define a Bluesky Plan component to get the current position of the device
    # (which may not necessarily be precisely the requested position)
    def get_position():
        yield from checkpoint()
        return point_detector.motor1.position, point_detector.motor2.position

    # Define an execution engine with the measurement and get_position functions
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
    core = ZMQCore()
    core.set_adaptive_engine(adaptive_engine)
    core.set_execution_engine(execution_engine)
    server_thread = Thread(target=core.main)
    server_thread.start()
    core.state = CoreState.Starting
    logger.info('setup complete')

    yield core

    core.exit()
    server_thread.join()
    logger.info('teardown complete')
