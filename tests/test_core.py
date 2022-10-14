import inspect
import os
import sys
import time
from pathlib import Path
from threading import Thread

import pytest
import numpy as np
from PIL import Image
from loguru import logger
from pytest import fixture
from scipy import ndimage

from tsuchinoko.adaptive.gpCAM_in_process import GPCAMInProcessEngine
from tsuchinoko.adaptive.random_in_process import RandomInProcess
from tsuchinoko.core import CoreState, ZMQCore
from tsuchinoko.execution.simple import SimpleEngine

# Disable logging to console when running tests
# NOTE: it seems there is a bug between loguru and pytest; pytest tries to log to a tempfile, but closes it when finished
# NOTE: if loguru has a backlog of messages
# logger.remove()

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)


@fixture
def image_data():
    # Load data from a jpg image to be used as a luminosity map
    image = np.flipud(np.asarray(Image.open(Path(__file__).parent.parent / 'examples' / 'sombrero_pug.jpg')))
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
                                                 (0, image_data.shape[0])])
    return adaptive


@fixture(params=[pytest.lazy_fixture('random_engine'),
                 pytest.lazy_fixture('gpcam_engine')])
def core(simple_execution_engine, request):
    adaptive_engine = request.param

    logger.info('starting setup')
    core = ZMQCore()
    core.set_adaptive_engine(adaptive_engine)
    core.set_execution_engine(simple_execution_engine)
    server_thread = Thread(target=core.main)
    server_thread.start()
    core.state = CoreState.Starting
    logger.info('setup complete')

    yield core

    core.exit()
    server_thread.join()
    logger.info('teardown complete')


def test_simple(core):
    time.sleep(1)
    assert len(core.data)
