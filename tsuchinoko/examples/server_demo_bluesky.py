from pathlib import Path
import numpy as np
from PIL import Image
from bluesky.plan_stubs import trigger_and_read, checkpoint, mov
import ophyd  # insulates from ophyd import errors
from ophyd.sim import SynAxis, Device, Cpt, SynSignal
from scipy import ndimage

from tsuchinoko.adaptive.gpCAM_in_process import GPCAMInProcessEngine
from tsuchinoko.core import ZMQCore
from tsuchinoko.execution.bluesky_in_process import BlueskyInProcessEngine

# Load data from a jpg image to be used as a luminosity map
image = np.flipud(np.asarray(Image.open(Path(__file__).parent/'sombrero_pug.jpg')))
luminosity = np.average(image, axis=2)

# Bilinear sampling will be used to effectively smooth pixel edges in source data
def bilinear_sample(img, pos):
    return ndimage.map_coordinates(img, [[pos[0]], [pos[1]]], order=1)

# Define a simulated Ophyd Device class for measurement and control of position;
# this pulls measurements from the image data above
class PointDetector(Device):
    motor1 = Cpt(SynAxis, name='motor1')
    motor2 = Cpt(SynAxis, name='motor2')
    value = Cpt(SynSignal, name='value')

    def __init__(self, name):
        super(PointDetector, self).__init__(name=name)
        self.value.sim_set_func(self.get_value)

    def get_value(self):
        return np.average(bilinear_sample(luminosity, [int(self.motor2.position), int(self.motor1.position)]))

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

# Define a gpCAM adaptive engine with initial parameters
adaptive = GPCAMInProcessEngine(dimensionality=2,
                                parameter_bounds=[(0, image.shape[1]),
                                                  (0, image.shape[0])],
                                hyperparameters=[255, 100, 100],
                                hyperparameter_bounds=[(0, 1e5),
                                                       (0, 1e5),
                                                       (0, 1e5)])

# Construct a core server
core = ZMQCore()
core.set_adaptive_engine(adaptive)
core.set_execution_engine(execution)

if __name__ == '__main__':
    # Start the core server
    core.main()
