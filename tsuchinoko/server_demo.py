import asyncio

import numpy as np
from ophyd.sim import SynAxis, Device, Cpt, SynSignalRO, SynSignal
from bluesky.plan_stubs import mv, trigger_and_read, create, stage, checkpoint, mov
from PIL import Image

from tsuchinoko.adaptive.gpCAM_in_process import GPCAMInProcessEngine
from tsuchinoko.core import Core, ZMQCore
from tsuchinoko.execution.bluesky_in_process import BlueskyInProcessEngine


if __name__ == '__main__':
    image = np.flipud(np.asarray(Image.open('test2.jpg')))
    luminosity = np.average(image, axis=2)
    blurred_luminosity = ndimage.gaussian_filter(luminosity, sigma=5)

    def bilinear_sample(img, pos):
        return ndimage.map_coordinates(img, [[pos[0]], [pos[1]]], order=1)


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


    point_detector = PointDetector('point_detector')

    def measure_target(target):
        yield from checkpoint()
        yield from mov(point_detector.motor1, target[0], point_detector.motor2, target[1])
        ret = (yield from trigger_and_read([point_detector]))
        return ret[point_detector.value.name]['value'], 2  # variance of 1

    def get_position():
        yield from checkpoint()
        return point_detector.motor1.position, point_detector.motor2.position


    adaptive = GPCAMInProcessEngine(dimensionality=2,
                                    parameter_bounds=[(0, image.shape[1]),
                                                      (0, image.shape[0])],
                                    hyperparameters=[255, 2, 2],
                                    hyperparameter_bounds=[(0, 255),
                                                           (0, 1e1),
                                                           (0, 1e1)])
    execution = BlueskyInProcessEngine(measure_target, get_position)

    core = ZMQCore()
    core.set_adaptive_engine(adaptive)
    core.set_execution_engine(execution)

    core.main()
