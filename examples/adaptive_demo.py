from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
import numpy as np

from tsuchinoko.adaptive.adaptive import Adaptive2D
from tsuchinoko.adaptive.gpCAM_in_process import GPCAMInProcessEngine
from tsuchinoko.core import ZMQCore
from tsuchinoko.execution.simple import SimpleEngine

# Load data from a jpg image to be used as a luminosity map
image = np.flipud(np.asarray(Image.open(Path(__file__).parent/'sombrero_pug.jpg')))
luminosity = np.average(image, axis=2)


# Bilinear sampling will be used to effectively smooth pixel edges in source data
def bilinear_sample(pos):
    return pos, ndimage.map_coordinates(luminosity, [[pos[1]], [pos[0]]], order=1)[0], 1, {}


# Poisson noise applied to measured value; variance is set in accordance with poisson statistics
def noisy_sample(pos):
    pos, value, variance, extra = bilinear_sample(pos)
    return pos, np.random.poisson(value), value, extra


def gaussian_sample(pos):
    pos, value, variance, extra = bilinear_sample(pos)
    return pos, np.random.standard_normal(1)[0]*30 + value, value, extra


execution = SimpleEngine(measure_func=gaussian_sample)


# Define a gpCAM adaptive engine with initial parameters
adaptive = Adaptive2D(
                                parameter_bounds=[(0, image.shape[1]),
                                                  (0, image.shape[0])],
                                )

# Construct a core server
core = ZMQCore()
core.set_adaptive_engine(adaptive)
core.set_execution_engine(execution)

if __name__ == '__main__':
    # Start the core server
    core.main()
