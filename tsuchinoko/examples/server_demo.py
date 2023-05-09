from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage

from tsuchinoko import examples
from tsuchinoko.adaptive.gpCAM_in_process import GPCAMInProcessEngine
from tsuchinoko.core import ZMQCore
from tsuchinoko.execution.simple import SimpleEngine

# Load data from a jpg image to be used as a luminosity map
image = np.flipud(np.asarray(Image.open(Path(examples.__file__).parent/'sombrero_pug.jpg')))
luminosity = np.average(image, axis=2)


# Bilinear sampling will be used to effectively smooth pixel edges in source data
def bilinear_sample(pos):
    return pos, ndimage.map_coordinates(luminosity, [[pos[1]], [pos[0]]], order=1)[0], 1, {}


execution = SimpleEngine(measure_func=bilinear_sample)


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
