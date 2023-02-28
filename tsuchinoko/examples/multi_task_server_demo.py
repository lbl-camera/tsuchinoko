from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from tsuchinoko.adaptive.fvgp_gpCAM_in_process import FvgpGPCAMInProcessEngine
from tsuchinoko.core import ZMQCore
from tsuchinoko.execution.simple import SimpleEngine

# Load data from a jpg image to be used as a luminosity map
images = [np.flipud(np.asarray(Image.open(Path(__file__).parent / f'peak{i + 1}.png'))) for i in range(2)]
luminosity = [np.average(image, axis=2) for image in images]


# Bilinear sampling will be used to effectively smooth pixel edges in source data
def bilinear_sample(pos):
    print(pos, ndimage.map_coordinates(luminosity, [[0, 1], [pos[1]] * 2, [pos[0]] * 2], order=1), 1, {})
    return pos, np.asarray(ndimage.map_coordinates(luminosity, [[0, 1], [pos[1]] * 2, [pos[0]] * 2], order=1)), (1, 1), {}


execution = SimpleEngine(measure_func=bilinear_sample)

# Define a gpCAM adaptive engine with initial parameters
adaptive = FvgpGPCAMInProcessEngine(dimensionality=2,
                                    output_dim=1,
                                    output_number=2,
                                    parameter_bounds=[(0, images[0].shape[1]),
                                                      (0, images[0].shape[0])],
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
