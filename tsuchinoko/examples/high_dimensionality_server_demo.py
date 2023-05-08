import subprocess
import sys

import numpy as np
from scipy import ndimage

from tsuchinoko.adaptive.gpCAM_in_process import GPCAMInProcessEngine
from tsuchinoko.core import ZMQCore
from tsuchinoko.execution.simple import SimpleEngine

install_warning = """
ATTENTION!
This demo depends on a package that is not included in the Tsuchinoko installation.
As a convenience, this demo will install the required dependencies for you.
"""

try:
    from perlin_noise import PerlinNoise
except ImportError:
    print(install_warning)
    response = input('Would you like to install perlin-noise [y/(n)]? ')
    if response != 'y':
        print('Aborting...')
        sys.exit(1)
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'perlin-noise'])
    from perlin_noise import PerlinNoise



# More useful for display purposes; PerlinNoise interpolates internally and accepts float positions in range [0,1)
noise = PerlinNoise(octaves=8)
# size = 10
# shape = (size, size, size)
# indices = np.transpose(np.indices(shape)/size,(3,2,1,0)).reshape((np.prod(shape), 3))
# noise_map = map(noise, indices)




execution = SimpleEngine(measure_func=lambda position: (position, noise(position), 1, {}))


dimensionality = 5
# Define a gpCAM adaptive engine with initial parameters
adaptive = GPCAMInProcessEngine(dimensionality=dimensionality,
                                parameter_bounds=[(0, 10)] * dimensionality,
                                hyperparameters=[255] + [100] * dimensionality,
                                hyperparameter_bounds=[(0, 1e5)] * (dimensionality+1))

# Construct a core server
core = ZMQCore()
core.set_adaptive_engine(adaptive)
core.set_execution_engine(execution)

if __name__ == '__main__':
    # Start the core server
    core.main()
