from pathlib import Path
import numpy as np
from PIL import Image
from pyqtgraph import mkColor
from scipy import ndimage
from itertools import count

from tsuchinoko.adaptive.gpCAM_in_process import GPCAMInProcessEngine
from tsuchinoko.core import ZMQCore
from tsuchinoko.execution.simple import SimpleEngine
from tsuchinoko.graphs.common import Plot, MultiPlot, DynamicColorMultiPlot

# Load data from a jpg image to be used as a luminosity map
image = np.flipud(np.asarray(Image.open(Path(__file__).parent/'sombrero_pug.jpg')))
luminosity = np.average(image, axis=2)

# A counter to track measurement number
counter = count(1)

# rosenbrock?
# "pos" is 'annealing curve'
# scalar_color is 'grain size' and derived from ackley function of curve values

def objective(x, y):
    return -20.0 * np.exp(-0.2 * np.sqrt(0.5 * (x ** 2 + y ** 2))) - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) + np.e + 20


# Bilinear sampling will be used to effectively smooth pixel edges in source data
def bilinear_sample(pos):
    i = next(counter)
    v = objective(pos[0], np.arange(-20, 20))
    grain_size = np.random.random()
    return pos, \
        ndimage.map_coordinates(luminosity, [[pos[1]], [pos[0]]], order=1)[0], \
        1, \
        {'plot_data': v,
         'plot_label': f'Measurement #{i}',
         # 'pen': {'width': int(np.max(v)), 'color': np.min(v)}
         'scalar_color': grain_size
         }


execution = SimpleEngine(measure_func=bilinear_sample)


# Define a gpCAM adaptive engine with initial parameters
adaptive = GPCAMInProcessEngine(dimensionality=2,
                                parameter_bounds=[(0, image.shape[1]),
                                                  (0, image.shape[0])],
                                hyperparameters=[255, 100, 100],
                                hyperparameter_bounds=[(0, 1e5),
                                                       (0, 1e5),
                                                       (0, 1e5)])

multiplot_graph_example = DynamicColorMultiPlot(color_scalar_key='scalar_color', data_key='plot_data', name='Vector Metric Example', label_key='plot_label', accumulates=True)

adaptive.graphs.append(multiplot_graph_example)

# Construct a core server
core = ZMQCore()
core.set_adaptive_engine(adaptive)
core.set_execution_engine(execution)

if __name__ == '__main__':
    # Start the core server
    core.main()
