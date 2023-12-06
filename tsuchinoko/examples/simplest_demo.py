import numpy as np
from tsuchinoko.adaptive.adaptive import Adaptive2D
from tsuchinoko.execution.simple import SimplestEngine
from tsuchinoko.core import ZMQCore

def my_measurement_function(position, fwhm=.5):  # (a gaussian)
    return np.exp(-4*np.log(2) * ((position[0])**2 + (position[1])**2) / fwhm**2)

# Define an execution engine
execution = SimplestEngine(measure_func=my_measurement_function)

# Define an adaptive engine with initial parameters
adaptive = Adaptive2D(parameter_bounds=[(-1, 1), (-1, 1)])

# Start the core server
ZMQCore(adaptive_engine=adaptive, execution_engine=execution).main()

# Now open a Tsuchinoko client to watch the experiment
# $> tsuchinoko
