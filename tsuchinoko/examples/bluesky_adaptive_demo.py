from tsuchinoko.adaptive.gpCAM_in_process import GPCAMInProcessEngine
from tsuchinoko.core import ZMQCore
from tsuchinoko.execution.bluesky_adaptive import BlueskyAdaptiveEngine

# NOTE: REQUIRES UNRELEASED DATABROKER VERSIONS
# NOTE: To run this demo, you must also start a TsuchinokoAgent with Bluesky-Adaptive.
#       See tsuchinoko.execution.bluesky_adaptive for a primitive mocked demo.

bounds = [(0, 100)] * 2

# Define a gpCAM adaptive engine with initial parameters
adaptive = GPCAMInProcessEngine(dimensionality=2,
                                parameter_bounds=bounds,
                                hyperparameters=[255, 100, 100],
                                hyperparameter_bounds=[(0, 1e5),
                                                       (0, 1e5),
                                                       (0, 1e5)])

execution = BlueskyAdaptiveEngine()

# Construct a core server
core = ZMQCore()
core.set_adaptive_engine(adaptive)
core.set_execution_engine(execution)

if __name__ == '__main__':
    # Start the core server
    core.main()
