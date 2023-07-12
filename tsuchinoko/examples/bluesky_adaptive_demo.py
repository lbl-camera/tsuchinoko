from tsuchinoko.adaptive.gpCAM_in_process import GPCAMInProcessEngine
from tsuchinoko.core import ZMQCore
from tsuchinoko.execution.bluesky_adaptive import BlueskyAdaptiveEngine

# NOTE: REQUIRES UNRELEASED DATABROKER AND TILED VERSIONS, AS WELL AS MISSING DEP FASTAPI AND CAPROTO

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

    # from tsuchinoko.execution.bluesky_adaptive import TsuchinokoAgent
    #
    # agent = TsuchinokoAgent()
    # while True:
    #     _, targets = agent.ask(0)
    #     agent.tell(targets[0], 1)
    #