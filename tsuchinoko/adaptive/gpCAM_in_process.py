import numpy as np
from gpcam.gp_optimizer import GPOptimizer

from . import Engine, Data

from gpcam.autonomous_experimenter import AutonomousExperimenterGP


class GPCAMInProcessEngine(Engine):

    def __init__(self, dimensionality, parameter_bounds, hyperparameters, hyperparameter_bounds, **kwargs):
        self.parameter_bounds = parameter_bounds
        self.hyperparameters = hyperparameters
        self.hyperparameter_bounds = hyperparameter_bounds
        self.kwargs = kwargs
        self.optimizer = GPOptimizer(dimensionality, parameter_bounds)

        self.optimizer.tell(np.empty((1, dimensionality)), np.empty((1,)), np.empty((1,)))  # we'll wipe this out later; required for initialization
        self.optimizer.init_gp(hyperparameters)
                               # compute_device = compute_device,
                               # gp_kernel_function = self.kernel_func,
                               # gp_mean_function = self.prior_mean_func,
                               # sparse = sparse)

    @property
    def dimensionality(self):
        return self.optimizer.iput_dim

    def update_measurements(self, data: Data):
        self.optimizer.tell(data.positions, data.scores, data.variances)

    def request_targets(self, position, n, **kwargs):
        return self.optimizer.ask(position, n, **kwargs)['x']

