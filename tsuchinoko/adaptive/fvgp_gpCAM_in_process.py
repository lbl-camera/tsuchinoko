import numpy as np

from gpcam.gp_optimizer import  fvGPOptimizer
from .acquisition_functions import explore_target_100, radical_gradient
from .gpCAM_in_process import GPCAMInProcessEngine
from ..graphs.common import GPCamPosteriorCovariance, GPCamAcquisitionFunction, GPCamPosteriorMean, Table

acquisition_functions = {s: s for s in ['variance', 'shannon_ig', 'ucb', 'maximum', 'minimum', 'covariance', 'gradient', 'explore_target_100']}
acquisition_functions['explore_target_100'] = explore_target_100
acquisition_functions['radical_gradient'] = radical_gradient


class FvgpGPCAMInProcessEngine(GPCAMInProcessEngine):
    """
    A multi-task adaptive engine powered by gpCAM: https://gpcam.readthedocs.io/en/latest/
    """

    def __init__(self, dimensionality, output_dim, output_number, parameter_bounds, hyperparameters, hyperparameter_bounds, **kwargs):
        self.kwargs = kwargs
        self.output_dim = output_dim
        self.output_number = output_number
        super(FvgpGPCAMInProcessEngine, self).__init__(dimensionality, parameter_bounds, hyperparameters, hyperparameter_bounds, **kwargs)

        if dimensionality == 2:
            self.graphs = [GPCamPosteriorCovariance(),
                           GPCamAcquisitionFunction(),
                           GPCamPosteriorMean(),
                           Table()]
        elif dimensionality > 2:
            self.graphs = [GPCamPosteriorCovariance(),
                           Table()]

    def init_optimizer(self):
        parameter_bounds = np.asarray([[self.parameters[('bounds', f'axis_{i}_{edge}')]
                                        for edge in ['min', 'max']]
                                       for i in range(self.dimensionality)])
        hyperparameters = np.asarray([self.parameters[('hyperparameters', f'hyperparameter_{i}')]
                                      for i in range(self.dimensionality + 1)])

        self.optimizer = fvGPOptimizer(self.dimensionality, self.output_dim, self.output_number, parameter_bounds)
        self.optimizer.tell(np.empty((1, self.dimensionality)), np.empty((1, self.output_number)))  # we'll wipe this out later; required for initialization
        self.optimizer.init_fvgp(hyperparameters)

    def _set_hyperparameter(self, parameter, value):
        self.optimizer.gp_initialized = False  # Force re-initialization
        self.optimizer.init_fvgp(np.asarray([self.parameters[('hyperparameters', f'hyperparameter_{i}')]
                                           for i in range(self.dimensionality + 1)]))
