import sys

import numpy as np

from gpcam.gp_optimizer import  fvGPOptimizer
from .gpCAM_in_process import GPCAMInProcessEngine
from ..graphs.common import GPCamPosteriorCovariance, GPCamAcquisitionFunction, GPCamPosteriorMean, Table, \
    GPCamHyperparameterPlot, Score


class FvgpGPCAMInProcessEngine(GPCAMInProcessEngine):
    """
    A multi-task adaptive engine powered by gpCAM: https://gpcam.readthedocs.io/en/latest/
    """

    def __init__(self, dimensionality, output_number, parameter_bounds, hyperparameters, hyperparameter_bounds, **kwargs):
        self.kwargs = kwargs
        self.output_number = output_number
        super(FvgpGPCAMInProcessEngine, self).__init__(dimensionality, parameter_bounds, hyperparameters, hyperparameter_bounds, **kwargs)

        if dimensionality == 2:
            self.graphs = [GPCamPosteriorCovariance(),
                           GPCamAcquisitionFunction(),
                           GPCamPosteriorMean(),
                           GPCamHyperparameterPlot(),
                           Score(),
                           Table()]
        elif dimensionality > 2:
            self.graphs = [GPCamPosteriorCovariance(),
                           Table()]

    # TODO: refactor this into base
    def init_optimizer(self):
        opts = self.gp_opts.copy()
        if sys.platform == 'darwin':
            opts['compute_device'] = 'numpy'

        hyperparameters = np.asarray([self.parameters[('hyperparameters', f'hyperparameter_{i}')]
                                      for i in range(self.num_hyperparameters)])

        self.optimizer = fvGPOptimizer(init_hyperparameters=hyperparameters,
                                       **opts)

    def _set_hyperparameter(self, parameter, value):
        self.optimizer.gp_initialized = False  # Force re-initialization
        self.optimizer.init_fvgp(np.asarray([self.parameters[('hyperparameters', f'hyperparameter_{i}')]
                                           for i in range(self.num_hyperparameters)]))

    def request_targets(self, position, **kwargs):
        kwargs.update({'x_out': np.arange(self.output_number)})
        return super().request_targets(position, **kwargs)