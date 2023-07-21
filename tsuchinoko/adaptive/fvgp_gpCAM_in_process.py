import sys

import numpy as np

from gpcam.gp_optimizer import  fvGPOptimizer
from .gpCAM_in_process import GPCAMInProcessEngine
from ..graphs.common import GPCamPosteriorCovariance, GPCamAcquisitionFunction, GPCamPosteriorMean, Table



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

    # TODO: refactor this into base
    def init_optimizer(self):
        parameter_bounds = np.asarray([[self.parameters[('bounds', f'axis_{i}_{edge}')]
                                        for edge in ['min', 'max']]
                                       for i in range(self.dimensionality)])
        hyperparameters = np.asarray([self.parameters[('hyperparameters', f'hyperparameter_{i}')]
                                      for i in range(self.num_hyperparameters)])

        self.optimizer = fvGPOptimizer(self.dimensionality, self.output_dim, self.output_number, parameter_bounds)

        if self.initial_x_data is not None and self.initial_y_data is not None:
            variance_kwargs = {}
            if self.initial_v_data is not None:
                variance_kwargs['variances'] = self.initial_v_data
            self.optimizer.tell(self.initial_x_data, self.initial_y_data, **variance_kwargs)

        opts = self.gp_opts.copy()
        # TODO: only fallback to numpy when packaged as an app
        if sys.platform == 'darwin':
            opts['compute_device'] = 'numpy'

        self.optimizer.init_fvgp(hyperparameters, **opts)

    def _set_hyperparameter(self, parameter, value):
        self.optimizer.gp_initialized = False  # Force re-initialization
        self.optimizer.init_fvgp(np.asarray([self.parameters[('hyperparameters', f'hyperparameter_{i}')]
                                           for i in range(self.num_hyperparameters)]))
