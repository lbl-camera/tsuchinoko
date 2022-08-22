from functools import cached_property

import numpy as np
from pyqtgraph.parametertree.parameterTypes import SimpleParameter, GroupParameter, ListParameter

from gpcam.gp_optimizer import GPOptimizer
from . import Engine, Data
from .acquisition_functions import explore_target_100, radical_gradient

acquisition_functions = {s: s for s in ['variance', 'shannon_ig', 'ucb', 'maximum', 'minimum', 'covariance', 'gradient', 'explore_target_100']}
acquisition_functions['explore_target_100'] = explore_target_100
acquisition_functions['radical_gradient'] = radical_gradient


class GPCAMInProcessEngine(Engine):

    def __init__(self, dimensionality, parameter_bounds, hyperparameters, hyperparameter_bounds, **kwargs):
        self.kwargs = kwargs
        self.optimizer = GPOptimizer(dimensionality, parameter_bounds)
        self.optimizer.tell(np.empty((1, dimensionality)), np.empty((1,)), np.empty((1,)))  # we'll wipe this out later; required for initialization
        self.optimizer.init_gp(hyperparameters)

        for i in range(dimensionality):
            for j, edge in enumerate(['min', 'max']):
                self.parameters[('bounds', f'axis_{i}_{edge}')] = parameter_bounds[i][j]
        for i in range(dimensionality+1):
            for j, edge in enumerate(['min', 'max']):
                self.parameters[('hyperparameters', f'hyperparameter_{i}_{edge}')] = hyperparameter_bounds[i][j]
            self.parameters.child('hyperparameters', f'hyperparameter_{i}').setValue(hyperparameters[i], blockSignal=self._set_hyperparameter)
                               # compute_device = compute_device,
                               # gp_kernel_function = self.kernel_func,
                               # gp_mean_function = self.prior_mean_func,
                               # sparse = sparse)

        self.optimizer.points = np.array([])
        self.optimizer.values = np.array([])
        self.optimizer.variances = np.array([])

    def reset(self):
        parameter_bounds = np.asarray([[self.parameters[('bounds', f'axis_{i}_{edge}')]
                                        for edge in ['min', 'max']]
                                       for i in range(self.dimensionality)])
        hyperparameters = np.asarray([self.parameters[('hyperparameters', f'hyperparameter_{i}')]
                                      for i in range(self.dimensionality+1)])

        self.optimizer = GPOptimizer(self.dimensionality, parameter_bounds)
        self.optimizer.tell(np.empty((1, self.dimensionality)), np.empty((1,)), np.empty((1,)))  # we'll wipe this out later; required for initialization
        self.optimizer.init_gp(hyperparameters)

        self.optimizer.points = np.array([])
        self.optimizer.values = np.array([])
        self.optimizer.variances = np.array([])

    @cached_property
    def parameters(self):
        hyper_parameters = [SimpleParameter(title=f'Hyperparameter #{i + 1}', name=f'hyperparameter_{i}', type='float')
                            for i in range(1 + self.dimensionality)]
        hyper_parameters_bounds = [SimpleParameter(title=f'Hyperparameter #{i + 1} {edge}', name=f'hyperparameter_{i}_{edge}', type='float')
                                   for i in range(1 + self.dimensionality) for edge in ['min', 'max']]
        bounds_parameters = [SimpleParameter(title=f'Axis #{i + 1} {edge}', name=f'axis_{i}_{edge}', type='float')
                             for i in range(self.dimensionality) for edge in ['min', 'max']]
        func_parameters = [ListParameter(title='Method', name='method', values=['global', 'local', 'hgdl']),
                           ListParameter(title='Acquisition Function', name='acquisition_function', values=list(acquisition_functions.keys())),
                           SimpleParameter(title='Queue Length', name='n', value=1, type='int'),
                           SimpleParameter(title='Population Size (global only)', name='pop_size', value=20, type='int'),
                           SimpleParameter(title='Tolerance', name='tol', value=1e-6, type='float')]

        # wireup callback-based parameters
        for param in hyper_parameters:
            param.sigValueChanged.connect(self._set_hyperparameter)

        parameters = func_parameters + [GroupParameter(name='bounds', title='Axis Bounds', children=bounds_parameters),
                                        GroupParameter(name='hyperparameters', title='Hyperparameter Bounds', children=hyper_parameters+hyper_parameters_bounds)]
        return GroupParameter(name='top', children=parameters)

    def _set_hyperparameter(self, parameter, value):
        self.optimizer.gp_initialized = False  # Force re-initialization
        self.optimizer.init_gp(np.asarray([self.parameters[('hyperparameters', f'hyperparameter_{i}')]
                                           for i in range(self.dimensionality+1)]))

    @property
    def dimensionality(self):
        return self.optimizer.iput_dim

    def update_measurements(self, data: Data):
        with data.r_lock():  # quickly grab values within lock before passing to optimizer
            positions = data.positions.copy()
            scores = data.scores.copy()
            variances = data.scores.copy()
        self.optimizer.tell(positions, scores, variances)
        self.update_metrics(data)

    def update_metrics(self, data: Data):
        with data.r_lock():  # quickly grab positions within lock before passing to optimizer
            positions = np.asarray(data.positions.copy())

        # compute posterior covariance without lock
        result_dict = self.optimizer.posterior_covariance(positions)

        # calculate acquisition function
        acquisition_function_value = list(self.optimizer.evaluate_acquisition_function(positions,
                                                     acquisition_function=acquisition_functions[self.parameters['acquisition_function']]))

        # assign to data object with lock
        with data.w_lock():
            data.states['Posterior Covariance'] = result_dict['S(x)']
            # data.metrics['Posterior Variance'] = list(result_dict['v(x)'])
            data.graphics_items['Posterior Covariance'] = 'imageitem'
            data.states['Acquisition Function'] = acquisition_function_value

    def request_targets(self, position, n, **kwargs):
        for key in ['acquisition_function', 'method', 'pop_size', 'tol']:
            kwargs.update({key: self.parameters[key]})
        kwargs.update({'bounds': np.asarray([[self.parameters[('bounds', f'axis_{i}_{edge}')]
                                              for edge in ['min', 'max']]
                                             for i in range(self.dimensionality)])})
        return self.optimizer.ask(position, n, acquisition_function=acquisition_functions[kwargs.pop('acquisition_function')], **kwargs)['x']

    def train(self):
        self.optimizer.train(np.asarray([[self.parameters[('hyperparameters', f'hyperparameter_{i}_{edge}')]
                                          for edge in ['min', 'max']]
                                         for i in range(self.dimensionality+1)]),
                             np.asarray([self.parameters[('hyperparameters', f'hyperparameter_{i}')]
                                         for i in range(self.dimensionality+1)]))


