import uuid
from functools import cached_property

import numpy as np
from pyqtgraph.parametertree.parameterTypes import SimpleParameter, GroupParameter, ListParameter

from gpcam.gp_optimizer import GPOptimizer
from . import Engine, Data
from .acquisition_functions import explore_target_100, radical_gradient
from ..graphs.common import Variance, GPCamPosteriorCovariance, Score, GPCamAcquisitionFunction, GPCamPosteriorMean, Table, HighDimensionalityGPCamPosteriorMean
from ..parameters import TrainingParameter

acquisition_functions = {s: s for s in ['variance', 'shannon_ig', 'ucb', 'maximum', 'minimum', 'covariance', 'gradient', 'explore_target_100']}
acquisition_functions['explore_target_100'] = explore_target_100
acquisition_functions['radical_gradient'] = radical_gradient


class GPCAMInProcessEngine(Engine):
    """
    An adaptive engine powered by gpCAM: https://gpcam.readthedocs.io/en/latest/
    """
    default_retrain_globally_at = (20, 50, 100, 400, 1000)
    default_retrain_locally_at = (20, 40, 60, 80, 100, 200, 400, 1000)

    def __init__(self, dimensionality, parameter_bounds, hyperparameters, hyperparameter_bounds, **kwargs):
        self.kwargs = kwargs
        self.dimensionality = dimensionality

        for i in range(dimensionality):
            for j, edge in enumerate(['min', 'max']):
                self.parameters[('bounds', f'axis_{i}_{edge}')] = parameter_bounds[i][j]
        for i in range(dimensionality + 1):
            for j, edge in enumerate(['min', 'max']):
                self.parameters[('hyperparameters', f'hyperparameter_{i}_{edge}')] = hyperparameter_bounds[i][j]
            self.parameters.child('hyperparameters', f'hyperparameter_{i}').setValue(hyperparameters[i], blockSignal=self._set_hyperparameter)

        self.reset()

        if dimensionality == 2:
            self.graphs = [GPCamPosteriorCovariance(),
                           GPCamAcquisitionFunction(),
                           GPCamPosteriorMean(),
                           # GPCamAverageCovariance(),
                           Table(),
                           Variance(),
                           Score()]
        elif dimensionality > 2:
            self.graphs = [GPCamPosteriorCovariance(),
                           HighDimensionalityGPCamPosteriorMean(dimensions=dimensionality, bounds=parameter_bounds),
                           # GPCamAverageCovariance(),
                           Table()]

    def init_optimizer(self):
        parameter_bounds = np.asarray([[self.parameters[('bounds', f'axis_{i}_{edge}')]
                                        for edge in ['min', 'max']]
                                       for i in range(self.dimensionality)])
        hyperparameters = np.asarray([self.parameters[('hyperparameters', f'hyperparameter_{i}')]
                                      for i in range(self.dimensionality + 1)])

        self.optimizer = GPOptimizer(self.dimensionality, parameter_bounds)
        self.optimizer.tell(np.empty((1, self.dimensionality)), np.empty((1,)), np.empty((1,)))  # we'll wipe this out later; required for initialization
        self.optimizer.init_gp(hyperparameters)

    def reset(self):
        self._completed_training = {'global': set(),
                                    'local': set()}
        self.init_optimizer()

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
        func_parameters = [ListParameter(title='Method', name='method', limits=['global', 'local', 'hgdl']),
                           ListParameter(title='Acquisition Function', name='acquisition_function', limits=list(acquisition_functions.keys())),
                           SimpleParameter(title='Queue Length', name='n', value=1, type='int'),
                           SimpleParameter(title='Population Size (global only)', name='pop_size', value=20, type='int'),
                           SimpleParameter(title='Tolerance', name='tol', value=1e-6, type='float')]

        global_train_parameter = TrainingParameter(title='Train globally at...', name='global_training', addText='Add', children=[
            SimpleParameter(title='N=', name=str(uuid.uuid4()), value=N, type='int') for N in self.default_retrain_globally_at
        ])
        local_train_parameter = TrainingParameter(title='Train locally at...', name='local_training', addText='Add', children=[
            SimpleParameter(title='N=', name=str(uuid.uuid4()), value=N, type='int') for N in self.default_retrain_locally_at
        ])

        # wireup callback-based parameters
        for param in hyper_parameters:
            param.sigValueChanged.connect(self._set_hyperparameter)

        parameters = func_parameters + [GroupParameter(name='bounds', title='Axis Bounds', children=bounds_parameters),
                                        GroupParameter(name='hyperparameters', title='Hyperparameter Bounds', children=hyper_parameters + hyper_parameters_bounds),
                                        global_train_parameter,
                                        local_train_parameter, ]
        return GroupParameter(name='top', children=parameters)

    def _set_hyperparameter(self, parameter, value):
        self.optimizer.gp_initialized = False  # Force re-initialization
        self.optimizer.init_gp(np.asarray([self.parameters[('hyperparameters', f'hyperparameter_{i}')]
                                           for i in range(self.dimensionality + 1)]))

    def update_measurements(self, data: Data):
        with data.r_lock():  # quickly grab values within lock before passing to optimizer
            positions = data.positions.copy()
            scores = data.scores.copy()
            variances = data.variances.copy()
        self.optimizer.tell(positions, scores, variances)

    def update_metrics(self, data: Data):
        for graph in self.graphs:
            graph.compute(data, self)

    def request_targets(self, position):
        kwargs = {key: self.parameters[key] for key in ['acquisition_function', 'method', 'pop_size', 'tol']}
        kwargs.update({'bounds': np.asarray([[self.parameters[('bounds', f'axis_{i}_{edge}')]
                                              for edge in ['min', 'max']]
                                             for i in range(self.dimensionality)])})
        n = self.parameters['n']
        return self.optimizer.ask(position, n, acquisition_function=acquisition_functions[kwargs.pop('acquisition_function')], **kwargs)['x']

    def train(self):
        for method in ['global', 'local']:
            train_at = set(child.value() for child in self.parameters.child(f'{method}_training').children())

            for N in train_at:
                if len(self.optimizer.values) > N and N not in self._completed_training[method]:
                    self.optimizer.train(np.asarray([[self.parameters[('hyperparameters', f'hyperparameter_{i}_{edge}')]
                                                      for edge in ['min', 'max']]
                                                     for i in range(self.dimensionality + 1)]),
                                         np.asarray([self.parameters[('hyperparameters', f'hyperparameter_{i}')]
                                                     for i in range(self.dimensionality + 1)]), method=method)
                    self._completed_training[method].add(N)
                    # return  # only does global training if specified for both
        return True