import sys
import uuid
from functools import cached_property
from typing import Callable

import numpy as np
from gpcam import kernels as _gpcam_kernels
from loguru import logger

from gpcam.gp_optimizer import GPOptimizer
from . import Engine, Data
from .acquisition_functions import explore_target_100, radical_gradient
from ..parameters.tree import SimpleParameter, GroupParameter, ListParameter, TrainingParameter

gpcam_acquisition_functions = {s: s for s in ['variance', 'shannon_ig', 'ucb', 'maximum', 'minimum', 'covariance', 'gradient', 'explore_target_100']}
gpcam_acquisition_functions['explore_target_100'] = explore_target_100
gpcam_acquisition_functions['radical_gradient'] = radical_gradient


def prepend_update_acquisition_functions(acquisition_functions:dict):
    cpy = gpcam_acquisition_functions.copy()
    gpcam_acquisition_functions.clear()
    gpcam_acquisition_functions.update(acquisition_functions)
    gpcam_acquisition_functions.update(cpy)


def _anisotropic_kernel(scalar_kernel):
    """Wrap a gpcam.kernels scalar kernel as a full kernel_function(x1, x2, hps).

    Mirrors fvgp.GP._default_kernel: hps[0] is the signal variance and
    hps[1:1+D] are per-axis length scales.
    """
    def kernel_function(x1, x2, hps):
        x1 = np.asarray(x1)
        x2 = np.asarray(x2)
        d = x1.shape[1]
        if len(hps) < 1 + d:
            raise ValueError(
                f"kernel expects {1 + d} hyperparameters (1 amplitude + {d} length scales); got {len(hps)}"
            )
        distance_matrix = np.zeros((len(x1), len(x2)))
        for i in range(d):
            distance_matrix += np.abs(
                np.subtract.outer(x1[:, i], x2[:, i]) / hps[1 + i]
            ) ** 2
        distance_matrix = np.sqrt(distance_matrix)
        return hps[0] * scalar_kernel(distance_matrix, 1.0)
    kernel_function.__name__ = f"anisotropic_{scalar_kernel.__name__}"
    return kernel_function


BUILTIN_KERNELS: dict[str, Callable] = {
    'matern_1_2': _anisotropic_kernel(_gpcam_kernels.exponential_kernel),
    'matern_3_2': _anisotropic_kernel(_gpcam_kernels.matern_kernel_diff1),
    'matern_5_2': _anisotropic_kernel(_gpcam_kernels.matern_kernel_diff2),
    'se': _anisotropic_kernel(_gpcam_kernels.squared_exponential_kernel),
}
# 'periodic' is intentionally omitted — gpCAM's periodic_kernel takes a
# period argument that has no place in the configure schema. Use a
# user:<name> ref to ship a custom kernel callable instead.


class GPCAMInProcessEngine(Engine):
    """
    An adaptive engine powered by gpCAM: https://gpcam.readthedocs.io/en/latest/
    """
    default_retrain_globally_at = (20, 50, 100, 400, 1000)
    default_retrain_locally_at = (20, 40, 60, 80, 100, 200, 400, 1000)
    default_retrain_mcmc_at = tuple()

    def __init__(self, dimensionality, parameter_bounds, hyperparameters, hyperparameter_bounds,
                 acquisition_functions:dict[str, Callable]=None,
                 gp_opts: dict = None, ask_opts: dict = None):
        self.dimensionality = dimensionality
        self.gp_opts = gp_opts or {}
        self.ask_opts = ask_opts or {}
        self.num_hyperparameters = len(hyperparameters)
        if acquisition_functions:
            prepend_update_acquisition_functions(acquisition_functions)

        for i in range(dimensionality):
            for j, edge in enumerate(['min', 'max']):
                self.parameters[('bounds', f'axis_{i}_{edge}')] = parameter_bounds[i][j]
        for i in range(self.num_hyperparameters):
            for j, edge in enumerate(['min', 'max']):
                self.parameters[('hyperparameters', f'hyperparameter_{i}_{edge}')] = hyperparameter_bounds[i][j]
            self.parameters.child('hyperparameters', f'hyperparameter_{i}').setValue(hyperparameters[i], blockSignal=self._set_hyperparameter)

        self.reset()

    def init_optimizer(self):
        opts = self.gp_opts.copy()
        if sys.platform == 'darwin':
            opts['compute_device'] = 'numpy'

        hyperparameters = np.asarray([self.parameters[('hyperparameters', f'hyperparameter_{i}')]
                                      for i in range(self.num_hyperparameters)])

        # Engine-level configure overrides (set by NATSService._handle_configure
        # or by callers directly). Each maps to a GPOptimizer kwarg; gp_opts
        # wins if the caller set the same key explicitly.
        kernel = getattr(self, 'kernel', None)
        if isinstance(kernel, str):
            try:
                kernel = BUILTIN_KERNELS[kernel]
            except KeyError:
                raise ValueError(
                    f"unknown kernel name {kernel!r}; expected one of "
                    f"{sorted(BUILTIN_KERNELS)} or a callable"
                )
        if callable(kernel):
            opts.setdefault('kernel_function', kernel)

        prior_mean = getattr(self, 'prior_mean', None)
        if callable(prior_mean):
            opts.setdefault('prior_mean_function', prior_mean)

        noise_function = getattr(self, 'noise_function', None)
        if callable(noise_function):
            opts.setdefault('noise_function', noise_function)

        noise_variances = getattr(self, 'noise_variances', None)
        if noise_variances is not None:
            opts.setdefault('noise_variances', noise_variances)

        self.optimizer = GPOptimizer(init_hyperparameters=hyperparameters,
                                     **opts)

    def reset(self):
        self._completed_training = {'global': set(),
                                    'local': set()}
        self.init_optimizer()

    @cached_property
    def parameters(self):
        hyper_parameters = [SimpleParameter(title=f'Hyperparameter #{i + 1}', name=f'hyperparameter_{i}', type='float')
                            for i in range(self.num_hyperparameters)]
        hyper_parameters_bounds = [SimpleParameter(title=f'Hyperparameter #{i + 1} {edge}', name=f'hyperparameter_{i}_{edge}', type='float')
                                   for i in range(self.num_hyperparameters) for edge in ['min', 'max']]
        bounds_parameters = [SimpleParameter(title=f'Axis #{i + 1} {edge}', name=f'axis_{i}_{edge}', type='float')
                             for i in range(self.dimensionality) for edge in ['min', 'max']]
        func_parameters = [ListParameter(title='Method', name='method', limits=['global', 'local', 'hgdl'], default='global'),
                           ListParameter(title='Acquisition Function', name='acquisition_function', limits=list(gpcam_acquisition_functions.keys()), default=list(gpcam_acquisition_functions.keys())[0]),
                           SimpleParameter(title='Queue Length', name='n', value=1, type='int'),
                           SimpleParameter(title='Population Size (global only)', name='pop_size', value=20, type='int'),
                           SimpleParameter(title='Tolerance', name='tol', value=1e-6, type='float')]

        global_train_parameter = TrainingParameter(title='Train globally at...', name='global_training', addText='Add', children=[
            SimpleParameter(title='N=', name=str(uuid.uuid4()), value=N, type='int') for N in self.default_retrain_globally_at
        ])
        local_train_parameter = TrainingParameter(title='Train locally at...', name='local_training', addText='Add', children=[
            SimpleParameter(title='N=', name=str(uuid.uuid4()), value=N, type='int') for N in self.default_retrain_locally_at
        ])
        mcmc_train_parameter = TrainingParameter(title='Train locally at...', name='mcmc_training', addText='Add',
                                                  children=[
                                                      SimpleParameter(title='N=', name=str(uuid.uuid4()), value=N,
                                                                      type='int') for N in
                                                      self.default_retrain_mcmc_at
                                                  ])

        # wireup callback-based parameters
        for param in hyper_parameters:
            param.sigValueChanged.connect(self._set_hyperparameter)

        parameters = func_parameters + [GroupParameter(name='bounds', title='Axis Bounds', children=bounds_parameters),
                                        GroupParameter(name='hyperparameters', title='Hyperparameter Bounds', children=hyper_parameters + hyper_parameters_bounds),
                                        global_train_parameter,
                                        local_train_parameter,
                                        mcmc_train_parameter,
                                        ]
        return GroupParameter(name='top', children=parameters)

    def _set_hyperparameter(self, parameter, value):
        if not self.optimizer.gp:
            # GP not built yet (no data told). init_optimizer reads
            # hyperparameters from the parameter tree at init_gp() time,
            # so the values still take effect once data arrives.
            return
        hyperparameters = np.asarray([self.parameters[('hyperparameters', f'hyperparameter_{i}')]
                                           for i in range(self.num_hyperparameters)])
        self.optimizer.set_hyperparameters(hyperparameters)

    def update_measurements(self, data: Data):
        with data.r_lock():  # quickly grab values within lock before passing to optimizer
            positions = data.positions.copy()
            scores = data.scores.copy()
            variances = data.variances.copy()
        self.optimizer.tell(np.asarray(positions), np.asarray(scores), np.asarray(variances))

    def init_gp(self, hyperparameters, **opts):
        self.optimizer.init_gp(hyperparameters, **opts)

    def update_metrics(self, data: Data):
        pass

    def request_targets(self, position, **kwargs):
        self.last_position = position
        bounds = np.asarray([[self.parameters[('bounds', f'axis_{i}_{edge}')]
                     for edge in ['min', 'max']]
                    for i in range(self.dimensionality)])
        n = self.parameters['n']

        # Random-exploration phase: either the GP isn't built yet, or the
        # configure-time `initial_points` quota hasn't been filled.
        initial_points = getattr(self, 'initial_points', None) or 0
        n_data = len(self.optimizer.x_data) if self.optimizer.gp else 0
        if not self.optimizer.gp or n_data < initial_points:
            return [[np.random.uniform(bounds[i][0], bounds[i][1]) for i in range(self.dimensionality)] for _ in range(n)]
        else:
            kwargs.update({key: self.parameters[key] for key in ['acquisition_function', 'method', 'pop_size', 'tol']})
            kwargs.update({'input_set': bounds})
            kwargs.update(self.ask_opts)
            return self.optimizer.ask(position=position,
                                      n=n,
                                      acquisition_function=gpcam_acquisition_functions[kwargs.pop('acquisition_function')],
                                      x0=np.asarray(position),
                                      **kwargs)['x'].astype(float)

    def train(self):
        # `training_method` (if set via configure) restricts training to a
        # single gpCAM method. Default is to iterate every method that has
        # its own schedule. adam/hgdl have no dedicated schedule param;
        # they reuse global_training's milestones.
        training_method = getattr(self, 'training_method', None)
        if training_method:
            schedule_method = training_method if training_method in ('global', 'local', 'mcmc') else 'global'
            methods = [(training_method, schedule_method)]
        else:
            methods = [(m, m) for m in ('global', 'local', 'mcmc')]

        for method, schedule_method in methods:
            self._completed_training.setdefault(method, set())
            train_at = set(child.value() for child in self.parameters.child(f'{schedule_method}_training').children())

            for N in train_at:
                if len(self.optimizer.y_data) > N and N not in self._completed_training[method]:
                    logger.info('Training in progress. This make take a while...')
                    self.optimizer.train(hyperparameter_bounds=
                                         np.asarray([[self.parameters[('hyperparameters', f'hyperparameter_{i}_{edge}')]
                                                      for edge in ['min', 'max']]
                                                     for i in range(self.num_hyperparameters)]),
                                         init_hyperparameters=
                                         np.asarray([self.parameters[('hyperparameters', f'hyperparameter_{i}')]
                                                     for i in range(self.num_hyperparameters)]), method=method)
                    self._completed_training[method].add(N)
                    logger.info(f"New hyperparameters: {self.optimizer.get_hyperparameters()}")

        return True
