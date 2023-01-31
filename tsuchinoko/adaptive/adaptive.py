from concurrent.futures import ProcessPoolExecutor
from functools import cached_property
# from queue import Queue

import adaptive
from . import _adaptive
from pyqtgraph.parametertree.parameterTypes import SimpleParameter, GroupParameter

from tsuchinoko.adaptive import Engine, Data
from tsuchinoko.graphs.common import Variance, Score
from tsuchinoko.utils import threads





class Adaptive2D(Engine):
    dimensionality: int = 2

    def __init__(self, parameter_bounds):
        for i in range(self.dimensionality):
            for j, edge in enumerate(['min', 'max']):
                self.parameters[('bounds', f'axis_{i}_{edge}')] = parameter_bounds[i][j]

        self.graphs = [Variance(), Score(), ]

        self.executor = None
        self.thread = None

        self.reset()

    # def __getstate__(self):
    #     state = self.__dict__.copy()
    #     del state['parameters']
    #     del state['thread']
    #     del state['learner']
    #     del state['runner']
    #
    #     return state

    @cached_property
    def parameters(self):
        bounds_parameters = [SimpleParameter(title=f'Axis #{i + 1} {edge}', name=f'axis_{i}_{edge}', type='float')
                             for i in range(self.dimensionality) for edge in ['min', 'max']]

        parameters = [GroupParameter(name='bounds', title='Axis Bounds', children=bounds_parameters), ]
        return GroupParameter(name='top', children=parameters)

    def update_measurements(self, data: Data):
        _adaptive.measurements.put(data.scores[-1])

    def request_targets(self, position, **kwargs):
        if not self.thread.isRunning():
            self.thread.start()
        return [_adaptive.targets.get()]

    def reset(self):
        # TODO: build safe exit for runner thread

        bounds = [tuple(self.parameters[('bounds', f'axis_{i}_{edge}')]
                  for edge in ['min', 'max'])
                  for i in range(self.dimensionality)]

        # clear queues
        while not _adaptive.targets.empty():
            _adaptive.targets.get()
        while not _adaptive.measurements.empty():
            _adaptive.measurements.get()

        self.learner = adaptive.Learner2D(_adaptive.measure, bounds=bounds)
        self.executor = ProcessPoolExecutor(initializer=_adaptive.set_global, initargs=(_adaptive.targets, _adaptive.measurements))

        # self.runner = adaptive.Runner(self.learner, , goal=lambda l: False)
        self.thread = threads.QThreadFuture(adaptive.BlockingRunner, self.learner, executor=self.executor,
                                            goal=lambda l: False)

    def train(self):
        ...

    def update_metrics(self, data: Data):
        ...
