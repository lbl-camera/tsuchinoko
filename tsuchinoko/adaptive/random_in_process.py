from functools import cached_property

import numpy as np
from pyqtgraph.parametertree.parameterTypes import SimpleParameter, ListParameter, GroupParameter

from tsuchinoko.adaptive import Engine, Data


class RandomInProcess(Engine):
    dimensionality: int = None

    def __init__(self, dimensionality: int, parameter_bounds):
        self.dimensionality = dimensionality

        for i in range(dimensionality):
            for j, edge in enumerate(['min', 'max']):
                self.parameters[('bounds', f'axis_{i}_{edge}')] = parameter_bounds[i][j]



    @cached_property
    def parameters(self):
        bounds_parameters = [SimpleParameter(title=f'Axis #{i + 1} {edge}', name=f'axis_{i}_{edge}', type='float')
                             for i in range(self.dimensionality) for edge in ['min', 'max']]
        func_parameters = [SimpleParameter(title='Queue Length', name='n', value=1, type='int'),]

        parameters = func_parameters + [GroupParameter(name='bounds', title='Axis Bounds', children=bounds_parameters),]
        return GroupParameter(name='top', children=parameters)

    def update_measurements(self, data: Data):
        ...

    def request_targets(self, position, n, **kwargs):
        bounds = [[self.parameters[('bounds', f'axis_{i}_{edge}')]
                   for edge in ['min', 'max']]
                  for i in range(self.dimensionality)]
        return [[np.random.uniform(min_, max_) for min_, max_ in bounds] for i in range(self.parameters['n'])]

    def reset(self):
        ...

    def train(self):
        ...

    def update_metrics(self, data: Data):
        ...