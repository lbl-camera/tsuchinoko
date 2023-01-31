from functools import cached_property
from itertools import product
import time

import numpy as np
from pyqtgraph.parametertree.parameterTypes import SimpleParameter, ListParameter, GroupParameter

from tsuchinoko.adaptive import Engine, Data
from tsuchinoko.graphs.common import Variance, Score


def unit_cells(dims=2, slow_dim=-1, sort=True):
    visited = list(product(range(2), repeat=dims))
    # visited = [(0, 0), (1, 0), (0, 1), (1, 1)]
    yield from visited
    shift = .5
    diagonal = True
    while True:
        next_sites = []
        for site in visited:
            # print('site:', site)
            if diagonal and not np.any(np.asarray(site)==1):
                next_sites.append(tuple(np.asarray(site)+shift))
                # print('new1:', np.asarray(site)+shift)
            elif not diagonal:
                if site[slow_dim] == 0 and np.all(np.asarray(site)<1):  # right (fast)
                    for dim in range(dims):
                        if dim == slow_dim % dims: continue
                        shifts = np.zeros((dims,))
                        shifts[dim] = shift
                        next = tuple(np.asarray(site) + shifts)
                        next_sites.append(next)
                        # print('new2:', np.asarray(site) + shifts)
                if site[slow_dim] < 1:  # up (slow)
                    shifts = np.zeros((dims,))
                    shifts[slow_dim] = shift
                    next = tuple(np.asarray(site) + shifts)
                    next_sites.append(next)
                    # print('new3:', np.asarray(site) + shifts)

        yield from next_sites
        visited.extend(next_sites)
        if sort:
            visited = sorted(visited, key=lambda site: site[slow_dim]*10000+sum(site)-site[slow_dim])

        if not diagonal:
            shift /= 2
        diagonal = not diagonal


class Grid(Engine):
    dimensionality: int = None

    def __init__(self, parameter_bounds):
        self.dimensionality = len(parameter_bounds)

        for i in range(self.dimensionality):
            for j, edge in enumerate(['min', 'max']):
                self.parameters[('bounds', f'axis_{i}_{edge}')] = parameter_bounds[i][j]

        self.reset()
        self.graphs = [Variance(), Score(), ]


    @cached_property
    def parameters(self):
        bounds_parameters = [SimpleParameter(title=f'Axis #{i + 1} {edge}', name=f'axis_{i}_{edge}', type='float')
                             for i in range(self.dimensionality) for edge in ['min', 'max']]

        parameters = [GroupParameter(name='bounds', title='Axis Bounds', children=bounds_parameters),]
        return GroupParameter(name='top', children=parameters)

    def update_measurements(self, data: Data):
        ...

    def request_targets(self, position, **kwargs):
        bounds = [[self.parameters[('bounds', f'axis_{i}_{edge}')]
                   for edge in ['min', 'max']]
                  for i in range(self.dimensionality)]
        mins, maxs = zip(*bounds)
        target = np.asarray(next(self.unit_cells)) * (np.asarray(maxs)-np.asarray(mins))+np.asarray(mins)
        # time.sleep(.1)
        return [target]

    def reset(self):
        self.unit_cells = unit_cells()

    def train(self):
        ...

    def update_metrics(self, data: Data):
        ...
