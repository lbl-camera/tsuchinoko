from dataclasses import dataclass, field

import numpy as np


@dataclass()
class Data:
    dimensionality: int
    positions: list = field(default_factory=list)
    scores: list = field(default_factory=list)
    variances: list = field(default_factory=list)

    def inject_new(self, data):
        for datum in data:
            self.positions.append(datum[0])
            self.scores.append(datum[1])
            self.variances.append(datum[2])


class Engine:
    dimensionality: int = None

    def update_measurements(self, data: Data):
        ...

    def request_targets(self, position, n, **kwargs):
        ...

