from dataclasses import dataclass, field, asdict
from collections import defaultdict


@dataclass()
class Data:
    dimensionality: int
    positions: list = field(default_factory=list)
    scores: list = field(default_factory=list)
    variances: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def inject_new(self, data):
        for datum in data:
            self.positions.append(datum[0])
            self.scores.append(datum[1])
            self.variances.append(datum[2])
            for metric in datum[3]:  # TODO: handle logical cases
                if metric not in self.metrics:
                    self.metrics[metric] = []
                self.metrics[metric].append(datum[3][metric])

    as_dict = asdict

    def __getitem__(self, item:slice):
        print(item)
        return Data(self.dimensionality, self.positions[item], self.scores[item], self.variances[item], {key: value[item] for key, value in self.metrics.items()})

    def __len__(self):
        return len(self.positions)

    def extend(self, data: 'Data'):
        self.positions += data.positions
        self.scores += data.scores
        self.variances += data.variances
        for key in self.metrics:
            self.metrics[key] += data[key]


class Engine:
    dimensionality: int = None

    def update_measurements(self, data: Data):
        ...

    def request_targets(self, position, n, **kwargs):
        ...

