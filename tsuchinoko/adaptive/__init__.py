from collections import defaultdict
from dataclasses import dataclass, field, asdict, fields
import threading
from copy import copy

@dataclass()
class Data:
    dimensionality: int = None
    positions: list = field(default_factory=list)
    scores: list = field(default_factory=list)
    variances: list = field(default_factory=list)
    metrics: dict = field(default_factory=lambda: defaultdict(list))
    states: dict = field(default_factory=dict)
    graphics_items: dict = field(default_factory=dict)

    def __post_init__(self):
        self._lock = threading.Lock()

    def inject_new(self, data):
        with self:
            for datum in data:
                self.positions.append(datum[0])
                self.scores.append(datum[1])
                self.variances.append(datum[2])
                for metric in datum[3]:  # TODO: handle logical cases
                    if metric not in self.metrics:
                        self.metrics[metric] = []
                    self.metrics[metric].append(datum[3][metric])

    def as_dict(self):
        self_copy = copy(self)
        self_copy.metrics = dict(self_copy.metrics)
        return asdict(self_copy)

    def __getitem__(self, item: slice):
        return Data(self.dimensionality,
                    self.positions[item],
                    self.scores[item],
                    self.variances[item],
                    {key: value[item] for key, value in self.metrics.items()},
                    self.states,
                    self.graphics_items)

    def __len__(self):
        return len(self.positions)

    def extend(self, data: 'Data'):
        with self:
            self.positions += data.positions
            self.scores += data.scores
            self.variances += data.variances
            for key in set(self.metrics) | set(data.metrics):
                self.metrics[key] += data.metrics.get(key, [])
            self.dimensionality = data.dimensionality
            self.graphics_items.update(data.graphics_items)
            self.states = data.states

    def __enter__(self):
        self._lock.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._lock.__exit__(exc_type, exc_val, exc_tb)

    def __bool__(self):
        return bool(len(self))


class Engine:
    dimensionality: int = None

    def update_measurements(self, data: Data):
        ...

    def request_targets(self, position, n, **kwargs):
        ...

    def reset(self):
        ...
