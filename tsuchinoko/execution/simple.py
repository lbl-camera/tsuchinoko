from queue import Queue
from typing import Callable, Tuple, List

from loguru import logger

from . import Engine


class SimpleEngine(Engine):
    def __init__(self, measure_func=Callable[[Tuple[float]], Tuple[float]], gives_variance=True, gives_position=True, gives_metrics=True):
        super(SimpleEngine, self).__init__()

        self.measure_func = measure_func
        self.position = None
        self.targets = Queue()
        self.new_measurements = []
        self._gives_variance = gives_variance
        self._gives_position = gives_position
        self._gives_metrics = gives_metrics

    def update_targets(self,  targets: List[Tuple]):
        with self.targets.mutex:
            self.targets.queue.clear()

        for target in targets:
            self.targets.put(target)

    def get_position(self) -> Tuple:
        return self.position

    def get_measurements(self) -> List[Tuple]:

        while not self.targets.empty():
            self.position = tuple(self.targets.get())
            measurement = self.measure_func(self.position)
            if not self._gives_position and not self._gives_position and not self._gives_metrics:
                measurement = [measurement]
            if not self._gives_position:
                measurement = [self.position, *measurement]
            if not self._gives_variance:
                measurement = [*measurement, .1]
            if not self._gives_metrics:
                measurement = [*measurement, {}]
            self.new_measurements.append(tuple(measurement))
            self.targets.task_done()

        measurements = self.new_measurements
        self.new_measurements = []
        return measurements


class SimplestEngine(SimpleEngine):
    def __init__(self, measure_func=Callable[[Tuple[float]], Tuple[float]]):
        super().__init__(measure_func, gives_position=False, gives_variance=False, gives_metrics=False)
