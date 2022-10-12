from queue import Queue
from typing import Callable, Tuple, List

from loguru import logger

from . import Engine


class SimpleEngine(Engine):
    def __init__(self, measure_func=Callable[[Tuple[float]], Tuple[float]]):
        super(SimpleEngine, self).__init__()

        self.measure_func = measure_func
        self.position = None
        self.targets = Queue()
        self.new_measurements = []

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
            self.new_measurements.append(self.measure_func(self.position))

        measurements = self.new_measurements
        self.new_measurements = []
        return measurements
