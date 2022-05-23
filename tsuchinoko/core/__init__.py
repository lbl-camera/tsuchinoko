import time

import numpy as np

from ..execution import Engine as ExecutionEngine
from ..adaptive import Engine as AdaptiveEngine, Data


class Core:
    def __init__(self):
        self.execution_engine: ExecutionEngine = None
        self.adaptive_engine: AdaptiveEngine = None

        self.iteration = 0

    def set_execution_engine(self, engine: ExecutionEngine):
        self.execution_engine = engine

    def set_adaptive_engine(self, engine: AdaptiveEngine):
        self.adaptive_engine = engine

    def start(self):
        data = Data(dimensionality=self.adaptive_engine.dimensionality)

        while True:
            print('getting position')
            position = self.execution_engine.get_position()
            print('getting targets')
            targets = self.adaptive_engine.request_targets(position, 10)
            print('updating targets')
            self.execution_engine.update_targets(targets)
            print('getting measurements')
            new_measurements = self.execution_engine.get_measurements()
            if len(new_measurements):
                print('injecting new measurements')
                data.inject_new(new_measurements)
                self.adaptive_engine.update_measurements(data)
