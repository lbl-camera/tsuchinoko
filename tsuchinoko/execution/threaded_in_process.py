from queue import Queue, Empty
from threading import Thread
import time

from loguru import logger

from . import Engine


class ThreadedInProcessEngine(Engine):
    """
    A simple Execution Engine which performs measurements in a background thread.
    """

    def __init__(self, measure_target, get_position=None):
        # These would normally be on the remote end
        self.exiting = False
        self.targets = Queue()
        self.position_getter = get_position
        if get_position:
            self.position = get_position()
        else:
            self.position = (0, 0)

        self.new_measurements = []
        self.measure_target = measure_target
        self.measure_thread = Thread(target=self.measure_loop)
        self.measure_thread.start()

    def update_targets(self, positions):
        with self.targets.mutex:
            self.targets.queue.clear()

        for position in positions:
            self.targets.put(position)

    def measure_loop(self):
        while not self.exiting:
            logger.critical('looking for target')
            try:
                target = tuple(self.targets.get(timeout=.1))
            except Empty:
                continue
            else:
                self.position = target
                measurement = self.measure_target(target)
                self.new_measurements.append(measurement)

    def get_position(self):
        return self.position or self.position_getter()

    def get_measurements(self):
        measurements = self.new_measurements
        self.new_measurements = []
        return measurements