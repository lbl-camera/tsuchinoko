from queue import Queue
from threading import Thread
import time

from . import Engine


class ThreadedInProcessEngine(Engine):
    def __init__(self, measure_target, get_position=None):
        # These would normally be on the remote end
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
        while True:
            target = tuple(self.targets.get())
            self.position = target
            value, variance = self.measure_target(target)
            self.new_measurements.append((self.position, value, variance, {'timestamp': time.time()}))

    def get_position(self):
        return self.position or self.position_getter()

    def get_measurements(self):
        measurements = self.new_measurements
        self.new_measurements = []
        return measurements