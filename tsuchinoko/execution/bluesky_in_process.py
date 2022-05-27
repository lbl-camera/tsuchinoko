from queue import Queue

from bluesky.plan_stubs import open_run

from tsuchinoko.utils.runengine import QRunEngine
from . import Engine


class BlueskyInProcessEngine(Engine):
    def __init__(self, measure_target, get_position):
        # These would normally be on the remote end
        self.targets = Queue()
        self.RE = QRunEngine()
        self.RE(self.target_queue_plan(measure_target, get_position))
        self.position = None
        self.new_measurements = []

    def update_targets(self, positions):
        with self.targets.mutex:
            self.targets.queue.clear()

        for position in positions:
            self.targets.put(position)

    def target_queue_plan(self, measure_target, get_position):
        yield from open_run()
        self.position = tuple((yield from get_position()))
        while True:
            target = tuple(self.targets.get())
            self.position = target
            value = yield from measure_target(target)
            self.new_measurements.append((self.position, value, 0, {}))  # TODO: Add variance; TODO: add metrics

    def get_position(self):
        return self.position

    def get_measurements(self):
        measurements = self.new_measurements
        self.new_measurements = []
        return measurements
