from queue import Queue, Empty

from bluesky.plan_stubs import open_run, null

from tsuchinoko.utils.runengine import get_run_engine
from . import Engine


class BlueskyInProcessEngine(Engine):
    """
    An execution engine powered by Bluesky. This engine utilized Bluesky's "plan" concept for experimental procedures,
    and measurements are natively compatible with Databroker ingestion.

    https://nsls-ii.github.io/bluesky/index.html
    """
    def __init__(self, measure_target, get_position):
        # These would normally be on the remote end
        self.targets = Queue()
        self.RE = get_run_engine()
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
            try:
                target = tuple(self.targets.get(timeout=.1))
            except Empty:
                yield from null()
            else:
                self.position = target
                value, variance = (yield from measure_target(target))
                self.new_measurements.append((self.position, value, variance, {}))  # TODO: Add variance; TODO: add metrics

    def get_position(self):
        return self.position

    def get_measurements(self):
        measurements = self.new_measurements
        self.new_measurements = []
        return measurements
