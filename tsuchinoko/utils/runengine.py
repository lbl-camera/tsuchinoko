import asyncio
import logging
import time
from dataclasses import dataclass, field
from functools import wraps
from queue import PriorityQueue, Empty
from typing import Any
import dask.distributed  # this insulates from errors related to dask asserting its own EventLoopPolicy as squashing the event loop setup for bluesky

from bluesky import RunEngine
from bluesky.utils import DuringTask, RunEngineInterrupted
from qtpy.QtCore import QObject, Signal

from tsuchinoko.utils import threads


@dataclass(order=True)
class PrioritizedPlan:
    priority: int
    args: Any = field(compare=False)


class QRunEngine(QObject):
    sigDocumentYield = Signal(str, dict)
    sigAbort = Signal()  # TODO: wireup me
    sigException = Signal(Exception)
    sigFinish = Signal()
    sigStart = Signal()
    sigPause = Signal()
    sigResume = Signal()
    sigReady = Signal()

    def __init__(self, **kwargs):
        super(QRunEngine, self).__init__()

        self._request_resume = False
        self._kwargs = kwargs
        self._RE = None

        # # TODO: pull from settings plugin
        # from suitcase.mongo_normalized import Serializer
        # #TODO create single databroker db
        # #python-dotenv stores name-value pairs in .env (add to .gitginore)
        # username=os.getenv("USER_MONGO")
        # pw = os.getenv("PASSWD_MONGO")
        # try:
        #     self._RE.subscribe(Serializer(f"mongodb://{username}:{pw}@localhost:27017/mds?authsource=mds",
        #                                  f"mongodb://{username}:{pw}@localhost:27017/fs?authsource=fs"))
        # except OperationFailure as err:
        #     msg.notifyMessage("Could not connect to local mongo database.",
        #                       title="xicam.Acquire Error",
        #                       level=msg.ERROR)
        #     msg.logError(err)

        self.sigFinish.connect(self._check_if_ready)

        self.queue = PriorityQueue()
        self.process_queue_thread = threads.QThreadFutureIterator(self.process_queue,
                                                                  # finished_slot=self._close,
                                                                  interrupt_callable=self._close_RE,
                                                                  name='runengine-qthread')
        self.process_queue_thread.start()

    @property
    def RE(self):
        return self._RE

    def _close_RE(self):
        if self._RE.state != 'idle':
            self._RE.abort('Application is closing.')

    # def _close(self):
    #     self.asyncio_loop.close()

    def process_queue(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self._RE = RunEngine(context_managers=[], during_task=DuringTask(), loop=self.loop, **self._kwargs)
        self._RE.subscribe(self.sigDocumentYield.emit)

        while True:
            yield
            # get a plan, wait up to .1 sec before hot looping
            if self._RE.state == 'idle':
                try:
                    priority_plan = self.queue.get(block=True, timeout=.1)  # timeout is arbitrary, we'll come right back
                except Empty:
                    continue
                priority, (args, kwargs) = priority_plan.priority, priority_plan.args
            elif self._RE.state == 'paused' and self._request_resume:
                pass
            else:
                time.sleep(.1)
                continue

            self.sigStart.emit()
            try:
                if self._RE.state == 'idle':
                    self._RE(*args, **kwargs)
                elif self._RE.state == 'paused' and self._request_resume:
                    self._request_resume = False
                    self._RE.resume()
            except RunEngineInterrupted:
                logging.critical("Run has been aborted by the user.")
            except RuntimeError as ex:
                logging.error("An error occured during a Bluesky plan. See the Tsuchinoko log for details.")
                logging.error(ex, exc_info=True)
                self.sigException.emit(ex)
            else:
                if priority_plan:
                    priority_plan = None
                    self.queue.task_done()
                    self.sigFinish.emit()
                # msg.showReady()

    @wraps(RunEngine.__call__)
    def __call__(self, *args, **kwargs):
        self.put(*args, **kwargs)

    @property
    def isIdle(self):
        return self._RE.state == 'idle'

    def abort(self, reason=''):
        if self._RE.state != 'idle':
            self._RE.abort(reason=reason)
            self.sigAbort.emit()

    def pause(self, defer=False):
        if self._RE.state != 'paused':
            self._RE.request_pause(defer)
            self.sigPause.emit()

    def resume(self, ):
        if self._RE.state == 'paused':
            self._request_resume = True
            self.sigResume.emit()

    def put(self, *args, priority=1, **kwargs):
        # handle ParameterizedPlan's
        # plan = args[0]
        # if isinstance(args[0], ParameterizedPlan):
        #     # Ask for parameters
        #     param = plan.parameter
        #     if param:
        #         ParameterDialog(param).exec_()

        # reserved = set(kwargs.keys()).union(['plan_type', 'plan_args', 'scan_id', 'time', 'uid'])
        # self._metadata_dialog = MetadataDialog(reserved=reserved)
        # self._metadata_dialog.open()
        # self._metadata_dialog.accepted.connect(partial(self._put, self._metadata_dialog, priority, args, kwargs))
        self._put(priority, args, kwargs)

    def _put(self, priority, args, kwargs):
        self.queue.put(PrioritizedPlan(priority, (args, kwargs)))
        if not self.process_queue_thread.running:
            self.process_queue_thread.start()

    def _check_if_ready(self):
        # RE has finished processing everything in the queue
        if self._RE.state == 'idle' and self.queue.unfinished_tasks == 0:
            self.sigReady.emit()


RE = None


def initialize():
    global RE


def get_run_engine() -> QRunEngine:
    global RE
    if RE is None:
        RE = QRunEngine()
        RE.sigDocumentYield.connect(logging.debug)
    return RE

threads.invoke_as_event(get_run_engine)
