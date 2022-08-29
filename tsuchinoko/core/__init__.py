from loguru import logger
from enum import Enum, auto
from asyncio import sleep, create_task, events
import time
import threading
from queue import Queue

from .messages import FullDataRequest, FullDataResponse, PartialDataRequest, PartialDataResponse, StartRequest, UnknownResponse, PauseRequest, StateRequest, GetParametersRequest, SetParameterRequest, GetParametersResponse, SetParameterResponse, StopRequest, StateResponse, MeasureRequest, \
    MeasureResponse, ConnectRequest, ConnectResponse
from ..execution import Engine as ExecutionEngine
from ..adaptive import Engine as AdaptiveEngine, Data
from ..utils.logging import log_time


class CoreState(Enum):
    Connecting = auto()
    Inactive = auto()
    Starting = auto()
    Running = auto()
    Pausing = auto()
    Paused = auto()
    Resuming = auto()
    Stopping = auto()
    Restarting = auto()


class Core:
    def __init__(self):
        self.execution_engine: ExecutionEngine = None
        self.adaptive_engine: AdaptiveEngine = None

        self.iteration = 0

        self._state = CoreState.Inactive
        self._exception_queue = Queue()

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        logger.info(f'Changing core state to {value}')
        self._state = value

    def set_execution_engine(self, engine: ExecutionEngine):
        self.execution_engine = engine

    def set_adaptive_engine(self, engine: AdaptiveEngine):
        self.adaptive_engine = engine

    def main(self, debug=False):
        loop = events.new_event_loop()  # <---- this ensures the current loop is replaced
        try:
            events.set_event_loop(loop)
            loop.set_debug(debug)
            return loop.run_until_complete(self._main())
        finally:
            try:
                # _cancel_all_tasks(loop)
                loop.run_until_complete(loop.shutdown_asyncgens())
            finally:
                events.set_event_loop(None)
                loop.close()
        # import asyncio
        # asyncio.run(self._main())

    async def _main(self, min_response_sleep=.1):
        data = Data()
        experiment_thread = None

        while True:

            if self.state == CoreState.Running:
                pass
                # await sleep(min_response_sleep)  # short-circuit case
            elif self.state == CoreState.Starting:
                if not len(data):
                    data = Data(dimensionality=self.adaptive_engine.dimensionality)
                self.adaptive_engine.reset()
                experiment_thread = threading.Thread(target=self.experiment_loop, args=(data,))  # must hold ref
                experiment_thread.start()
                self.state = CoreState.Running

            elif self.state == CoreState.Inactive:
                pass
                # await sleep(min_response_sleep)

            elif self.state == CoreState.Paused:
                pass
                # await sleep(min_response_sleep)

            elif self.state == CoreState.Pausing:
                self.state = CoreState.Paused

            elif self.state == CoreState.Resuming:
                self.state = CoreState.Running

            elif self.state == CoreState.Stopping:
                self.state = CoreState.Inactive
                data = Data()
                # await sleep(min_response_sleep)

            await self.notify_clients(data)

    def experiment_loop(self, data):
        while True:
            if self.state == CoreState.Running:
                logger.info(f'Iteration: {len(data)}')
                try:
                    self.experiment_iteration(data)
                except Exception as ex:
                    self._exception_queue.put(ex)
                    self.state = CoreState.Pausing
                    logger.exception(ex)
            elif self.state == CoreState.Stopping:
                return
            else:
                time.sleep(.1)

    def experiment_iteration(self, data):
        with log_time('getting position', cumulative_key='getting position'):
            position = tuple(self.execution_engine.get_position())
        with log_time('getting targets', cumulative_key='getting targets'):
            targets = self.adaptive_engine.request_targets(position, n=1, acquisition_function='covariance')
        with log_time('updating targets', cumulative_key='updating targets'):
            self.execution_engine.update_targets(targets)
        with log_time('getting measurements', cumulative_key='getting measurements'):
            new_measurements = self.execution_engine.get_measurements()
        if len(new_measurements):
            with log_time('stashing new measurements', cumulative_key='injecting new measurements'):
                data.inject_new(new_measurements)
            with log_time('updating engine with new measurements', cumulative_key='updating engine with new measurements'):
                self.adaptive_engine.update_measurements(data)
        with log_time('training', cumulative_key='training'):
            self.adaptive_engine.train()

    async def notify_clients(self, data):
        ...


class ZMQCore(Core):
    def __init__(self):
        super(ZMQCore, self).__init__()
        # self.start_server()
        self.context = None
        self.poller = None

    def start_server(self):
        import zmq
        from zmq.asyncio import Context, Poller
        self.poller = Poller()
        self.context = Context()
        socket = self.context.socket(zmq.REP)
        socket.bind("tcp://*:5555")
        self.poller.register(socket, zmq.POLLIN)

    async def notify_clients(self, data: Data):
        import zmq
        if not self.poller:
            self.start_server()

        sockets = dict(await self.poller.poll())
        for socket in sockets:
            try:
                request = await socket.recv_pyobj()  # zmq.NOBLOCK)
            except zmq.ZMQError as ex:
                logger.exception(ex)
            else:
                logger.info(f"Received request: {request}")
                with log_time('preparing response', cumulative_key='preparing response'):

                    if isinstance(request, FullDataRequest):
                        with data.r_lock():
                            response = FullDataResponse(data.as_dict())
                    elif isinstance(request, PartialDataRequest):
                        if data and request.payload[0] <= len(data) and self.state == CoreState.Running:
                            with data.r_lock():
                                partial_data = data[request.payload[0]:]
                            response = PartialDataResponse(partial_data.as_dict(), request.payload[0])
                        else:
                            response = StateResponse(self.state)
                    elif isinstance(request, StartRequest):
                        if self.state == CoreState.Paused:
                            self.state = CoreState.Resuming
                        elif self.state == CoreState.Inactive:
                            self.state = CoreState.Starting
                        response = StateResponse(self.state)
                    elif isinstance(request, StopRequest):
                        self.state = CoreState.Stopping
                        response = StateResponse(self.state)
                    elif isinstance(request, PauseRequest):
                        self.state = CoreState.Pausing
                        response = StateResponse(self.state)
                    elif isinstance(request, StateRequest):
                        response = StateResponse(self.state)
                    elif isinstance(request, GetParametersRequest):
                        response = GetParametersResponse(self.adaptive_engine.parameters.saveState())
                    elif isinstance(request, SetParameterRequest):
                        child_path, value = request.payload
                        self.adaptive_engine.parameters.child(*child_path).setValue(value)
                        response = SetParameterResponse(True)
                    elif isinstance(request, MeasureRequest):
                        self.execution_engine.update_targets([request.payload[0]])
                        response = MeasureResponse(True)
                    elif isinstance(request, ConnectRequest):
                        response = ConnectResponse(self.state)
                    else:
                        response = UnknownResponse()

                logger.info(f'Sending response: {response}')
                await socket.send_pyobj(response)

                if isinstance(response, UnknownResponse):
                    logger.exception(ValueError(f'Unknown request received: {request}'))


