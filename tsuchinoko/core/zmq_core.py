"""ZMQ-enabled Core server module.

Provides ZMQCore, a network-accessible extension of Core that handles
client requests over a ZMQ REP socket.

Backward compatibility: code that previously imported ZMQCore from
tsuchinoko.core can now import from tsuchinoko.core.zmq_core.
"""

import time
from pickle import UnpicklingError

from loguru import logger

from tsuchinoko.core import Core, CoreState
from tsuchinoko.core.messages import (
    Message, FullDataRequest, FullDataResponse, PartialDataRequest, PartialDataResponse,
    StartRequest, UnknownResponse, PauseRequest, StateRequest, GetParametersRequest,
    SetParameterRequest, GetParametersResponse, SetParameterResponse, StopRequest, StateResponse,
    MeasureRequest, MeasureResponse, ConnectRequest, ConnectResponse, ExceptionResponse,
    PushDataRequest, PushDataResponse, GraphsResponse, PullGraphsRequest, PushGraphsRequest,
    ReplayRequest, ReplayResponse, ExitRequest, SetComputeMetricsRequest,
)
from tsuchinoko.adaptive import Data
from tsuchinoko.utils.logging import log_time


class ZMQCore(Core):
    """ZMQ-enabled Core providing network server functionality.

    Extends Core with a ZMQ REP socket server that handles client
    requests. Each request type has a corresponding respond_* method
    that processes the request and returns an appropriate response.

    The server uses async polling to check for incoming messages
    during the main loop's notify_clients() calls.

    Attributes:
        context: ZMQ async context
        poller: ZMQ async poller for socket events
    """

    def __init__(self, *args, **kwargs):
        """Initialize ZMQCore with network components.

        Socket and poller are initialized lazily on first use.
        """
        super(ZMQCore, self).__init__(*args, **kwargs)
        self.context = None
        self.poller = None

    def start_server(self) -> None:
        """Initialize and bind the ZMQ server socket.

        Creates a REP socket bound to tcp://*:5555 and registers
        it with the poller for incoming message detection.
        """
        import zmq
        from zmq.asyncio import Context, Poller
        self.poller = Poller()
        self.context = Context()
        socket = self.context.socket(zmq.REP)
        socket.bind("tcp://*:5555")
        self.poller.register(socket, zmq.POLLIN)

    def respond_FullDataRequest(self, request: FullDataRequest) -> FullDataResponse:
        with self.data.r_lock():
            return FullDataResponse(self.data.as_dict())

    def respond_PartialDataRequest(self, request: PartialDataRequest) -> Message:
        if self.data and request.iteration <= len(self.data) and self.state == CoreState.Running:
            with self.data.r_lock():
                partial_data = self.data[request.iteration:]
            return PartialDataResponse(partial_data.as_dict(), request.iteration)
        else:
            return StateResponse(self.state, self.compute_metrics)

    def respond_PushDataRequest(self, request: PushDataRequest) -> PushDataResponse:
        self.data = Data(**request.data)
        return PushDataResponse()

    def respond_StartRequest(self, request: StartRequest) -> StateResponse:
        if self.state == CoreState.Paused:
            self.state = CoreState.Resuming
        elif self.state == CoreState.Inactive:
            self.state = CoreState.Starting
        return StateResponse(self.state, self.compute_metrics)

    def respond_StopRequest(self, request: StopRequest) -> StateResponse:
        self.state = CoreState.Stopping
        self.experiment_thread.join()
        return StateResponse(self.state, self.compute_metrics)

    def respond_ExitRequest(self, request: ExitRequest) -> StateResponse:
        self.state = CoreState.Exiting
        return StateResponse(self.state, self.compute_metrics)

    def respond_PauseRequest(self, request: PauseRequest) -> StateResponse:
        self.state = CoreState.Pausing
        return StateResponse(self.state, self.compute_metrics)

    def respond_StateRequest(self, request: StateRequest) -> Message:
        if not self._exception_queue.empty():
            return ExceptionResponse(self._exception_queue.get())
        else:
            return StateResponse(self.state, self.compute_metrics)

    def respond_GetParametersRequest(self, request: GetParametersRequest) -> GetParametersResponse:
        return GetParametersResponse(self.adaptive_engine.parameters.saveState())

    def respond_SetParameterRequest(self, request: SetParameterRequest) -> SetParameterResponse:
        self.adaptive_engine.parameters.child(*request.child_path).setValue(request.value)
        return SetParameterResponse(True)

    def respond_MeasureRequest(self, request: MeasureRequest) -> MeasureResponse:
        self._forced_position_queue.put(request.position)
        return MeasureResponse(True)

    def respond_ConnectRequest(self, request: ConnectRequest) -> ConnectResponse:
        return ConnectResponse(self.state, self.compute_metrics)

    def respond_PullGraphsRequest(self, request):
        return GraphsResponse([])

    def respond_PushGraphsRequest(self, request):
        return StateResponse(self.state, self.compute_metrics)

    def respond_SetComputeMetricsRequest(self, request: SetComputeMetricsRequest) -> StateResponse:
        self.compute_metrics = request.compute_metrics
        return StateResponse(self.state, self.compute_metrics)

    def respond_ReplayRequest(self, request: ReplayRequest) -> ReplayResponse:
        self._forced_measurement_queue.queue.clear()
        self._forced_position_queue.queue.clear()

        for position in request.positions:
            self._forced_position_queue.put(position)
        for measurement in request.measurements:
            self._forced_measurement_queue.put(measurement)
        logger.critical(f'Queue lengths: {len(self._forced_measurement_queue.queue)} {len(self._forced_position_queue.queue)}')
        return ReplayResponse(True)

    async def notify_clients(self) -> None:
        """Poll for and handle client requests.

        Called by the main loop to check for pending requests.
        For each received request:
        1. Deserialize the request object
        2. Find matching respond_* method
        3. Execute responder and send response
        4. Handle any errors with ExceptionResponse
        """
        import zmq
        if not self.poller:
            self.start_server()

        sockets = dict(await self.poller.poll(timeout=.1))
        for socket in sockets:
            try:
                request = await socket.recv_pyobj(zmq.NOBLOCK)
            except (zmq.ZMQError, zmq.error.Again) as ex:
                logger.exception(ex)
            except UnpicklingError as ex:
                logger.exception(ex)
                logger.critical('The above error prevented unpacking data from the client.')
            else:
                if not request:
                    time.sleep(.1)
                    continue

                logger.info(f"Received request: {request}")
                with log_time('preparing response', cumulative_key='preparing response'):
                    responder = getattr(self, f'respond_{request.__class__.__name__}', None)
                    if responder:
                        try:
                            response = responder(request)
                        except Exception as ex:
                            response = ExceptionResponse(ex)
                    else:
                        response = UnknownResponse()

                logger.info(f'Sending response: {response}')
                await socket.send_pyobj(response)

                if isinstance(response, UnknownResponse):
                    logger.exception(ValueError(f'Unknown request received: {request}'))
                    time.sleep(.1)

    def exit_later(self) -> None:
        """Request core exit without waiting.

        Sets state to Exiting, allowing current operations to complete.
        """
        self.state = CoreState.Exiting

    def exit(self) -> None:
        """Request exit and wait for experiment thread to finish."""
        self.exit_later()
        if self.experiment_thread:
            self.experiment_thread.join()
