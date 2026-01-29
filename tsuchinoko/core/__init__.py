import os
import sys
import threading
import time
from asyncio import events
from enum import Enum, auto
from pickle import UnpicklingError
from queue import Queue
from typing import List
from appdirs import user_state_dir

from loguru import logger
from yaml import dump

from tsuchinoko.graphs import Graph

from .messages import (
    Message, FullDataRequest, FullDataResponse, PartialDataRequest, PartialDataResponse,
    StartRequest, UnknownResponse, PauseRequest, StateRequest, GetParametersRequest,
    SetParameterRequest, GetParametersResponse, SetParameterResponse, StopRequest, StateResponse,
    MeasureRequest, MeasureResponse, ConnectRequest, ConnectResponse, ExceptionResponse,
    PushDataRequest, PushDataResponse, GraphsResponse, PullGraphsRequest, PushGraphsRequest,
    ReplayRequest, ReplayResponse, ExitRequest, SetComputeMetricsRequest
)
from ..adaptive import Engine as AdaptiveEngine, Data
from ..execution import Engine as ExecutionEngine
from ..utils.logging import log_time

user_state_dir = user_state_dir('tsuchinoko', 'camera')

class CoreState(Enum):
    """Enumeration of possible states for the experiment core.

    States represent the lifecycle of an experiment from connection
    through execution and termination.

    Attributes:
        Connecting: Initial state, attempting to establish connection
        Inactive: Connected but no experiment running
        Starting: Experiment initialization in progress
        Running: Experiment actively executing
        Pausing: Transitioning from Running to Paused
        Paused: Experiment temporarily halted, can resume
        Resuming: Transitioning from Paused to Running
        Stopping: Experiment termination in progress
        Restarting: Stopping then immediately Starting
        Exiting: Application shutdown in progress
    """
    Connecting = auto()
    Inactive = auto()
    Starting = auto()
    Running = auto()
    Pausing = auto()
    Paused = auto()
    Resuming = auto()
    Stopping = auto()
    Restarting = auto()
    Exiting = auto()


SLEEP_FOR_FRESH_DATA_TIME = .1


class Core:
    """Core experiment orchestrator managing adaptive optimization loops.

    The Core class coordinates between execution engines (which perform
    measurements) and adaptive engines (which determine optimal targets).
    It manages the experiment lifecycle through a state machine and
    handles data collection and checkpointing.

    The core runs an async event loop that monitors state transitions
    and delegates work to a background experiment thread when running.

    Attributes:
        execution_engine: Engine performing physical measurements
        adaptive_engine: Engine determining next measurement targets
        data: Accumulated experiment measurements and metrics
        compute_metrics: Whether to compute visualization metrics each iteration
    """
    # Import state machine lazily to avoid circular imports
    _state_machine_class = None

    @classmethod
    def _get_state_machine_class(cls):
        if cls._state_machine_class is None:
            from .state_machine import CoreStateMachine
            cls._state_machine_class = CoreStateMachine
        return cls._state_machine_class

    def __init__(self,
                 execution_engine: ExecutionEngine = None,
                 adaptive_engine: AdaptiveEngine = None,
                 compute_metrics: bool = True):
        """Initialize the experiment core.

        Args:
            execution_engine: Engine for performing measurements. Can be set
                later via set_execution_engine().
            adaptive_engine: Engine for determining targets. Can be set later
                via set_adaptive_engine().
            compute_metrics: If True, compute visualization metrics after
                each measurement update. Disable for faster execution.
        """
        self.execution_engine = execution_engine
        self.adaptive_engine = adaptive_engine

        self.iteration = 0

        # Initialize state machine
        self._state_machine = self._get_state_machine_class()()
        self._exception_queue = Queue()
        self._forced_position_queue = Queue()
        self._forced_measurement_queue = Queue()
        self._has_fresh_data = True
        self.compute_metrics = compute_metrics
        self.checkpoint_template = 'checkpoint_{n}.yml'
        self.checkpoint_at = []
        self.pause_at = []
        self.stop_at = []
        self.exit_at = []
        self.compute_metrics_at = []

        self.data = Data()
        self._graphs = []

        self.experiment_thread = None

    @property
    def state(self) -> CoreState:
        """Get current state from the state machine."""
        return self._state_machine.state

    @state.setter
    def state(self, value: CoreState):
        """Set state via state machine transitions.

        Maps direct state assignments to appropriate state machine triggers.
        This setter maintains backward compatibility while using the state machine.
        """
        current = self._state_machine.state

        # Map state assignments to triggers
        if value == current:
            return  # No change needed

        # Define transition mappings from current state to target state
        transition_map = {
            (CoreState.Inactive, CoreState.Starting): 'start',
            (CoreState.Starting, CoreState.Running): 'started',
            (CoreState.Running, CoreState.Pausing): 'pause',
            (CoreState.Pausing, CoreState.Paused): 'paused',
            (CoreState.Paused, CoreState.Resuming): 'resume',
            (CoreState.Resuming, CoreState.Running): 'resumed',
            (CoreState.Running, CoreState.Stopping): 'stop',
            (CoreState.Starting, CoreState.Stopping): 'stop',
            (CoreState.Pausing, CoreState.Stopping): 'stop',
            (CoreState.Paused, CoreState.Stopping): 'stop',
            (CoreState.Resuming, CoreState.Stopping): 'stop',
            (CoreState.Stopping, CoreState.Inactive): 'stopped',
        }

        # Handle exit from any state
        if value == CoreState.Exiting:
            self._state_machine.try_transition('exit')
            return

        key = (current, value)
        if key in transition_map:
            trigger = transition_map[key]
            if not self._state_machine.try_transition(trigger):
                logger.warning(f'State transition failed: {current} → {value}')
        else:
            logger.warning(f'No transition mapping for {current} → {value}')

    def set_execution_engine(self, engine: ExecutionEngine) -> None:
        """Set the execution engine for performing measurements.

        Args:
            engine: ExecutionEngine instance to use for measurements
        """
        self.execution_engine = engine

    def set_adaptive_engine(self, engine: AdaptiveEngine) -> None:
        """Set the adaptive engine for determining targets.

        Args:
            engine: AdaptiveEngine instance for optimization
        """
        self.adaptive_engine = engine

    def main(self, debug: bool = False) -> None:
        """Run the main async event loop.

        Creates a new event loop and runs the _main() coroutine until
        the core state becomes Exiting.

        Args:
            debug: If True, enable asyncio debug mode
        """
        loop = events.new_event_loop()  # <---- this ensures the current loop is replaced
        try:
            events.set_event_loop(loop)
            loop.set_debug(debug)
            loop.run_until_complete(self._main())
        finally:
            try:
                # _cancel_all_tasks(loop)
                loop.run_until_complete(loop.shutdown_asyncgens())
            finally:
                events.set_event_loop(None)
                loop.close()

    async def _main(self, min_response_sleep: float = .1) -> None:
        while self.state != CoreState.Exiting:

            if self.state == CoreState.Running:
                pass
                # await sleep(min_response_sleep)  # short-circuit case
            elif self.state == CoreState.Starting:
                if not len(self.data):
                    self.data = Data(dimensionality=self.adaptive_engine.dimensionality)
                self.adaptive_engine.reset()
                self.experiment_thread = threading.Thread(target=self.experiment_loop, args=())  # must hold ref
                self.experiment_thread.start()
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
                self.data = Data()
                # await sleep(min_response_sleep)

            if self.state not in [CoreState.Stopping, CoreState.Exiting, CoreState.Resuming, CoreState.Restarting]:
                await self.notify_clients()

    def experiment_loop(self) -> None:
        """Background thread running the experiment iteration loop.

        Continuously calls experiment_iteration() while in Running state.
        Handles checkpointing, pause triggers, and exception recovery.
        Exits when state becomes Stopping, Inactive, or Exiting.
        """
        while True:
            if self.state == CoreState.Running:
                logger.info(f'Iteration: {self.data._completed_iterations}, Data count: {len(self.data)}')
                if self.data._completed_iterations in self.checkpoint_at:
                    self.save_checkpoint()
                if self.data._completed_iterations in self.pause_at:
                    self.state = CoreState.Pausing
                    continue
                if self.data._completed_iterations in self.stop_at:
                    self.state = CoreState.Stopping
                    return
                if self.data._completed_iterations in self.exit_at:
                    self.state = CoreState.Exiting
                    return
                try:
                    self.experiment_iteration()
                except Exception as ex:
                    self._exception_queue.put(ex)
                    self.state = CoreState.Pausing
                    logger.exception(ex)
            elif self.state in [CoreState.Stopping, CoreState.Inactive, CoreState.Exiting]:
                return
            else:
                time.sleep(.1)

    def experiment_iteration(self) -> None:
        """Execute a single experiment iteration.

        One iteration consists of:
        1. Get current position from execution engine
        2. Request optimal targets from adaptive engine
        3. Update execution engine with new targets
        4. Collect measurements from execution engine
        5. Update adaptive engine with new data
        6. Optionally compute visualization metrics
        7. Train the adaptive model
        """
        with self.data.iteration():
            if self._has_fresh_data:
                with log_time('getting position', cumulative_key='getting position'):
                    position = self.execution_engine.get_position()
                    logger.info(f'position: {position}')
                    if position is None:
                        position = [0] * self.data.dimensionality
                    position = tuple(position)
                if self._forced_position_queue.empty():
                    with log_time('getting targets', cumulative_key='getting targets'):
                        targets = self.adaptive_engine.request_targets(position)
                    logger.info(f'targets: {targets}')
                else:
                    targets = [self._forced_position_queue.get()]

            if self._forced_measurement_queue.empty():
                if self._has_fresh_data:
                    with log_time('updating targets', cumulative_key='updating targets'):
                        self.execution_engine.update_targets(targets)
                    self._has_fresh_data = False
                with log_time('getting measurements', cumulative_key='getting measurements'):
                    new_measurements = self.execution_engine.get_measurements()
                logger.info(f'new measurements: {new_measurements}')
            else:
                new_measurements = [self._forced_measurement_queue.get()]
            if len(new_measurements):
                self._has_fresh_data = True
                with log_time('stashing new measurements', cumulative_key='injecting new measurements'):
                    self.data.inject_new(new_measurements)
                with log_time('updating engine with new measurements', cumulative_key='updating engine with new measurements'):
                    self.adaptive_engine.update_measurements(self.data)
                if self.compute_metrics or len(self.data) in self.compute_metrics_at:
                    with log_time('updating metrics', cumulative_key='updating metrics'):
                        self.adaptive_engine.update_metrics(self.data)
            else:
                time.sleep(SLEEP_FOR_FRESH_DATA_TIME)
            if self._has_fresh_data:
                with log_time('training', cumulative_key='training'):
                    self.adaptive_engine.train()
            else:
                logger.info('Current data is stale. Waiting for an update with fresh data.')

    async def notify_clients(self) -> None:
        ...

    @property
    def graphs(self) -> List[Graph]:
        execution_graphs = getattr(self.execution_engine, 'graphs', []) or []
        adaptive_graphs = getattr(self.adaptive_engine, 'graphs', []) or []
        return execution_graphs + adaptive_graphs + self._graphs

    @graphs.setter
    def graphs(self, graphs: List[Graph]) -> None:
        raise NotImplementedError('Updating graphs on server not supported yet.')

    def update_graph(self, new_graph: Graph) -> None:
        """Update a graph configuration by ID.

        Searches for a graph with matching ID in execution, adaptive,
        and local graph lists, then replaces it with the new version.

        Args:
            new_graph: Graph instance with updated configuration

        Raises:
            ValueError: If no graph with matching ID is found
        """
        execution_graphs = getattr(self.execution_engine, 'graphs', []) or []
        adaptive_graphs = getattr(self.adaptive_engine, 'graphs', []) or []
        self_graphs = self._graphs

        for graph_list in [execution_graphs, adaptive_graphs, self_graphs]:
            for i, old_graph in enumerate(graph_list):
                if old_graph.id == new_graph.id:
                    graph_list[i] = new_graph
                    return
        else:
            raise ValueError('Graph not found in graphs lists.')

    def initialize_data(self, x: List[tuple], y: List[float], v: List[float]) -> None:
        """Initialize the experiment with pre-existing data.

        Used to seed the adaptive engine with historical measurements
        before starting a new experiment run.

        Args:
            x: List of position tuples
            y: List of score/objective values
            v: List of variance values
        """
        with log_time('updating engine with initial measurements'):
            self.data = Data(dimensionality=len(x[0]), positions=x, scores=y, variances=v)
            self.adaptive_engine.update_measurements(self.data)

    def save_checkpoint(self, directory: str = user_state_dir) -> None:
        """Save current data state to a checkpoint file.

        Creates a YAML file with the current data dictionary.
        Filename follows checkpoint_template with iteration number.

        Args:
            directory: Directory to save checkpoint files
        """
        checkpoint_file_path = os.path.join(directory,
                                            self.checkpoint_template.format(n=self.data._completed_iterations))
        os.makedirs(os.path.dirname(checkpoint_file_path), exist_ok=True)
        dump(self.data.as_dict(), open(checkpoint_file_path, 'w'))


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
        # self.start_server()
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

    def respond_PullGraphsRequest(self, request: PullGraphsRequest) -> GraphsResponse:
        return GraphsResponse(self.graphs)

    def respond_PushGraphsRequest(self, request: PushGraphsRequest) -> Message:
        for graph in request.graphs:
            try:
                self.update_graph(graph)
            except ValueError as ex:
                return ExceptionResponse("Graph ID not found in server's graphs.")
        # self.graphs = request.graphs
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
        self.experiment_thread.join()
