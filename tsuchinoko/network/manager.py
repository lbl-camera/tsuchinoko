"""Network manager for ZMQ-based client-server communication."""

import time
from collections import defaultdict
from pickle import UnpicklingError
from queue import Queue, Empty
from typing import Any, Callable, Optional, Type, Union

import zmq
from zmq.error import ZMQError, Again
from loguru import logger

from tsuchinoko.config import NetworkConfig
from tsuchinoko.core import CoreState
from tsuchinoko.core.messages import (
    Message, ConnectRequest, StateRequest, PauseRequest, StartRequest,
    StopRequest, GetParametersRequest, SetParameterRequest,
    PartialDataRequest, FullDataRequest, MeasureRequest,
    PushGraphsRequest, PullGraphsRequest, ReplayRequest,
    ExitRequest, PushDataRequest, SetComputeMetricsRequest
)
from tsuchinoko.utils.threads import QThreadFutureIterator, invoke_as_event


class NetworkManager:
    """Manages ZMQ communication with the Tsuchinoko server.

    Handles socket lifecycle, message queuing, and callback dispatch.
    Runs the communication loop in a background thread.
    """

    def __init__(self, address: str = 'localhost', port: int = 5555,
                 socket_linger: int = 5, recv_timeout_ms: int = 5000):
        self.address = address
        self.port = port
        self.socket_linger = socket_linger
        self.recv_timeout_ms = recv_timeout_ms

        self.context: Optional[zmq.Context] = zmq.Context()
        self.socket: Optional[zmq.Socket] = None
        self.message_queue: Queue = Queue()
        self.callbacks: dict = defaultdict(list)

        self._update_thread: Optional[QThreadFutureIterator] = None
        self._state_getter: Optional[Callable[[], CoreState]] = None
        self._on_connection_lost: Optional[Callable[[], None]] = None
        self._on_connected: Optional[Callable[[], None]] = None

    @classmethod
    def from_config(cls, config: NetworkConfig) -> 'NetworkManager':
        """Create NetworkManager from configuration.

        Args:
            config: NetworkConfig instance

        Returns:
            Configured NetworkManager instance
        """
        return cls(
            address=config.address,
            port=config.port,
            socket_linger=config.socket_linger,
            recv_timeout_ms=config.recv_timeout_ms
        )

    def set_state_getter(self, getter: Callable[[], CoreState]):
        """Set a callback to get the current application state."""
        self._state_getter = getter

    def set_on_connection_lost(self, callback: Callable[[], None]):
        """Set callback invoked when connection to server is lost."""
        self._on_connection_lost = callback

    def set_on_connected(self, callback: Callable[[], None]):
        """Set callback invoked on successful connection."""
        self._on_connected = callback

    @property
    def current_state(self) -> CoreState:
        """Get current state from the state getter callback."""
        if self._state_getter:
            return self._state_getter()
        return CoreState.Connecting

    def init_socket(self):
        """Initialize or reinitialize the ZMQ socket."""
        if self.socket:
            logger.debug("Closing socket")
            self.socket.close()

        logger.info("Connecting to core server…")
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, self.socket_linger)
        self.socket.connect(f"tcp://{self.address}:{self.port}")
        self.socket.RCVTIMEO = self.recv_timeout_ms

    def start(self, finished_slot: Optional[Callable] = None):
        """Start the network communication thread."""
        self.init_socket()
        self._update_thread = QThreadFutureIterator(
            self._update_loop,
            finished_slot=finished_slot or self.close,
            name='tsuchinoko-network'
        )
        self._update_thread.start()

    def close(self):
        """Stop communication thread and clean up resources."""
        if self._update_thread and self._update_thread.running:
            logger.info('Waiting for network thread to finish')
            self._update_thread.requestInterruption()
            self._update_thread.wait()

        if self.socket:
            logger.debug('Closing socket')
            self.socket.close()
            self.socket = None

        if self.context:
            logger.debug('Closing context')
            self.context.term()
            self.context = None

    def subscribe(self, callback: Callable, response_type: Optional[Type[Message]] = None,
                  invoke_as_event: bool = False):
        """Subscribe to receive callbacks for a response type.

        Args:
            callback: Function to call when response is received
            response_type: Type of Message to subscribe to, or None for all
            invoke_as_event: If True, callback is invoked on Qt main thread
        """
        self.callbacks[response_type].append((callback, invoke_as_event))

    def unsubscribe(self, callback: Callable, response_type: Optional[Type[Message]] = None):
        """Unsubscribe a callback from a response type."""
        self.callbacks[response_type] = [
            (cb, as_event) for cb, as_event in self.callbacks[response_type]
            if cb != callback
        ]

    # Request methods - queue messages for the network thread
    def try_connect(self):
        self.message_queue.put(ConnectRequest())

    def get_state(self):
        self.message_queue.put(StateRequest())

    def pause(self):
        self.message_queue.put(PauseRequest())

    def start_experiment(self):
        self.message_queue.put(StartRequest())

    def stop(self):
        self.message_queue.put(StopRequest())

    def request_exit(self):
        self.message_queue.put(ExitRequest())

    def replay(self, positions, measurements):
        """Send replay request with provided data."""
        message = ReplayRequest(positions, measurements)
        self.message_queue.put(StopRequest())
        self.message_queue.put(message)
        self.message_queue.put(StartRequest())

    def set_compute_metrics(self, value: bool):
        self.message_queue.put(SetComputeMetricsRequest(value))

    def request_measure(self, pos):
        self.message_queue.put(MeasureRequest(pos))

    def request_parameters(self):
        self.message_queue.put(GetParametersRequest())

    def push_graph(self, graph):
        self.message_queue.put(PushGraphsRequest([graph]))

    def pull_graphs(self):
        self.message_queue.put(PullGraphsRequest())

    def push_data(self, data_dict: dict):
        self.message_queue.put(PushDataRequest(data_dict))

    def set_parameter(self, child_path: str, value: Any):
        if child_path:
            self.message_queue.put(SetParameterRequest(child_path, value))

    def request_full_data(self):
        self.message_queue.put(FullDataRequest())

    def request_partial_data(self, from_index: int):
        self.message_queue.put(PartialDataRequest(from_index))

    def clear_queue(self):
        """Clear all pending messages from the queue."""
        self.message_queue.queue.clear()

    def _update_loop(self):
        """Main communication loop - runs in background thread."""
        while True:
            yield  # Allow thread interruption check

            state = self.current_state
            request = None

            # Auto-connect when in connecting state
            if state == CoreState.Connecting:
                self.try_connect()

            # Poll for state during transitions
            if state in [CoreState.Pausing, CoreState.Starting,
                         CoreState.Resuming, CoreState.Stopping]:
                self.get_state()

            # Get next request from queue
            try:
                request = self.message_queue.get(timeout=0.2)
            except Empty:
                # Auto-request data when running
                if state == CoreState.Running:
                    request = self._create_data_request()

            if not request:
                self.get_state()
                continue

            logger.info(f'request: {request}')

            # Send request and handle response
            try:
                self.socket.send_pyobj(request)
                response = self.socket.recv_pyobj()
            except (ZMQError, Again):
                self._handle_connection_error()
            except UnpicklingError as ex:
                logger.exception(ex)
                logger.critical('The above error prevented unpacking data from the server.')
            else:
                self._handle_response(response)

    def _create_data_request(self) -> Optional[Message]:
        """Create appropriate data request based on current data state.

        Override or set callback to customize data request behavior.
        Returns FullDataRequest by default.
        """
        return FullDataRequest()

    def _handle_connection_error(self):
        """Handle connection failure to server."""
        logger.warning(f'Unable to connect to core server at {self.address}...')
        time.sleep(1)

        if self.context:
            self.init_socket()

        if self._on_connection_lost:
            self._on_connection_lost()

    def _handle_response(self, response):
        """Dispatch response to registered callbacks."""
        logger.info(f'response: {response}')

        if not response:
            self.get_state()
            return

        # Notify on first successful connection
        if self.current_state == CoreState.Connecting:
            logger.critical(f'Successfully connected to server at {self.address}.')
            if self._on_connected:
                self._on_connected()

        # Dispatch to callbacks
        for callback, as_event in self.callbacks[type(response)]:
            if as_event:
                invoke_as_event(callback, *response.payload)
            else:
                callback(*response.payload)
