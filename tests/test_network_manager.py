# tests/test_network_manager.py
"""Tests for NetworkManager class."""
import pytest
from unittest.mock import MagicMock, patch, call
from queue import Empty

from tsuchinoko.network import NetworkManager
from tsuchinoko.core import CoreState
from tsuchinoko.core.messages import (
    ConnectRequest, StateRequest, PauseRequest, StartRequest, StopRequest,
    GetParametersRequest, SetParameterRequest, FullDataRequest, PartialDataRequest,
    ExitRequest, MeasureRequest, PushGraphsRequest, PullGraphsRequest,
    ReplayRequest, PushDataRequest, SetComputeMetricsRequest
)


@pytest.fixture
def network_manager():
    """Create NetworkManager with mocked ZMQ components."""
    with patch('tsuchinoko.network.manager.zmq.Context') as MockContext:
        mock_context = MagicMock()
        mock_socket = MagicMock()
        MockContext.return_value = mock_context
        mock_context.socket.return_value = mock_socket

        manager = NetworkManager(address='localhost', port=5555)
        # Replace the context and socket with mocks for testing
        manager.context = mock_context
        manager.socket = mock_socket
        yield manager


@pytest.fixture
def network_manager_no_context():
    """Create NetworkManager without initializing context (for init tests)."""
    with patch('tsuchinoko.network.manager.zmq.Context') as MockContext:
        mock_context = MagicMock()
        mock_socket = MagicMock()
        MockContext.return_value = mock_context
        mock_context.socket.return_value = mock_socket

        manager = NetworkManager(address='testhost', port=9999)
        yield manager, mock_context, mock_socket


class TestNetworkManagerInit:
    """Tests for NetworkManager initialization."""

    def test_init_stores_address(self, network_manager_no_context):
        """Test that init stores the address parameter."""
        manager, _, _ = network_manager_no_context
        assert manager.address == 'testhost'

    def test_init_stores_port(self, network_manager_no_context):
        """Test that init stores the port parameter."""
        manager, _, _ = network_manager_no_context
        assert manager.port == 9999

    def test_init_stores_socket_linger(self):
        """Test that init stores socket_linger parameter."""
        with patch('tsuchinoko.network.manager.zmq.Context'):
            manager = NetworkManager(socket_linger=10)
            assert manager.socket_linger == 10

    def test_init_stores_recv_timeout(self):
        """Test that init stores recv_timeout_ms parameter."""
        with patch('tsuchinoko.network.manager.zmq.Context'):
            manager = NetworkManager(recv_timeout_ms=10000)
            assert manager.recv_timeout_ms == 10000

    def test_init_creates_context(self, network_manager_no_context):
        """Test that init creates ZMQ context."""
        manager, mock_context, _ = network_manager_no_context
        assert manager.context is not None

    def test_init_creates_message_queue(self, network_manager_no_context):
        """Test that init creates empty message queue."""
        manager, _, _ = network_manager_no_context
        assert manager.message_queue is not None
        assert manager.message_queue.empty()

    def test_init_creates_callbacks_dict(self, network_manager_no_context):
        """Test that init creates empty callbacks dictionary."""
        manager, _, _ = network_manager_no_context
        assert manager.callbacks is not None
        assert len(manager.callbacks) == 0

    def test_init_socket_is_none(self, network_manager_no_context):
        """Test that socket starts as None before start()."""
        manager, _, _ = network_manager_no_context
        assert manager.socket is None

    def test_init_update_thread_is_none(self, network_manager_no_context):
        """Test that update thread starts as None."""
        manager, _, _ = network_manager_no_context
        assert manager._update_thread is None


class TestNetworkManagerQueue:
    """Tests for NetworkManager message queue operations."""

    def test_try_connect_queues_request(self, network_manager):
        """Test try_connect() queues a ConnectRequest."""
        network_manager.try_connect()
        request = network_manager.message_queue.get_nowait()
        assert isinstance(request, ConnectRequest)

    def test_get_state_queues_request(self, network_manager):
        """Test get_state() queues a StateRequest."""
        network_manager.get_state()
        request = network_manager.message_queue.get_nowait()
        assert isinstance(request, StateRequest)

    def test_pause_queues_request(self, network_manager):
        """Test pause() queues a PauseRequest."""
        network_manager.pause()
        request = network_manager.message_queue.get_nowait()
        assert isinstance(request, PauseRequest)

    def test_start_experiment_queues_request(self, network_manager):
        """Test start_experiment() queues a StartRequest."""
        network_manager.start_experiment()
        request = network_manager.message_queue.get_nowait()
        assert isinstance(request, StartRequest)

    def test_stop_queues_request(self, network_manager):
        """Test stop() queues a StopRequest."""
        network_manager.stop()
        request = network_manager.message_queue.get_nowait()
        assert isinstance(request, StopRequest)

    def test_request_exit_queues_request(self, network_manager):
        """Test request_exit() queues an ExitRequest."""
        network_manager.request_exit()
        request = network_manager.message_queue.get_nowait()
        assert isinstance(request, ExitRequest)

    def test_request_parameters_queues_request(self, network_manager):
        """Test request_parameters() queues a GetParametersRequest."""
        network_manager.request_parameters()
        request = network_manager.message_queue.get_nowait()
        assert isinstance(request, GetParametersRequest)

    def test_set_parameter_queues_request(self, network_manager):
        """Test set_parameter() queues a SetParameterRequest with correct values."""
        network_manager.set_parameter('path.to.param', 42)
        request = network_manager.message_queue.get_nowait()
        assert isinstance(request, SetParameterRequest)
        assert request.child_path == 'path.to.param'
        assert request.value == 42

    def test_set_parameter_empty_path_does_not_queue(self, network_manager):
        """Test set_parameter() with empty path does not queue anything."""
        network_manager.set_parameter('', 42)
        with pytest.raises(Empty):
            network_manager.message_queue.get_nowait()

    def test_set_parameter_none_path_does_not_queue(self, network_manager):
        """Test set_parameter() with None path does not queue anything."""
        network_manager.set_parameter(None, 42)
        with pytest.raises(Empty):
            network_manager.message_queue.get_nowait()

    def test_request_full_data_queues_request(self, network_manager):
        """Test request_full_data() queues a FullDataRequest."""
        network_manager.request_full_data()
        request = network_manager.message_queue.get_nowait()
        assert isinstance(request, FullDataRequest)

    def test_request_partial_data_queues_request(self, network_manager):
        """Test request_partial_data() queues a PartialDataRequest with from_index."""
        network_manager.request_partial_data(100)
        request = network_manager.message_queue.get_nowait()
        assert isinstance(request, PartialDataRequest)
        assert request.iteration == 100

    def test_request_measure_queues_request(self, network_manager):
        """Test request_measure() queues a MeasureRequest with position."""
        network_manager.request_measure((1.0, 2.0, 3.0))
        request = network_manager.message_queue.get_nowait()
        assert isinstance(request, MeasureRequest)
        assert request.position == (1.0, 2.0, 3.0)

    def test_push_graph_queues_request(self, network_manager):
        """Test push_graph() queues a PushGraphsRequest with graph list."""
        mock_graph = MagicMock()
        network_manager.push_graph(mock_graph)
        request = network_manager.message_queue.get_nowait()
        assert isinstance(request, PushGraphsRequest)
        assert request.graphs == [mock_graph]

    def test_pull_graphs_queues_request(self, network_manager):
        """Test pull_graphs() queues a PullGraphsRequest."""
        network_manager.pull_graphs()
        request = network_manager.message_queue.get_nowait()
        assert isinstance(request, PullGraphsRequest)

    def test_push_data_queues_request(self, network_manager):
        """Test push_data() queues a PushDataRequest with data dict."""
        data_dict = {'positions': [[1, 2]], 'scores': [0.5]}
        network_manager.push_data(data_dict)
        request = network_manager.message_queue.get_nowait()
        assert isinstance(request, PushDataRequest)
        assert request.data == data_dict

    def test_set_compute_metrics_queues_request(self, network_manager):
        """Test set_compute_metrics() queues a SetComputeMetricsRequest."""
        network_manager.set_compute_metrics(True)
        request = network_manager.message_queue.get_nowait()
        assert isinstance(request, SetComputeMetricsRequest)
        assert request.compute_metrics is True

    def test_set_compute_metrics_false(self, network_manager):
        """Test set_compute_metrics(False) queues request with False value."""
        network_manager.set_compute_metrics(False)
        request = network_manager.message_queue.get_nowait()
        assert isinstance(request, SetComputeMetricsRequest)
        assert request.compute_metrics is False

    def test_clear_queue_removes_all_messages(self, network_manager):
        """Test clear_queue() removes all pending messages."""
        # Queue several messages
        network_manager.try_connect()
        network_manager.get_state()
        network_manager.pause()
        assert not network_manager.message_queue.empty()

        # Clear the queue
        network_manager.clear_queue()

        # Queue should now be empty
        with pytest.raises(Empty):
            network_manager.message_queue.get_nowait()

    def test_multiple_messages_preserve_order(self, network_manager):
        """Test that multiple queued messages are retrieved in FIFO order."""
        network_manager.try_connect()
        network_manager.get_state()
        network_manager.pause()
        network_manager.start_experiment()

        # Verify FIFO order
        assert isinstance(network_manager.message_queue.get_nowait(), ConnectRequest)
        assert isinstance(network_manager.message_queue.get_nowait(), StateRequest)
        assert isinstance(network_manager.message_queue.get_nowait(), PauseRequest)
        assert isinstance(network_manager.message_queue.get_nowait(), StartRequest)


class TestNetworkManagerReplay:
    """Tests for NetworkManager replay functionality."""

    def test_replay_queues_stop_replay_start(self, network_manager):
        """Test replay() queues Stop, Replay, Start requests in order."""
        positions = [(0, 0), (1, 1)]
        measurements = [0.5, 0.8]

        network_manager.replay(positions, measurements)

        # Should queue: StopRequest, ReplayRequest, StartRequest
        req1 = network_manager.message_queue.get_nowait()
        req2 = network_manager.message_queue.get_nowait()
        req3 = network_manager.message_queue.get_nowait()

        assert isinstance(req1, StopRequest)
        assert isinstance(req2, ReplayRequest)
        assert isinstance(req3, StartRequest)

    def test_replay_request_contains_data(self, network_manager):
        """Test ReplayRequest contains the provided positions and measurements."""
        positions = [(0, 0), (1, 1), (2, 2)]
        measurements = [0.1, 0.2, 0.3]

        network_manager.replay(positions, measurements)

        # Skip StopRequest
        network_manager.message_queue.get_nowait()
        # Get ReplayRequest
        replay_req = network_manager.message_queue.get_nowait()

        assert replay_req.positions == positions
        assert replay_req.measurements == measurements


class TestNetworkManagerCallbacks:
    """Tests for NetworkManager callback subscription."""

    def test_subscribe_adds_callback(self, network_manager):
        """Test subscribe() adds callback to the list."""
        callback = MagicMock()
        network_manager.subscribe(callback, StateRequest)

        assert (callback, False) in network_manager.callbacks[StateRequest]

    def test_subscribe_with_invoke_as_event(self, network_manager):
        """Test subscribe() with invoke_as_event=True."""
        callback = MagicMock()
        network_manager.subscribe(callback, StateRequest, invoke_as_event=True)

        assert (callback, True) in network_manager.callbacks[StateRequest]

    def test_subscribe_none_type_for_all(self, network_manager):
        """Test subscribe() with None type subscribes to all responses."""
        callback = MagicMock()
        network_manager.subscribe(callback, None)

        assert (callback, False) in network_manager.callbacks[None]

    def test_unsubscribe_removes_callback(self, network_manager):
        """Test unsubscribe() removes the callback."""
        callback = MagicMock()
        network_manager.subscribe(callback, StateRequest)
        network_manager.unsubscribe(callback, StateRequest)

        assert (callback, False) not in network_manager.callbacks[StateRequest]

    def test_unsubscribe_keeps_other_callbacks(self, network_manager):
        """Test unsubscribe() keeps other callbacks intact."""
        callback1 = MagicMock()
        callback2 = MagicMock()
        network_manager.subscribe(callback1, StateRequest)
        network_manager.subscribe(callback2, StateRequest)

        network_manager.unsubscribe(callback1, StateRequest)

        assert (callback1, False) not in network_manager.callbacks[StateRequest]
        assert (callback2, False) in network_manager.callbacks[StateRequest]

    def test_multiple_subscriptions_same_type(self, network_manager):
        """Test multiple callbacks can subscribe to the same type."""
        callback1 = MagicMock()
        callback2 = MagicMock()

        network_manager.subscribe(callback1, StateRequest)
        network_manager.subscribe(callback2, StateRequest)

        assert len(network_manager.callbacks[StateRequest]) == 2


class TestNetworkManagerState:
    """Tests for NetworkManager state management."""

    def test_set_state_getter(self, network_manager):
        """Test set_state_getter() stores the callback."""
        getter = MagicMock(return_value=CoreState.Running)
        network_manager.set_state_getter(getter)

        assert network_manager._state_getter is getter

    def test_current_state_uses_getter(self, network_manager):
        """Test current_state property calls the state getter."""
        getter = MagicMock(return_value=CoreState.Running)
        network_manager.set_state_getter(getter)

        state = network_manager.current_state

        getter.assert_called_once()
        assert state == CoreState.Running

    def test_current_state_default_connecting(self, network_manager):
        """Test current_state returns Connecting when no getter is set."""
        state = network_manager.current_state
        assert state == CoreState.Connecting

    def test_set_on_connection_lost(self, network_manager):
        """Test set_on_connection_lost() stores the callback."""
        callback = MagicMock()
        network_manager.set_on_connection_lost(callback)

        assert network_manager._on_connection_lost is callback

    def test_set_on_connected(self, network_manager):
        """Test set_on_connected() stores the callback."""
        callback = MagicMock()
        network_manager.set_on_connected(callback)

        assert network_manager._on_connected is callback


class TestNetworkManagerSocket:
    """Tests for NetworkManager socket initialization."""

    def test_init_socket_creates_req_socket(self):
        """Test init_socket() creates a REQ socket."""
        import zmq
        with patch('tsuchinoko.network.manager.zmq.Context') as MockContext:
            mock_context = MagicMock()
            mock_socket = MagicMock()
            MockContext.return_value = mock_context
            mock_context.socket.return_value = mock_socket

            manager = NetworkManager(address='localhost', port=5555)
            manager.init_socket()

            mock_context.socket.assert_called_with(zmq.REQ)

    def test_init_socket_sets_linger(self):
        """Test init_socket() sets socket linger option."""
        import zmq
        with patch('tsuchinoko.network.manager.zmq.Context') as MockContext:
            mock_context = MagicMock()
            mock_socket = MagicMock()
            MockContext.return_value = mock_context
            mock_context.socket.return_value = mock_socket

            manager = NetworkManager(address='localhost', port=5555, socket_linger=10)
            manager.init_socket()

            mock_socket.setsockopt.assert_called_with(zmq.LINGER, 10)

    def test_init_socket_connects_to_address(self):
        """Test init_socket() connects to the configured address."""
        with patch('tsuchinoko.network.manager.zmq.Context') as MockContext:
            mock_context = MagicMock()
            mock_socket = MagicMock()
            MockContext.return_value = mock_context
            mock_context.socket.return_value = mock_socket

            manager = NetworkManager(address='testhost', port=9999)
            manager.init_socket()

            mock_socket.connect.assert_called_with('tcp://testhost:9999')

    def test_init_socket_sets_recv_timeout(self):
        """Test init_socket() sets receive timeout."""
        with patch('tsuchinoko.network.manager.zmq.Context') as MockContext:
            mock_context = MagicMock()
            mock_socket = MagicMock()
            MockContext.return_value = mock_context
            mock_context.socket.return_value = mock_socket

            manager = NetworkManager(recv_timeout_ms=8000)
            manager.init_socket()

            assert mock_socket.RCVTIMEO == 8000

    def test_init_socket_closes_existing_socket(self):
        """Test init_socket() closes existing socket before creating new one."""
        with patch('tsuchinoko.network.manager.zmq.Context') as MockContext:
            mock_context = MagicMock()
            mock_socket_old = MagicMock()
            mock_socket_new = MagicMock()
            MockContext.return_value = mock_context
            mock_context.socket.side_effect = [mock_socket_old, mock_socket_new]

            manager = NetworkManager()
            manager.init_socket()  # Creates first socket
            manager.init_socket()  # Should close first, create second

            mock_socket_old.close.assert_called_once()


class TestNetworkManagerClose:
    """Tests for NetworkManager cleanup."""

    def test_close_closes_socket(self, network_manager):
        """Test close() closes the socket."""
        mock_socket = network_manager.socket
        network_manager.close()
        mock_socket.close.assert_called()

    def test_close_terminates_context(self, network_manager):
        """Test close() terminates the ZMQ context."""
        mock_context = network_manager.context
        network_manager.close()
        mock_context.term.assert_called()

    def test_close_sets_socket_to_none(self, network_manager):
        """Test close() sets socket to None."""
        network_manager.close()
        assert network_manager.socket is None

    def test_close_sets_context_to_none(self, network_manager):
        """Test close() sets context to None."""
        network_manager.close()
        assert network_manager.context is None

    def test_close_requests_thread_interruption(self):
        """Test close() requests thread interruption if thread is running."""
        with patch('tsuchinoko.network.manager.zmq.Context') as MockContext:
            mock_context = MagicMock()
            mock_socket = MagicMock()
            MockContext.return_value = mock_context
            mock_context.socket.return_value = mock_socket

            manager = NetworkManager()
            manager.socket = mock_socket
            manager.context = mock_context

            # Create a mock thread that appears to be running
            mock_thread = MagicMock()
            mock_thread.running = True
            manager._update_thread = mock_thread

            manager.close()

            mock_thread.requestInterruption.assert_called_once()
            mock_thread.wait.assert_called_once()


class TestNetworkManagerCreateDataRequest:
    """Tests for NetworkManager data request creation."""

    def test_create_data_request_returns_full_data_request(self, network_manager):
        """Test _create_data_request() returns FullDataRequest by default."""
        request = network_manager._create_data_request()
        assert isinstance(request, FullDataRequest)
