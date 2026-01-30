# tests/test_zmq_queue.py
"""Tests for zmq_queue CustomQueue class."""
import pytest
import zmq
from unittest.mock import patch, MagicMock

from tsuchinoko.utils.zmq_queue import CustomQueue


class TestCustomQueueInit:
    """Tests for CustomQueue initialization."""

    def test_init_stores_parameters(self):
        """Test that init stores connection parameters."""
        with patch('tsuchinoko.utils.zmq_queue.zmq.Context') as MockContext:
            mock_context = MagicMock()
            mock_socket = MagicMock()
            MockContext.return_value = mock_context
            mock_context.socket.return_value = mock_socket

            q = CustomQueue(
                from_port=5551,
                to_port=5552,
                from_ip='localhost',
                to_ip='*',
                name='test',
                verbosity=0
            )

            assert q.from_port == 5551
            assert q.to_port == 5552
            assert q.from_ip == 'localhost'
            assert q.to_ip == '*'
            assert q.name == 'test'

    def test_init_creates_context(self):
        """Test that init creates ZMQ context."""
        with patch('tsuchinoko.utils.zmq_queue.zmq.Context') as MockContext:
            mock_context = MagicMock()
            mock_socket = MagicMock()
            MockContext.return_value = mock_context
            mock_context.socket.return_value = mock_socket

            q = CustomQueue(
                from_port=5551,
                to_port=5552,
                verbosity=0
            )

            assert q.context is not None
            MockContext.assert_called_once()

    def test_init_creates_push_socket(self):
        """Test that init creates and binds a PUSH socket."""
        with patch('tsuchinoko.utils.zmq_queue.zmq.Context') as MockContext:
            mock_context = MagicMock()
            mock_socket = MagicMock()
            MockContext.return_value = mock_context
            mock_context.socket.return_value = mock_socket

            q = CustomQueue(
                from_port=5551,
                to_port=5552,
                to_ip='*',
                verbosity=0
            )

            # Should create a socket with PUSH type
            mock_context.socket.assert_any_call(zmq.PUSH)
            # Should bind to the to_port
            mock_socket.bind.assert_called_once_with("tcp://*:5552")

    def test_init_creates_pull_socket(self):
        """Test that init creates and connects a PULL socket."""
        with patch('tsuchinoko.utils.zmq_queue.zmq.Context') as MockContext:
            mock_context = MagicMock()
            mock_socket = MagicMock()
            MockContext.return_value = mock_context
            mock_context.socket.return_value = mock_socket

            q = CustomQueue(
                from_port=5551,
                to_port=5552,
                from_ip='localhost',
                verbosity=0
            )

            # Should create a socket with PULL type
            mock_context.socket.assert_any_call(zmq.PULL)
            # Should connect to the from_port
            mock_socket.connect.assert_called_once_with("tcp://localhost:5551")

    def test_init_stores_save_dir(self):
        """Test that init stores save_dir parameter."""
        with patch('tsuchinoko.utils.zmq_queue.zmq.Context') as MockContext:
            mock_context = MagicMock()
            mock_socket = MagicMock()
            MockContext.return_value = mock_context
            mock_context.socket.return_value = mock_socket

            q = CustomQueue(
                from_port=5551,
                to_port=5552,
                save_dir='/custom/path',
                verbosity=0
            )

            assert q.save_dir == '/custom/path'

    def test_init_stores_kwargs(self):
        """Test that init stores extra kwargs."""
        with patch('tsuchinoko.utils.zmq_queue.zmq.Context') as MockContext:
            mock_context = MagicMock()
            mock_socket = MagicMock()
            MockContext.return_value = mock_context
            mock_context.socket.return_value = mock_socket

            q = CustomQueue(
                from_port=5551,
                to_port=5552,
                verbosity=0,
                extra_param='value'
            )

            assert q.kwargs == {'extra_param': 'value'}


class TestCustomQueueUtilities:
    """Tests for CustomQueue utility methods."""

    @pytest.fixture
    def mock_queue(self):
        """Create a CustomQueue with mocked sockets."""
        with patch('tsuchinoko.utils.zmq_queue.zmq.Context') as MockContext:
            mock_context = MagicMock()
            mock_socket = MagicMock()
            MockContext.return_value = mock_context
            mock_context.socket.return_value = mock_socket
            q = CustomQueue(
                from_port=5551,
                to_port=5552,
                verbosity=0
            )
            yield q

    def test_now_returns_string(self, mock_queue):
        """Test now() returns a formatted time string."""
        result = mock_queue.now()
        assert isinstance(result, str)
        # Should match format like '2026-01-29 12:34:56'
        assert len(result) == 19

    def test_time_str_formats_timestamp(self, mock_queue):
        """Test time_str() formats a Unix timestamp."""
        import time
        timestamp = time.time()
        result = mock_queue.time_str(timestamp)
        assert isinstance(result, str)
        assert len(result) == 19

    def test_time_delta_seconds(self, mock_queue):
        """Test time_delta() for seconds difference."""
        result = mock_queue.time_delta(0, 5)
        assert '5.0 s' in result
        assert 'later' in result

    def test_time_delta_minutes(self, mock_queue):
        """Test time_delta() for minutes difference."""
        result = mock_queue.time_delta(0, 120)  # 2 minutes
        assert 'minutes' in result

    def test_time_delta_hours(self, mock_queue):
        """Test time_delta() for hours difference."""
        result = mock_queue.time_delta(0, 7200)  # 2 hours
        assert 'hours' in result

    def test_time_delta_earlier(self, mock_queue):
        """Test time_delta() for negative difference."""
        result = mock_queue.time_delta(5, 0)
        assert 'earlier' in result

    def test_msg_prints_when_verbose(self, mock_queue, capsys):
        """Test msg() prints when verbosity is high enough."""
        mock_queue.verbosity = 5
        mock_queue.msg('test message', threshold=3)
        captured = capsys.readouterr()
        assert 'test message' in captured.out

    def test_msg_silent_when_not_verbose(self, mock_queue, capsys):
        """Test msg() is silent when verbosity is too low."""
        mock_queue.verbosity = 1
        mock_queue.msg('test message', threshold=3)
        captured = capsys.readouterr()
        assert 'test message' not in captured.out
