# tests/test_zmq_queue.py
"""Tests for zmq_queue CustomQueue class."""
import pytest
from unittest.mock import patch, MagicMock


class TestCustomQueueInit:
    """Tests for CustomQueue initialization."""

    def test_init_stores_parameters(self):
        """Test that init stores connection parameters."""
        with patch('tsuchinoko.utils.zmq_queue.zmq.Context') as MockContext:
            mock_context = MagicMock()
            mock_socket = MagicMock()
            MockContext.return_value = mock_context
            mock_context.socket.return_value = mock_socket

            from tsuchinoko.utils.zmq_queue import CustomQueue

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

            from tsuchinoko.utils.zmq_queue import CustomQueue

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
            with patch('tsuchinoko.utils.zmq_queue.zmq.PUSH', 8):  # ZMQ PUSH constant
                mock_context = MagicMock()
                mock_socket = MagicMock()
                MockContext.return_value = mock_context
                mock_context.socket.return_value = mock_socket

                from tsuchinoko.utils.zmq_queue import CustomQueue

                q = CustomQueue(
                    from_port=5551,
                    to_port=5552,
                    to_ip='*',
                    verbosity=0
                )

                # Should create a socket with PUSH type
                mock_context.socket.assert_any_call(8)
                # Should bind to the to_port
                mock_socket.bind.assert_called_once_with("tcp://*:5552")

    def test_init_creates_pull_socket(self):
        """Test that init creates and connects a PULL socket."""
        with patch('tsuchinoko.utils.zmq_queue.zmq.Context') as MockContext:
            with patch('tsuchinoko.utils.zmq_queue.zmq.PUSH', 8):
                with patch('tsuchinoko.utils.zmq_queue.zmq.PULL', 7):  # ZMQ PULL constant
                    mock_context = MagicMock()
                    mock_socket = MagicMock()
                    MockContext.return_value = mock_context
                    mock_context.socket.return_value = mock_socket

                    from tsuchinoko.utils.zmq_queue import CustomQueue

                    q = CustomQueue(
                        from_port=5551,
                        to_port=5552,
                        from_ip='localhost',
                        verbosity=0
                    )

                    # Should create a socket with PULL type
                    mock_context.socket.assert_any_call(7)
                    # Should connect to the from_port
                    mock_socket.connect.assert_called_once_with("tcp://localhost:5551")

    def test_init_stores_save_dir(self):
        """Test that init stores save_dir parameter."""
        with patch('tsuchinoko.utils.zmq_queue.zmq.Context') as MockContext:
            mock_context = MagicMock()
            mock_socket = MagicMock()
            MockContext.return_value = mock_context
            mock_context.socket.return_value = mock_socket

            from tsuchinoko.utils.zmq_queue import CustomQueue

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

            from tsuchinoko.utils.zmq_queue import CustomQueue

            q = CustomQueue(
                from_port=5551,
                to_port=5552,
                verbosity=0,
                extra_param='value'
            )

            assert q.kwargs == {'extra_param': 'value'}
