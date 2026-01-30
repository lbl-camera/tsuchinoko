"""Tests for async network manager."""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from tsuchinoko.network.async_manager import AsyncNetworkManager
from tsuchinoko.core.messages import (
    ConnectRequest, ConnectResponse, StateRequest, StateResponse,
    PauseRequest, StartRequest, StopRequest
)
from tsuchinoko.core import CoreState
from tsuchinoko.config import reset_config, get_config


@pytest.fixture(autouse=True)
def reset_config_fixture():
    """Reset config before each test."""
    reset_config()
    yield
    reset_config()


class TestAsyncNetworkManagerInit:
    """Tests for AsyncNetworkManager initialization."""

    def test_init_default_values(self):
        """Test AsyncNetworkManager uses config defaults."""
        manager = AsyncNetworkManager()
        config = get_config()
        assert manager.address == config.network.address
        assert manager.port == config.network.port
        assert manager.recv_timeout_ms == config.network.recv_timeout_ms
        assert manager.context is None
        assert manager.socket is None
        assert manager.connected is False

    def test_init_custom_values(self):
        """Test AsyncNetworkManager with custom values."""
        manager = AsyncNetworkManager(address='192.168.1.100', port=6666, recv_timeout_ms=10000)
        assert manager.address == '192.168.1.100'
        assert manager.port == 6666
        assert manager.recv_timeout_ms == 10000

    def test_init_partial_custom_values(self):
        """Test AsyncNetworkManager with partial custom values."""
        manager = AsyncNetworkManager(port=7777)
        config = get_config()
        assert manager.address == config.network.address
        assert manager.port == 7777
        assert manager.recv_timeout_ms == config.network.recv_timeout_ms

    def test_from_config(self):
        """Test creating AsyncNetworkManager from NetworkConfig."""
        from tsuchinoko.config import NetworkConfig
        config = NetworkConfig(address='test.server.com', port=8888, recv_timeout_ms=3000)
        manager = AsyncNetworkManager.from_config(config)
        assert manager.address == 'test.server.com'
        assert manager.port == 8888
        assert manager.recv_timeout_ms == 3000


class TestAsyncNetworkManagerConnect:
    """Tests for AsyncNetworkManager connection methods."""

    @pytest.mark.asyncio
    async def test_connect_creates_context_and_socket(self):
        """Test that connect creates ZMQ context and socket."""
        with patch('tsuchinoko.network.async_manager.zmq.asyncio.Context') as mock_ctx_class:
            mock_ctx = MagicMock()
            mock_socket = MagicMock()
            mock_ctx_class.return_value = mock_ctx
            mock_ctx.socket.return_value = mock_socket

            manager = AsyncNetworkManager()
            await manager.connect()

            mock_ctx_class.assert_called_once()
            mock_ctx.socket.assert_called_once()
            mock_socket.connect.assert_called_once_with('tcp://localhost:5555')
            assert manager.connected is True

    @pytest.mark.asyncio
    async def test_connect_skips_if_already_connected(self):
        """Test that connect skips if already connected."""
        with patch('tsuchinoko.network.async_manager.zmq.asyncio.Context') as mock_ctx_class:
            mock_ctx = MagicMock()
            mock_socket = MagicMock()
            mock_ctx_class.return_value = mock_ctx
            mock_ctx.socket.return_value = mock_socket

            manager = AsyncNetworkManager()
            await manager.connect()
            await manager.connect()  # Second call should skip

            # Should only be called once
            assert mock_ctx_class.call_count == 1

    @pytest.mark.asyncio
    async def test_close_cleans_up_resources(self):
        """Test that close properly cleans up socket and context."""
        with patch('tsuchinoko.network.async_manager.zmq.asyncio.Context') as mock_ctx_class:
            mock_ctx = MagicMock()
            mock_socket = MagicMock()
            mock_ctx_class.return_value = mock_ctx
            mock_ctx.socket.return_value = mock_socket

            manager = AsyncNetworkManager()
            await manager.connect()
            await manager.close()

            mock_socket.close.assert_called_once()
            mock_ctx.term.assert_called_once()
            assert manager.socket is None
            assert manager.context is None
            assert manager.connected is False

    @pytest.mark.asyncio
    async def test_close_handles_not_connected(self):
        """Test that close handles case when not connected."""
        manager = AsyncNetworkManager()
        # Should not raise
        await manager.close()
        assert manager.connected is False


class TestAsyncNetworkManagerSendRequest:
    """Tests for AsyncNetworkManager send_request method."""

    @pytest.mark.asyncio
    async def test_send_request_success(self):
        """Test async send_request method with successful response."""
        with patch('tsuchinoko.network.async_manager.zmq.asyncio.Context') as mock_ctx_class:
            mock_ctx = MagicMock()
            mock_socket = AsyncMock()
            mock_ctx_class.return_value = mock_ctx
            mock_ctx.socket.return_value = mock_socket

            expected_response = ConnectResponse(CoreState.Inactive, True)
            mock_socket.recv_pyobj = AsyncMock(return_value=expected_response)

            manager = AsyncNetworkManager()
            await manager.connect()

            response = await manager.send_request(ConnectRequest())

            mock_socket.send_pyobj.assert_called_once()
            assert isinstance(response, ConnectResponse)
            assert response.state == CoreState.Inactive

    @pytest.mark.asyncio
    async def test_send_request_auto_connects(self):
        """Test that send_request auto-connects if not connected."""
        with patch('tsuchinoko.network.async_manager.zmq.asyncio.Context') as mock_ctx_class:
            mock_ctx = MagicMock()
            mock_socket = AsyncMock()
            mock_ctx_class.return_value = mock_ctx
            mock_ctx.socket.return_value = mock_socket

            expected_response = ConnectResponse(CoreState.Inactive, True)
            mock_socket.recv_pyobj = AsyncMock(return_value=expected_response)

            manager = AsyncNetworkManager()
            assert manager.connected is False

            response = await manager.send_request(ConnectRequest())

            # Should have connected automatically
            assert manager.connected is True
            mock_socket.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_request_timeout(self):
        """Test that send_request raises TimeoutError on timeout."""
        with patch('tsuchinoko.network.async_manager.zmq.asyncio.Context') as mock_ctx_class:
            mock_ctx = MagicMock()
            mock_socket = AsyncMock()
            mock_ctx_class.return_value = mock_ctx
            mock_ctx.socket.return_value = mock_socket

            # Make recv_pyobj never complete (simulating timeout)
            async def never_complete():
                await asyncio.sleep(100)

            mock_socket.recv_pyobj = never_complete

            manager = AsyncNetworkManager(recv_timeout_ms=100)  # Short timeout
            await manager.connect()

            with pytest.raises(asyncio.TimeoutError):
                await manager.send_request(ConnectRequest())

    @pytest.mark.asyncio
    async def test_try_connect(self):
        """Test try_connect convenience method."""
        with patch('tsuchinoko.network.async_manager.zmq.asyncio.Context') as mock_ctx_class:
            mock_ctx = MagicMock()
            mock_socket = AsyncMock()
            mock_ctx_class.return_value = mock_ctx
            mock_ctx.socket.return_value = mock_socket

            expected_response = ConnectResponse(CoreState.Paused, False)
            mock_socket.recv_pyobj = AsyncMock(return_value=expected_response)

            manager = AsyncNetworkManager()
            response = await manager.try_connect()

            assert isinstance(response, ConnectResponse)
            assert response.state == CoreState.Paused
            assert response.compute_metrics is False


class TestAsyncNetworkManagerContextManager:
    """Tests for AsyncNetworkManager async context manager."""

    @pytest.mark.asyncio
    async def test_context_manager_connects_and_closes(self):
        """Test async context manager connects on entry and closes on exit."""
        with patch('tsuchinoko.network.async_manager.zmq.asyncio.Context') as mock_ctx_class:
            mock_ctx = MagicMock()
            mock_socket = AsyncMock()
            mock_ctx_class.return_value = mock_ctx
            mock_ctx.socket.return_value = mock_socket

            expected_response = ConnectResponse(CoreState.Running, True)
            mock_socket.recv_pyobj = AsyncMock(return_value=expected_response)

            async with AsyncNetworkManager() as manager:
                assert manager.connected is True
                response = await manager.send_request(ConnectRequest())
                assert isinstance(response, ConnectResponse)

            # After exit, should be closed
            mock_socket.close.assert_called_once()
            mock_ctx.term.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager_closes_on_exception(self):
        """Test async context manager closes even on exception."""
        with patch('tsuchinoko.network.async_manager.zmq.asyncio.Context') as mock_ctx_class:
            mock_ctx = MagicMock()
            mock_socket = AsyncMock()
            mock_ctx_class.return_value = mock_ctx
            mock_ctx.socket.return_value = mock_socket

            with pytest.raises(ValueError):
                async with AsyncNetworkManager() as manager:
                    assert manager.connected is True
                    raise ValueError("Test exception")

            # Should still close
            mock_socket.close.assert_called_once()
            mock_ctx.term.assert_called_once()


class TestAsyncNetworkManagerConnectedProperty:
    """Tests for connected property."""

    def test_connected_initially_false(self):
        """Test connected is False initially."""
        manager = AsyncNetworkManager()
        assert manager.connected is False

    @pytest.mark.asyncio
    async def test_connected_true_after_connect(self):
        """Test connected is True after connect."""
        with patch('tsuchinoko.network.async_manager.zmq.asyncio.Context') as mock_ctx_class:
            mock_ctx = MagicMock()
            mock_socket = MagicMock()
            mock_ctx_class.return_value = mock_ctx
            mock_ctx.socket.return_value = mock_socket

            manager = AsyncNetworkManager()
            await manager.connect()
            assert manager.connected is True

    @pytest.mark.asyncio
    async def test_connected_false_after_close(self):
        """Test connected is False after close."""
        with patch('tsuchinoko.network.async_manager.zmq.asyncio.Context') as mock_ctx_class:
            mock_ctx = MagicMock()
            mock_socket = MagicMock()
            mock_ctx_class.return_value = mock_ctx
            mock_ctx.socket.return_value = mock_socket

            manager = AsyncNetworkManager()
            await manager.connect()
            await manager.close()
            assert manager.connected is False


class TestAsyncRequestMethods:
    """Tests for typed async request methods."""

    @pytest.mark.asyncio
    async def test_async_get_state(self):
        """Test async get_state method."""
        with patch('tsuchinoko.network.async_manager.zmq.asyncio.Context') as mock_ctx_class:
            mock_ctx = MagicMock()
            mock_socket = AsyncMock()
            mock_ctx_class.return_value = mock_ctx
            mock_ctx.socket.return_value = mock_socket
            mock_socket.recv_pyobj = AsyncMock(
                return_value=StateResponse(CoreState.Running, True)
            )

            manager = AsyncNetworkManager()
            await manager.connect()

            state, compute_metrics = await manager.async_get_state()

            assert state == CoreState.Running
            assert compute_metrics is True
            # Verify a StateRequest was sent
            call_args = mock_socket.send_pyobj.call_args
            assert isinstance(call_args[0][0], StateRequest)

    @pytest.mark.asyncio
    async def test_async_pause(self):
        """Test async pause method."""
        with patch('tsuchinoko.network.async_manager.zmq.asyncio.Context') as mock_ctx_class:
            mock_ctx = MagicMock()
            mock_socket = AsyncMock()
            mock_ctx_class.return_value = mock_ctx
            mock_ctx.socket.return_value = mock_socket
            mock_socket.recv_pyobj = AsyncMock(
                return_value=StateResponse(CoreState.Pausing, False)
            )

            manager = AsyncNetworkManager()
            await manager.connect()

            state, compute_metrics = await manager.async_pause()

            assert state == CoreState.Pausing
            assert compute_metrics is False
            # Verify a PauseRequest was sent
            call_args = mock_socket.send_pyobj.call_args
            assert isinstance(call_args[0][0], PauseRequest)

    @pytest.mark.asyncio
    async def test_async_start(self):
        """Test async start method."""
        with patch('tsuchinoko.network.async_manager.zmq.asyncio.Context') as mock_ctx_class:
            mock_ctx = MagicMock()
            mock_socket = AsyncMock()
            mock_ctx_class.return_value = mock_ctx
            mock_ctx.socket.return_value = mock_socket
            mock_socket.recv_pyobj = AsyncMock(
                return_value=StateResponse(CoreState.Starting, True)
            )

            manager = AsyncNetworkManager()
            await manager.connect()

            state, compute_metrics = await manager.async_start()

            assert state == CoreState.Starting
            assert compute_metrics is True
            # Verify a StartRequest was sent
            call_args = mock_socket.send_pyobj.call_args
            assert isinstance(call_args[0][0], StartRequest)

    @pytest.mark.asyncio
    async def test_async_stop(self):
        """Test async stop method."""
        with patch('tsuchinoko.network.async_manager.zmq.asyncio.Context') as mock_ctx_class:
            mock_ctx = MagicMock()
            mock_socket = AsyncMock()
            mock_ctx_class.return_value = mock_ctx
            mock_ctx.socket.return_value = mock_socket
            mock_socket.recv_pyobj = AsyncMock(
                return_value=StateResponse(CoreState.Stopping, True)
            )

            manager = AsyncNetworkManager()
            await manager.connect()

            state, compute_metrics = await manager.async_stop()

            assert state == CoreState.Stopping
            assert compute_metrics is True
            # Verify a StopRequest was sent
            call_args = mock_socket.send_pyobj.call_args
            assert isinstance(call_args[0][0], StopRequest)

    @pytest.mark.asyncio
    async def test_async_connect_to_server(self):
        """Test async connect_to_server method."""
        with patch('tsuchinoko.network.async_manager.zmq.asyncio.Context') as mock_ctx_class:
            mock_ctx = MagicMock()
            mock_socket = AsyncMock()
            mock_ctx_class.return_value = mock_ctx
            mock_ctx.socket.return_value = mock_socket
            mock_socket.recv_pyobj = AsyncMock(
                return_value=ConnectResponse(CoreState.Inactive, False)
            )

            manager = AsyncNetworkManager()
            await manager.connect()

            state, compute_metrics = await manager.async_connect_to_server()

            assert state == CoreState.Inactive
            assert compute_metrics is False
            # Verify a ConnectRequest was sent
            call_args = mock_socket.send_pyobj.call_args
            assert isinstance(call_args[0][0], ConnectRequest)

    @pytest.mark.asyncio
    async def test_async_methods_auto_connect(self):
        """Test that async methods auto-connect if not connected."""
        with patch('tsuchinoko.network.async_manager.zmq.asyncio.Context') as mock_ctx_class:
            mock_ctx = MagicMock()
            mock_socket = AsyncMock()
            mock_ctx_class.return_value = mock_ctx
            mock_ctx.socket.return_value = mock_socket
            mock_socket.recv_pyobj = AsyncMock(
                return_value=StateResponse(CoreState.Running, True)
            )

            manager = AsyncNetworkManager()
            assert manager.connected is False

            # Call async method without explicitly connecting first
            await manager.async_get_state()

            # Should have auto-connected
            assert manager.connected is True
            mock_socket.connect.assert_called_once()
