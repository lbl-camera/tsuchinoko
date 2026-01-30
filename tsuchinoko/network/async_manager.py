"""Async network manager using zmq.asyncio.

Provides non-blocking network communication with the Tsuchinoko server
using ZMQ's asyncio integration for seamless async/await patterns.
"""
import asyncio
from typing import Optional, Tuple

import zmq
import zmq.asyncio
from loguru import logger

from tsuchinoko.core import CoreState
from tsuchinoko.core.messages import (
    Message, ConnectRequest, ConnectResponse, StateRequest, StateResponse,
    PauseRequest, StartRequest, StopRequest
)
from tsuchinoko.config import get_config, NetworkConfig


class AsyncNetworkManager:
    """Async ZMQ network manager.

    Provides non-blocking network communication using zmq.asyncio.
    This class is intended for use in async contexts where blocking
    operations would interfere with the event loop.

    Attributes:
        address: Server hostname or IP address
        port: Server port number
        recv_timeout_ms: Timeout for receiving responses in milliseconds
        context: ZMQ asyncio context (created on connect)
        socket: ZMQ asyncio socket (created on connect)
    """

    def __init__(self, address: Optional[str] = None, port: Optional[int] = None,
                 recv_timeout_ms: Optional[int] = None):
        """Initialize async network manager.

        Args:
            address: Server address (defaults to config value)
            port: Server port (defaults to config value)
            recv_timeout_ms: Receive timeout in ms (defaults to config value)
        """
        config = get_config()
        self.address = address if address is not None else config.network.address
        self.port = port if port is not None else config.network.port
        self.recv_timeout_ms = recv_timeout_ms if recv_timeout_ms is not None else config.network.recv_timeout_ms

        self.context: Optional[zmq.asyncio.Context] = None
        self.socket: Optional[zmq.asyncio.Socket] = None
        self._connected = False

    @classmethod
    def from_config(cls, config: NetworkConfig) -> 'AsyncNetworkManager':
        """Create AsyncNetworkManager from configuration.

        Args:
            config: NetworkConfig instance

        Returns:
            Configured AsyncNetworkManager instance
        """
        return cls(
            address=config.address,
            port=config.port,
            recv_timeout_ms=config.recv_timeout_ms
        )

    async def connect(self) -> None:
        """Connect to the server asynchronously.

        Creates a new ZMQ context and REQ socket, then connects
        to the configured server address and port.

        Raises:
            zmq.ZMQError: If connection fails
        """
        if self._connected:
            logger.debug("Already connected, skipping connect")
            return

        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{self.address}:{self.port}")
        self._connected = True
        logger.info(f"Async connection established to {self.address}:{self.port}")

    async def close(self) -> None:
        """Close the connection and clean up resources.

        Safely closes the socket and terminates the context.
        """
        if self.socket:
            self.socket.close()
            self.socket = None
        if self.context:
            self.context.term()
            self.context = None
        self._connected = False
        logger.info("Async connection closed")

    async def send_request(self, request: Message) -> Message:
        """Send a request and await the response.

        Automatically connects if not already connected.

        Args:
            request: Message object to send

        Returns:
            Response Message from server

        Raises:
            asyncio.TimeoutError: If response not received within timeout
            zmq.ZMQError: If network error occurs
        """
        if not self._connected:
            await self.connect()

        await self.socket.send_pyobj(request)

        try:
            response = await asyncio.wait_for(
                self.socket.recv_pyobj(),
                timeout=self.recv_timeout_ms / 1000.0
            )
            return response
        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for response to {type(request).__name__}")
            raise

    async def try_connect(self) -> ConnectResponse:
        """Attempt to connect and send a connect request.

        Returns:
            ConnectResponse with server state

        Raises:
            asyncio.TimeoutError: If connection times out
        """
        response = await self.send_request(ConnectRequest())
        return response

    @property
    def connected(self) -> bool:
        """Check if connected to server.

        Returns:
            True if socket is connected, False otherwise
        """
        return self._connected

    async def __aenter__(self) -> 'AsyncNetworkManager':
        """Async context manager entry - connects to server."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit - closes connection."""
        await self.close()

    async def async_get_state(self) -> Tuple[CoreState, bool]:
        """Get current server state asynchronously.

        Returns:
            Tuple of (CoreState, compute_metrics) from server

        Raises:
            asyncio.TimeoutError: If response not received within timeout
        """
        response = await self.send_request(StateRequest())
        return response.payload

    async def async_pause(self) -> Tuple[CoreState, bool]:
        """Pause the experiment asynchronously.

        Returns:
            Tuple of (CoreState, compute_metrics) after pause request

        Raises:
            asyncio.TimeoutError: If response not received within timeout
        """
        response = await self.send_request(PauseRequest())
        return response.payload

    async def async_start(self) -> Tuple[CoreState, bool]:
        """Start the experiment asynchronously.

        Returns:
            Tuple of (CoreState, compute_metrics) after start request

        Raises:
            asyncio.TimeoutError: If response not received within timeout
        """
        response = await self.send_request(StartRequest())
        return response.payload

    async def async_stop(self) -> Tuple[CoreState, bool]:
        """Stop the experiment asynchronously.

        Returns:
            Tuple of (CoreState, compute_metrics) after stop request

        Raises:
            asyncio.TimeoutError: If response not received within timeout
        """
        response = await self.send_request(StopRequest())
        return response.payload

    async def async_connect_to_server(self) -> Tuple[CoreState, bool]:
        """Connect and get initial server state.

        Returns:
            Tuple of (CoreState, compute_metrics) from server

        Raises:
            asyncio.TimeoutError: If response not received within timeout
        """
        response = await self.send_request(ConnectRequest())
        return response.payload
