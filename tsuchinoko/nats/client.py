"""NATS client for Tsuchinoko — connection, auth, publish/subscribe."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable, Optional

import nats
import nats.errors
from loguru import logger

from .config import NATSConfig


class NATSClient:
    """Shared NATS connection owner."""

    def __init__(self) -> None:
        self._nc: Optional[nats.NATS] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._auth_state: dict[str, str] = {}
        self._tiled_credentials: dict[str, dict] = {}
        self.is_connected: bool = False

    async def connect(self, config: NATSConfig) -> None:
        self._nc = await nats.connect(
            config.url,
            connect_timeout=int(config.connect_timeout),
            allow_reconnect=config.reconnect,
            error_cb=self._on_error,
            disconnected_cb=self._on_disconnect,
            reconnected_cb=self._on_reconnect,
        )
        self._loop = asyncio.get_running_loop()
        self.is_connected = True
        logger.info(f"Connected to NATS at {config.url}")

    async def close(self) -> None:
        if self._nc:
            await self._nc.drain()
        self.is_connected = False
        self._nc = None
        logger.info("NATS connection closed")

    async def authenticate(self, lucid_prefix: str, app_name: str, app_version: str, timeout: float) -> dict:
        """Auth handshake with LUCID. Returns Tiled creds. Caches per prefix."""
        cached = self._auth_state.get(lucid_prefix)
        if cached == "approved":
            return self._tiled_credentials.get(lucid_prefix, {})
        if cached == "denied":
            raise PermissionError(f"LUCID at '{lucid_prefix}' previously denied access")

        subject = f"{lucid_prefix}.auth.request"
        payload = json.dumps({"app_name": app_name, "app_version": app_version}).encode()
        msg = await self._nc.request(subject, payload, timeout=timeout)
        data = json.loads(msg.data)

        if data.get("status") == "approved":
            self._auth_state[lucid_prefix] = "approved"
            self._tiled_credentials[lucid_prefix] = {
                "tiled_token": data.get("tiled_token"),
                "tiled_url": data.get("tiled_url"),
            }
            logger.info(f"Authenticated with LUCID at '{lucid_prefix}'")
            return self._tiled_credentials[lucid_prefix]
        else:
            reason = data.get("reason", "denied by operator")
            self._auth_state[lucid_prefix] = "denied"
            raise PermissionError(f"LUCID denied access: {reason}")

    async def publish(self, subject: str, payload: dict) -> None:
        if self._nc and self.is_connected:
            await self._nc.publish(subject, json.dumps(payload).encode())

    async def subscribe(self, subject: str, callback: Callable) -> Any:
        return await self._nc.subscribe(subject, cb=callback)

    async def request(self, subject: str, payload: dict, timeout: float = 5.0) -> dict:
        msg = await self._nc.request(subject, json.dumps(payload).encode(), timeout=timeout)
        return json.loads(msg.data)

    def publish_threadsafe(self, subject: str, payload: dict) -> None:
        if self._nc and self.is_connected and self._loop:
            asyncio.run_coroutine_threadsafe(self.publish(subject, payload), self._loop)

    async def _on_error(self, exc: Exception) -> None:
        logger.error(f"NATS error: {exc}")

    async def _on_disconnect(self) -> None:
        self.is_connected = False
        logger.warning("NATS disconnected")

    async def _on_reconnect(self) -> None:
        self.is_connected = True
        logger.info("NATS reconnected")
