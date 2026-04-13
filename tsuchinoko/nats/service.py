"""Tsuchinoko's service identity on the NATS bus."""

from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from tsuchinoko.core import Core
    from .client import NATSClient

ACTIONS = [
    {"suffix": "experiment.configure", "description": "Set up experiment parameters"},
    {"suffix": "experiment.start", "description": "Begin the adaptive loop"},
    {"suffix": "experiment.pause", "description": "Pause the loop"},
    {"suffix": "experiment.resume", "description": "Resume from pause"},
    {"suffix": "experiment.stop", "description": "Stop and finalize"},
    {"suffix": "engine.set_parameter", "description": "Update a single engine parameter"},
    {"suffix": "engine.get_parameters", "description": "Retrieve current engine parameters"},
    {"suffix": "status", "description": "Query current state and progress"},
]

EVENTS = [
    {"suffix": "state", "description": "State machine transitions"},
    {"suffix": "targets", "description": "New targets computed"},
    {"suffix": "gp.updated", "description": "GP outputs updated"},
    {"suffix": "error", "description": "Error in adaptive loop"},
]


class NATSService:
    def __init__(self, core: Core, client: NATSClient) -> None:
        self._core = core
        self._client = client
        self._subscriptions = []
        self._instance_id = str(uuid.uuid4())

    async def start(self) -> None:
        handler_map = {
            "experiment.configure": self._handle_configure,
            "experiment.start": self._handle_start,
            "experiment.pause": self._handle_pause,
            "experiment.resume": self._handle_resume,
            "experiment.stop": self._handle_stop,
            "engine.set_parameter": self._handle_set_parameter,
            "engine.get_parameters": self._handle_get_parameters,
            "status": self._handle_status,
        }
        for suffix, handler in handler_map.items():
            sub = await self._client.subscribe(f"tsuchinoko.{suffix}", handler)
            self._subscriptions.append(sub)

        sub = await self._client.subscribe("_tsuchinoko.discover", self._handle_discover)
        self._subscriptions.append(sub)
        sub = await self._client.subscribe("tsuchinoko.meta.actions", self._handle_meta_actions)
        self._subscriptions.append(sub)
        sub = await self._client.subscribe("tsuchinoko.meta.events", self._handle_meta_events)
        self._subscriptions.append(sub)

        logger.info(f"NATSService started: {len(handler_map)} actions, instance={self._instance_id[:8]}")

    async def stop(self) -> None:
        for sub in self._subscriptions:
            await sub.unsubscribe()
        self._subscriptions.clear()
        logger.info("NATSService stopped")

    async def _reply(self, msg, data: dict) -> None:
        await msg.respond(json.dumps(data).encode())

    async def _handle_configure(self, msg) -> None:
        try:
            data = json.loads(msg.data)
            if "parameter_bounds" in data:
                bounds = data["parameter_bounds"]
                for i, (lo, hi) in enumerate(bounds):
                    self._core.adaptive_engine.parameters[("bounds", f"axis_{i}_min")] = lo
                    self._core.adaptive_engine.parameters[("bounds", f"axis_{i}_max")] = hi
            await self._reply(msg, {"status": "ok"})
        except Exception as e:
            logger.exception(e)
            await self._reply(msg, {"status": "error", "message": str(e)})

    async def _handle_start(self, msg) -> None:
        try:
            from tsuchinoko.core import CoreState
            self._core.state = CoreState.Starting
            await self._reply(msg, {"status": "ok", "state": self._core.state.name})
        except Exception as e:
            logger.exception(e)
            await self._reply(msg, {"status": "error", "message": str(e)})

    async def _handle_pause(self, msg) -> None:
        try:
            from tsuchinoko.core import CoreState
            self._core.state = CoreState.Pausing
            await self._reply(msg, {"status": "ok", "state": self._core.state.name})
        except Exception as e:
            logger.exception(e)
            await self._reply(msg, {"status": "error", "message": str(e)})

    async def _handle_resume(self, msg) -> None:
        try:
            from tsuchinoko.core import CoreState
            self._core.state = CoreState.Resuming
            await self._reply(msg, {"status": "ok", "state": self._core.state.name})
        except Exception as e:
            logger.exception(e)
            await self._reply(msg, {"status": "error", "message": str(e)})

    async def _handle_stop(self, msg) -> None:
        try:
            from tsuchinoko.core import CoreState
            self._core.state = CoreState.Stopping
            await self._reply(msg, {"status": "ok", "state": self._core.state.name})
        except Exception as e:
            logger.exception(e)
            await self._reply(msg, {"status": "error", "message": str(e)})

    async def _handle_set_parameter(self, msg) -> None:
        try:
            data = json.loads(msg.data)
            path = data["path"]
            value = data["value"]
            self._core.adaptive_engine.parameters.child(*path).setValue(value)
            await self._reply(msg, {"status": "ok"})
        except Exception as e:
            logger.exception(e)
            await self._reply(msg, {"status": "error", "message": str(e)})

    async def _handle_get_parameters(self, msg) -> None:
        try:
            state = self._core.adaptive_engine.parameters.saveState()
            await self._reply(msg, {"status": "ok", "parameters": state})
        except Exception as e:
            logger.exception(e)
            await self._reply(msg, {"status": "error", "message": str(e)})

    async def _handle_status(self, msg) -> None:
        try:
            await self._reply(msg, {
                "status": "ok",
                "state": self._core.state.name,
                "iteration": self._core.data._completed_iterations,
                "data_count": len(self._core.data),
            })
        except Exception as e:
            logger.exception(e)
            await self._reply(msg, {"status": "error", "message": str(e)})

    async def _handle_discover(self, msg) -> None:
        await self._reply(msg, {
            "instance_id": self._instance_id,
            "app_name": "tsuchinoko",
            "app_version": "",
            "prefix": "tsuchinoko",
            "actions_count": len(ACTIONS),
            "events_count": len(EVENTS),
            "state": self._core.state.name,
        })

    async def _handle_meta_actions(self, msg) -> None:
        await self._reply(msg, {"actions": ACTIONS})

    async def _handle_meta_events(self, msg) -> None:
        await self._reply(msg, {"events": EVENTS})
