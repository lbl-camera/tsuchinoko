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
    {"suffix": "experiment.bind_run", "description": "Bind a bluesky run for Tiled I/O"},
    {"suffix": "experiment.start", "description": "Begin the adaptive loop"},
    {"suffix": "experiment.pause", "description": "Pause the loop"},
    {"suffix": "experiment.resume", "description": "Resume from pause"},
    {"suffix": "experiment.stop", "description": "Stop and finalize"},
    {"suffix": "experiment.upload_design_code", "description": "Upload an agent-authored callable (acquisition/kernel/prior_mean/noise)"},
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
    # Recognised configure keys (strict: unknown keys are an error)
    _CONFIGURE_KEYS = frozenset({
        "parameter_bounds",
        "dimensionality",
        "kernel",
        "acquisition_function",
        "prior_mean",
        "noise_function",
        "noise_variances",
        "initial_points",
        "training_method",
        "hyperparameters",
        "hyperparameter_bounds",
        "global_training",
        "local_training",
        "mcmc_training",
        "x_out",
    })

    _CONFIGURE_KIND_BY_KEY = {
        "acquisition_function": "acquisition",
        "kernel": "kernel",
        "prior_mean": "prior_mean",
        "noise_function": "noise",
    }

    def __init__(self, core: Core, client: NATSClient) -> None:
        self._core = core
        self._client = client
        self._subscriptions = []
        self._instance_id = str(uuid.uuid4())

    async def start(self) -> None:
        handler_map = {
            "experiment.configure": self._handle_configure,
            "experiment.bind_run": self._handle_bind_run,
            "experiment.start": self._handle_start,
            "experiment.pause": self._handle_pause,
            "experiment.resume": self._handle_resume,
            "experiment.stop": self._handle_stop,
            "experiment.upload_design_code": self._handle_upload_design_code,
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
        if not msg.reply:
            return
        await msg.respond(json.dumps(data).encode())

    async def _handle_bind_run(self, msg) -> None:
        """Bind a bluesky run: create TiledReader + TiledPublisher.

        Expects payload from LUCID with Tiled credentials (URL + API key
        from LUCID's session-key cache, sent as ``tiled_api_key``), so
        Tsuchinoko never authenticates independently.
        """
        try:
            data = json.loads(msg.data)
            run_uid = data["run_uid"]
            tiled_url = data.get("tiled_url", "")
            tiled_api_key = data.get("tiled_api_key")
            proxy_url = data.get("proxy_url")
            motor_names = data.get("motor_names", [])
            detector_name = data.get("detector_name", "det")

            from tsuchinoko.config import get_config
            config = get_config()
            effective_url = tiled_url or config.tiled.url

            if not effective_url:
                await self._reply(msg, {"status": "error", "message": "No tiled_url"})
                return

            lucid_prefix = data.get("lucid_prefix", config.nats.lucid_prefix)

            from tsuchinoko.tiled.connect import connect_tiled
            tiled_client = connect_tiled(
                effective_url,
                api_key=tiled_api_key,
                proxy_url=proxy_url,
            )

            # Wire TiledReader into LUCIDEngine
            from tsuchinoko.execution.lucid import LUCIDEngine
            if isinstance(self._core.execution_engine, LUCIDEngine):
                from tsuchinoko.tiled.reader import TiledReader
                reader = TiledReader(
                    tiled_client, run_uid, motor_names, detector_name,
                )
                self._core.execution_engine.bind_run(reader)

            # Wire TiledPublisher into Core
            from tsuchinoko.tiled.writer import TiledPublisher
            dim = self._core.adaptive_engine.dimensionality
            publisher = TiledPublisher(tiled_client, run_uid, dim)
            publisher.write_config(self._core.adaptive_engine)
            self._core._tiled_publisher = publisher

            # Subscribe to {lucid_prefix}.adaptive.measured for unblocking
            measured_subject = f"{lucid_prefix}.adaptive.measured"

            async def on_measured(nats_msg):
                if isinstance(self._core.execution_engine, LUCIDEngine):
                    self._core.execution_engine.signal_measurements_ready()

            sub = await self._client.subscribe(measured_subject, on_measured)
            self._subscriptions.append(sub)

            # Auto-start: LUCID expects tsuchinoko to begin after bind_run
            from tsuchinoko.core import CoreState
            if self._core.state == CoreState.Inactive:
                self._core.state = CoreState.Starting

            logger.info(f"Bound run {run_uid[:8]}… (tiled={effective_url})")
            await self._reply(msg, {"status": "ok", "run_uid": run_uid})
        except Exception as e:
            logger.exception(e)
            await self._reply(msg, {"status": "error", "message": str(e)})

    async def _handle_configure(self, msg) -> None:
        try:
            from tsuchinoko.nats.user_designs import UserDesignError, resolve_user_ref
            data = json.loads(msg.data)

            unknown = set(data) - self._CONFIGURE_KEYS
            if unknown:
                await self._reply(msg, {
                    "status": "error",
                    "message": f"unknown configure field(s): {sorted(unknown)}",
                })
                return

            engine = self._core.adaptive_engine

            shape_errors = self._validate_configure(data, engine)
            if shape_errors:
                await self._reply(msg, {
                    "status": "error",
                    "message": "; ".join(shape_errors),
                })
                return

            # Resolve user:<name> refs FIRST so a missing ref aborts before
            # any engine mutation. Keeps configure transactional from the
            # caller's POV: status=error implies engine unchanged.
            resolved: dict[str, object] = {}
            for key, kind in self._CONFIGURE_KIND_BY_KEY.items():
                value = data.get(key)
                if isinstance(value, str) and value.startswith("user:"):
                    resolved[key] = resolve_user_ref(value, kind)

            # Resolution succeeded — now apply everything.

            if "parameter_bounds" in data:
                for i, (lo, hi) in enumerate(data["parameter_bounds"]):
                    engine.parameters[("bounds", f"axis_{i}_min")] = lo
                    engine.parameters[("bounds", f"axis_{i}_max")] = hi

            if "hyperparameter_bounds" in data:
                for i, (lo, hi) in enumerate(data["hyperparameter_bounds"]):
                    engine.parameters[("hyperparameters", f"hyperparameter_{i}_min")] = lo
                    engine.parameters[("hyperparameters", f"hyperparameter_{i}_max")] = hi

            if data.get("hyperparameters") is not None:
                for i, value in enumerate(data["hyperparameters"]):
                    engine.parameters[("hyperparameters", f"hyperparameter_{i}")] = value

            for sched_key in ("global_training", "local_training", "mcmc_training"):
                if sched_key in data:
                    engine.parameters.child(sched_key).setSchedule(data[sched_key])

            if "acquisition_function" in data:
                self._apply_acquisition_function(
                    engine,
                    data["acquisition_function"],
                    resolved.get("acquisition_function"),
                )

            # In-place keys — the engine reads these fresh on each iteration,
            # so no optimizer rebuild is required.
            for key in (
                "dimensionality",
                "initial_points", "training_method", "x_out",
            ):
                if key in data:
                    setattr(engine, key, data[key])

            # Optimizer-rebuild keys — the engine reads these only at
            # init_optimizer time, so we setattr and then trigger reset()
            # so the GP is rebuilt with the new choices. Safe because
            # configure happens before bind_run in the LUCID flow (no
            # data to lose).
            rebuild_keys = ("kernel", "prior_mean", "noise_function", "noise_variances")
            needs_rebuild = False
            for key in rebuild_keys:
                if key in data:
                    setattr(engine, key, resolved.get(key, data[key]))
                    needs_rebuild = True
            if needs_rebuild and hasattr(engine, "reset"):
                engine.reset()

            await self._reply(msg, {"status": "ok"})
        except UserDesignError as exc:
            logger.debug("configure rejected: {}", exc)
            await self._reply(msg, {"status": "error", "message": str(exc)})
        except Exception as exc:
            logger.exception(exc)
            await self._reply(msg, {"status": "error", "message": str(exc)})

    @staticmethod
    def _validate_configure(data: dict, engine) -> list[str]:
        """Validate types/shapes of a configure payload.

        Returns a list of error messages (empty on success). When the
        engine's expected hyperparameter count is introspectable
        (``num_hyperparameters`` is a real int), length checks run; when
        it's not (e.g. a ``MagicMock`` engine in tests), those length
        checks are skipped.
        """
        errors: list[str] = []

        def _is_pair_list(x) -> bool:
            return (
                isinstance(x, list)
                and all(
                    isinstance(p, (list, tuple))
                    and len(p) == 2
                    and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in p)
                    for p in x
                )
            )

        if "parameter_bounds" in data and not _is_pair_list(data["parameter_bounds"]):
            errors.append("parameter_bounds: expected list of [min, max] pairs")

        if "hyperparameter_bounds" in data and not _is_pair_list(data["hyperparameter_bounds"]):
            errors.append("hyperparameter_bounds: expected list of [min, max] pairs")

        if "hyperparameters" in data and data["hyperparameters"] is not None:
            hps = data["hyperparameters"]
            if not isinstance(hps, list) or any(
                not isinstance(v, (int, float)) or isinstance(v, bool) for v in hps
            ):
                errors.append("hyperparameters: expected list of numbers or null")

        num_hp = getattr(engine, "num_hyperparameters", None)
        if isinstance(num_hp, int):
            hps = data.get("hyperparameters")
            if isinstance(hps, list) and len(hps) != num_hp:
                errors.append(
                    f"hyperparameters: expected {num_hp} values, got {len(hps)}"
                )
            hpb = data.get("hyperparameter_bounds")
            if isinstance(hpb, list) and len(hpb) != num_hp:
                errors.append(
                    f"hyperparameter_bounds: expected {num_hp} pairs, got {len(hpb)}"
                )

        dim = getattr(engine, "dimensionality", None)
        if isinstance(dim, int):
            pb = data.get("parameter_bounds")
            if isinstance(pb, list) and len(pb) != dim:
                errors.append(
                    f"parameter_bounds: expected {dim} pairs, got {len(pb)}"
                )

        for key in ("global_training", "local_training", "mcmc_training"):
            if key in data:
                v = data[key]
                if not isinstance(v, list) or any(
                    not isinstance(n, int) or isinstance(n, bool) or n <= 0
                    for n in v
                ):
                    errors.append(f"{key}: expected list of positive ints")

        if "kernel" in data:
            from tsuchinoko.adaptive.gpCAM_in_process import BUILTIN_KERNELS
            k = data["kernel"]
            if k is not None:
                if not isinstance(k, str):
                    errors.append("kernel: expected string or null")
                elif not (k.startswith("user:") or k in BUILTIN_KERNELS):
                    errors.append(
                        f"kernel: unknown kernel {k!r}; expected one of "
                        f"{sorted(BUILTIN_KERNELS)} or 'user:<name>'"
                    )

        for key in ("prior_mean", "noise_function"):
            if key in data:
                v = data[key]
                if v is not None and not (isinstance(v, str) and v.startswith("user:")):
                    errors.append(f"{key}: expected null or 'user:<name>'")

        if "noise_variances" in data:
            nv = data["noise_variances"]
            ok = (
                nv is None
                or (isinstance(nv, (int, float)) and not isinstance(nv, bool))
                or (isinstance(nv, list)
                    and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in nv))
            )
            if not ok:
                errors.append("noise_variances: expected null, number, or list of numbers")

        if "initial_points" in data and data["initial_points"] is not None:
            ip = data["initial_points"]
            if not isinstance(ip, int) or isinstance(ip, bool) or ip < 0:
                errors.append("initial_points: expected non-negative int or null")

        if "training_method" in data and data["training_method"] is not None:
            tm = data["training_method"]
            if tm not in {"global", "local", "mcmc", "adam", "hgdl"}:
                errors.append(
                    "training_method: expected one of "
                    "['adam', 'global', 'hgdl', 'local', 'mcmc'] or null"
                )

        if "x_out" in data and data["x_out"] is not None:
            xo = data["x_out"]
            if not isinstance(xo, list):
                errors.append("x_out: expected list or null")

        return errors

    @staticmethod
    def _apply_acquisition_function(engine, raw_value, resolved_value) -> None:
        """Route acquisition_function to the place the engine actually reads it.

        Builtins (string in ``gpcam_acquisition_functions``) go to the
        ListParameter so ``request_targets`` picks them up. User-refs
        (resolved to a callable) are registered under their ref name in
        ``gpcam_acquisition_functions`` and the ref string is written to
        the param tree, so the same lookup path keeps working.
        """
        if callable(resolved_value):
            from tsuchinoko.adaptive.gpCAM_in_process import gpcam_acquisition_functions
            ref = raw_value
            gpcam_acquisition_functions[ref] = resolved_value
            try:
                acq_param = engine.parameters.child("acquisition_function")
            except (AttributeError, KeyError):
                setattr(engine, "acquisition_function", resolved_value)
                return
            limits = getattr(acq_param, "limits", None)
            if isinstance(limits, list) and ref not in limits:
                limits.append(ref)
            engine.parameters["acquisition_function"] = ref
            return

        try:
            engine.parameters["acquisition_function"] = raw_value
        except (KeyError, AttributeError):
            setattr(engine, "acquisition_function", raw_value)

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

    async def _handle_upload_design_code(self, msg) -> None:
        try:
            from tsuchinoko.nats.user_designs import UserDesignError, write_design
            data = json.loads(msg.data)
            name = data["name"]
            kind = data["kind"]
            code = data["code"]
            ref, path = write_design(name, kind, code)
            await self._reply(msg, {
                "status": "ok",
                "ref": ref,
                "path": str(path),
            })
        except UserDesignError as exc:
            logger.debug("upload_design_code rejected: {}", exc)
            await self._reply(msg, {"status": "error", "message": str(exc)})
        except Exception as exc:
            logger.exception(exc)
            await self._reply(msg, {"status": "error", "message": str(exc)})

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
