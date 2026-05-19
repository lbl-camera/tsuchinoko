"""User-authored design code: name validation, filesystem layout, resolver.

Agent-authored callables (custom acquisition / kernel / prior_mean / noise
functions) are landed on disk under ``user_designs_root()`` and referenced
by ``"user:<name>"`` strings inside ``experiment.configure`` payloads.

Trust boundary: code dropped here is executed in this process. The wire
gate is in ``NATSService._handle_upload_design_code`` (this module only
implements the resolver and the on-disk layout).
"""
from __future__ import annotations

import importlib.util
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

KINDS: tuple[str, ...] = ("acquisition", "kernel", "prior_mean", "noise")

EXPECTED_CALLABLES: dict[str, str] = {
    "acquisition": "acquisition_function",
    "kernel": "kernel",
    "prior_mean": "prior_mean",
    "noise": "noise_function",
}

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,62}$")


class UserDesignError(Exception):
    """Validation, compile, or resolution failure for a user-authored design."""


def validate_name(name: str) -> None:
    """Raise ``UserDesignError`` if *name* is not a safe filesystem stem."""
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise UserDesignError(
            f"invalid design name {name!r}: must match {_NAME_RE.pattern}"
        )


def _validate_kind(kind: str) -> None:
    if kind not in KINDS:
        raise UserDesignError(
            f"unknown kind {kind!r}; expected one of {KINDS}"
        )


def user_designs_root() -> Path:
    """Return the root directory for uploaded designs.

    Honours ``$TSUCHINOKO_USER_DIR`` if set; otherwise
    ``~/.tsuchinoko/user_designs/``. Does not create directories.
    """
    base = os.environ.get("TSUCHINOKO_USER_DIR")
    if base:
        return Path(base) / "user_designs"
    return Path.home() / ".tsuchinoko" / "user_designs"


def _path_for(kind: str, name: str) -> Path:
    return user_designs_root() / kind / f"{name}.py"


def write_design(name: str, kind: str, code: str) -> tuple[str, Path]:
    """Validate, compile-check, and persist *code* under <kind>/<name>.py.

    Returns ``("user:<name>", absolute_path)``. Raises ``UserDesignError``
    on any validation failure; no file is written if validation fails.
    """
    _validate_kind(kind)
    validate_name(name)
    try:
        compiled = compile(code, f"<user_design:{kind}/{name}>", "exec")
    except SyntaxError as exc:
        raise UserDesignError(f"SyntaxError in design code: {exc}") from exc

    expected = EXPECTED_CALLABLES[kind]
    ns: dict[str, object] = {}
    try:
        exec(compiled, ns)
    except Exception as exc:
        raise UserDesignError(
            f"failed to evaluate design code: {type(exc).__name__}: {exc}"
        ) from exc

    fn = ns.get(expected)
    if not callable(fn):
        raise UserDesignError(
            f"design code must bind a callable named {expected!r} "
            f"(kind={kind!r})"
        )

    path = _path_for(kind, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(code, encoding="utf-8")
    return f"user:{name}", path


def resolve_user_ref(ref: str, kind: str) -> Callable[..., Any]:
    """Import the callable referenced by ``"user:<name>"``.

    Looks under ``<user_designs_root>/<kind>/<name>.py``. Raises
    ``UserDesignError`` if the ref is malformed or the file is missing.
    """
    _validate_kind(kind)
    if not isinstance(ref, str) or not ref.startswith("user:"):
        raise UserDesignError(
            f"expected 'user:<name>' ref, got {ref!r}"
        )
    name = ref[len("user:"):]
    validate_name(name)
    path = _path_for(kind, name)
    if not path.is_file():
        raise UserDesignError(
            f"unknown user design {kind}/{name} at {path}"
        )

    spec = importlib.util.spec_from_file_location(
        f"_tsuchinoko_user_design_{kind}_{name}", path
    )
    if spec is None or spec.loader is None:
        raise UserDesignError(f"cannot load module for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = EXPECTED_CALLABLES[kind]
    fn = getattr(module, expected, None)
    if not callable(fn):
        raise UserDesignError(
            f"{path} does not bind callable {expected!r}"
        )
    return fn
