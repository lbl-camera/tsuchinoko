# Phase 5 — Rich `experiment.configure` + `upload_design_code`

**Status:** Implemented
**Date:** 2026-05-19
**Canonical spec:** `ncs/docs/superpowers/specs/2026-05-19-lucid-autonomous-experiments-design.md`

## Summary

Two NATS-side changes that let the LUCID embedded Claude agent drive
end-to-end autonomous experiments:

1. **`experiment.configure`** grows from `{parameter_bounds}` into a
   typed payload (kernel, acquisition_function, prior_mean,
   noise_function/variances, initial_points, training_method,
   hyperparameters, x_out, dimensionality). Validation is strict —
   unknown top-level keys return an error.
2. **`experiment.upload_design_code`** (new action) accepts
   ``{name, kind, code}`` and persists agent-authored Python under
   ``<user_designs_root>/<kind>/<name>.py``. Configure resolves
   ``"user:<name>"`` refs at apply time.

## Expected callable signatures (per kind)

| kind | expected callable | signature (per gpCAM upstream) |
|---|---|---|
| `acquisition` | `acquisition_function` | `(x, gp, **_) -> ndarray` |
| `kernel` | `kernel` | `(x1, x2, hyperparameters) -> ndarray` |
| `prior_mean` | `prior_mean` | `(x, hyperparameters, gp) -> ndarray` |
| `noise` | `noise_function` | `(x, hyperparameters, gp) -> ndarray` |

The handler validates the bound name only; signature errors surface
at run time inside the adaptive engine, where they're already handled.

## User-designs root

`~/.tsuchinoko/user_designs/<kind>/<name>.py` by default, or
`$TSUCHINOKO_USER_DIR/user_designs/<kind>/<name>.py` if set.

## Trust boundary

Code dropped through `upload_design_code` runs in this process. This
exposure is identical to what `engine.set_parameter` already permits
via the pyqtgraph param tree. NATS-level authentication is governed
by the existing IPC design (TLS, no broker creds).

## References

- LUCID canonical spec: `ncs/docs/superpowers/specs/2026-05-19-lucid-autonomous-experiments-design.md`
- LUCID implementation plan: `ncs/docs/superpowers/plans/2026-05-19-lucid-autonomous-experiments.md`
- Previous phase: `docs/design/2026-04-12-phase2-nats-integration.md`
