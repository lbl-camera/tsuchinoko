# Phase 5 — Implementation Plan

Tracked in the LUCID repo:
`ncs/docs/superpowers/plans/2026-05-19-lucid-autonomous-experiments.md`

Tsuchinoko-side tasks land first (Phase A of that plan), in this
order:

1. `tsuchinoko/nats/user_designs.py` + `tests/test_user_designs.py`
2. `experiment.upload_design_code` handler + `tests/test_nats_upload_design_code.py`
3. Typed `experiment.configure` + `tests/test_nats_configure_extended.py`
4. This design doc + this plan doc
5. MR against `LUCID-refactor` titled
   `feat(nats): typed configure schema + upload_design_code`

Deploy this MR before LUCID's `feature/autonomous-experiment-agent`
branch — strict validation in the new configure means old Tsuchinoko
would reject payloads sent by the new LUCID plugin.
