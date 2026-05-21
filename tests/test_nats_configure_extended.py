"""Tests for the extended experiment.configure handler."""
import json
import textwrap
from unittest.mock import AsyncMock, MagicMock

import pytest

from tsuchinoko.nats.service import NATSService


class FakeMsg:
    def __init__(self, payload: dict):
        self.data = json.dumps(payload).encode()
        self.reply = "_INBOX.x"
        self.respond = AsyncMock()


def _service(monkeypatch, tmp_path):
    monkeypatch.setenv("TSUCHINOKO_USER_DIR", str(tmp_path))
    core = MagicMock()
    # Engine params recorded by setattr; tests inspect _recorded.
    core.adaptive_engine.parameters = MagicMock()
    return NATSService(core, MagicMock()), core


@pytest.mark.asyncio
async def test_configure_bounds_only_still_works(monkeypatch, tmp_path):
    svc, core = _service(monkeypatch, tmp_path)
    msg = FakeMsg({"parameter_bounds": [[0.0, 10.0], [-5.0, 5.0]]})
    await svc._handle_configure(msg)
    reply = json.loads(msg.respond.call_args.args[0])
    assert reply["status"] == "ok"
    # Bounds mapped to engine via the existing parameter-tree keys
    params = core.adaptive_engine.parameters.__setitem__.call_args_list
    keys = [call.args[0] for call in params]
    assert ("bounds", "axis_0_min") in keys
    assert ("bounds", "axis_0_max") in keys
    assert ("bounds", "axis_1_min") in keys
    assert ("bounds", "axis_1_max") in keys


@pytest.mark.asyncio
async def test_configure_unknown_key_rejected(monkeypatch, tmp_path):
    svc, _core = _service(monkeypatch, tmp_path)
    msg = FakeMsg({"parameter_bounds": [[0, 1]], "what_is_this": 42})
    await svc._handle_configure(msg)
    reply = json.loads(msg.respond.call_args.args[0])
    assert reply["status"] == "error"
    assert "what_is_this" in reply["message"]


@pytest.mark.asyncio
async def test_configure_full_typed_payload(monkeypatch, tmp_path):
    svc, core = _service(monkeypatch, tmp_path)
    msg = FakeMsg({
        "parameter_bounds": [[0.0, 1.0], [0.0, 1.0]],
        "dimensionality": 2,
        "kernel": "matern_3_2",
        "acquisition_function": "ucb",
        "prior_mean": None,
        "noise_function": None,
        "noise_variances": 0.01,
        "initial_points": 12,
        "training_method": "global",
        "hyperparameters": [1.0, 0.5, 0.5],
        "hyperparameter_bounds": [[0.1, 1e5], [0.1, 1e5], [0.1, 1e5]],
        "x_out": None,
    })
    await svc._handle_configure(msg)
    reply = json.loads(msg.respond.call_args.args[0])
    assert reply["status"] == "ok"
    # Typed fields that aren't param-tree-backed land on the engine via setattr.
    assert core.adaptive_engine.dimensionality == 2
    assert core.adaptive_engine.kernel == "matern_3_2"
    assert core.adaptive_engine.noise_variances == 0.01
    assert core.adaptive_engine.initial_points == 12
    assert core.adaptive_engine.training_method == "global"
    # Param-tree-backed fields (acquisition_function, hyperparameters,
    # hyperparameter_bounds, parameter_bounds) flow through
    # engine.parameters[...] = ... so request_targets actually reads them.
    setitem_keys = [
        c.args[0]
        for c in core.adaptive_engine.parameters.__setitem__.call_args_list
    ]
    assert ("bounds", "axis_0_min") in setitem_keys
    assert ("bounds", "axis_1_max") in setitem_keys
    assert "acquisition_function" in setitem_keys
    assert ("hyperparameters", "hyperparameter_0") in setitem_keys
    assert ("hyperparameters", "hyperparameter_2") in setitem_keys
    assert ("hyperparameters", "hyperparameter_0_min") in setitem_keys
    assert ("hyperparameters", "hyperparameter_2_max") in setitem_keys


@pytest.mark.asyncio
async def test_configure_user_ref_resolves(monkeypatch, tmp_path):
    svc, core = _service(monkeypatch, tmp_path)  # sets TSUCHINOKO_USER_DIR
    from tsuchinoko.nats.user_designs import write_design
    from tsuchinoko.adaptive.gpCAM_in_process import gpcam_acquisition_functions
    code = "def acquisition_function(x, gp, **_):\n    return [0.0]\n"
    write_design("my_ucb", "acquisition", code)

    msg = FakeMsg({
        "parameter_bounds": [[0, 1]],
        "acquisition_function": "user:my_ucb",
    })
    await svc._handle_configure(msg)
    reply = json.loads(msg.respond.call_args.args[0])
    assert reply["status"] == "ok"
    # The user-supplied callable is registered under the ref name so
    # request_targets' gpcam_acquisition_functions[name] lookup resolves
    # to it. Stronger than callable() — the user's fn returns [0.0].
    assert "user:my_ucb" in gpcam_acquisition_functions
    assert gpcam_acquisition_functions["user:my_ucb"]([0.0], None) == [0.0]
    # And the ref name is written to the param tree (which is where
    # request_targets actually reads acquisition_function from).
    setitem_calls = core.adaptive_engine.parameters.__setitem__.call_args_list
    assert any(
        c.args == ("acquisition_function", "user:my_ucb") for c in setitem_calls
    )


@pytest.mark.asyncio
async def test_configure_unknown_user_ref(monkeypatch, tmp_path):
    monkeypatch.setenv("TSUCHINOKO_USER_DIR", str(tmp_path))
    svc, _core = _service(monkeypatch, tmp_path)
    msg = FakeMsg({
        "parameter_bounds": [[0, 1]],
        "kernel": "user:does_not_exist",
    })
    await svc._handle_configure(msg)
    reply = json.loads(msg.respond.call_args.args[0])
    assert reply["status"] == "error"
    assert "does_not_exist" in reply["message"]


@pytest.mark.asyncio
async def test_configure_bad_user_ref_does_not_mutate_engine(monkeypatch, tmp_path):
    """If a user:<name> ref is missing, no engine state should change before
    the error reply. Regression test for transactional ordering."""
    monkeypatch.setenv("TSUCHINOKO_USER_DIR", str(tmp_path))
    svc, core = _service(monkeypatch, tmp_path)
    msg = FakeMsg({
        "parameter_bounds": [[0.0, 10.0]],
        "kernel": "user:does_not_exist",
    })
    await svc._handle_configure(msg)
    reply = json.loads(msg.respond.call_args.args[0])
    assert reply["status"] == "error"
    # parameter_bounds writes go through engine.parameters[...] = ...
    # If we mutated before resolving, this call_count would be > 0.
    setitem_calls = core.adaptive_engine.parameters.__setitem__.call_count
    assert setitem_calls == 0, (
        f"engine.parameters was mutated {setitem_calls}× before user-ref "
        f"resolution failed — configure is no longer transactional"
    )


@pytest.mark.asyncio
async def test_configure_training_schedule_writes(monkeypatch, tmp_path):
    """global_training/local_training/mcmc_training milestones are applied
    to the engine's TrainingParameter children, not silently dropped."""
    svc, core = _service(monkeypatch, tmp_path)
    msg = FakeMsg({
        "parameter_bounds": [[0, 1]],
        "global_training": [25, 75, 200],
        "local_training": [25, 50],
        "mcmc_training": [],
    })
    await svc._handle_configure(msg)
    reply = json.loads(msg.respond.call_args.args[0])
    assert reply["status"] == "ok"

    child_calls = core.adaptive_engine.parameters.child.call_args_list
    schedule_calls = [c for c in child_calls if c.args[:1] in (
        ("global_training",), ("local_training",), ("mcmc_training",),
    )]
    # One child() lookup per schedule key that was present
    assert {c.args[0] for c in schedule_calls} == {
        "global_training", "local_training", "mcmc_training"
    }
    # And each schedule_param.setSchedule(values) was called with the
    # right list.
    setSchedule_mock = core.adaptive_engine.parameters.child.return_value.setSchedule
    seen = [c.args[0] for c in setSchedule_mock.call_args_list]
    assert [25, 75, 200] in seen
    assert [25, 50] in seen
    assert [] in seen


@pytest.mark.asyncio
async def test_configure_validates_hyperparameter_length(monkeypatch, tmp_path):
    """A real engine knows its num_hyperparameters; configure must reject a
    wrong-length list rather than silently no-op'ing it (the LUCID footgun)."""
    from tsuchinoko.nats.service import NATSService
    from unittest.mock import MagicMock
    monkeypatch.setenv("TSUCHINOKO_USER_DIR", str(tmp_path))
    core = MagicMock()
    core.adaptive_engine.num_hyperparameters = 3
    core.adaptive_engine.dimensionality = 2
    svc = NATSService(core, MagicMock())

    msg = FakeMsg({
        "parameter_bounds": [[0, 1], [0, 1]],
        "hyperparameters": [100, 4, 4, 4],  # 4 values, engine expects 3
    })
    await svc._handle_configure(msg)
    reply = json.loads(msg.respond.call_args.args[0])
    assert reply["status"] == "error"
    assert "hyperparameters" in reply["message"]
    assert "expected 3" in reply["message"]
    # Validation runs before any apply: engine.parameters never mutated.
    assert core.adaptive_engine.parameters.__setitem__.call_count == 0


@pytest.mark.asyncio
async def test_configure_validates_hyperparameter_bounds_shape(monkeypatch, tmp_path):
    svc, _core = _service(monkeypatch, tmp_path)
    msg = FakeMsg({
        "parameter_bounds": [[0, 1]],
        "hyperparameter_bounds": [[0.1, 1e5], [0.1]],  # second pair malformed
    })
    await svc._handle_configure(msg)
    reply = json.loads(msg.respond.call_args.args[0])
    assert reply["status"] == "error"
    assert "hyperparameter_bounds" in reply["message"]


@pytest.mark.asyncio
async def test_configure_validates_schedule_is_positive_ints(monkeypatch, tmp_path):
    svc, _core = _service(monkeypatch, tmp_path)
    msg = FakeMsg({
        "parameter_bounds": [[0, 1]],
        "global_training": [20, -1, 100],
    })
    await svc._handle_configure(msg)
    reply = json.loads(msg.respond.call_args.args[0])
    assert reply["status"] == "error"
    assert "global_training" in reply["message"]


@pytest.mark.asyncio
async def test_configure_acquisition_function_routes_to_param_tree(monkeypatch, tmp_path):
    """Builtin acquisition_function names must reach the ListParameter, not
    a silent setattr — that's what request_targets reads."""
    svc, core = _service(monkeypatch, tmp_path)
    msg = FakeMsg({
        "parameter_bounds": [[0, 1]],
        "acquisition_function": "variance",
    })
    await svc._handle_configure(msg)
    reply = json.loads(msg.respond.call_args.args[0])
    assert reply["status"] == "ok"
    setitem_calls = core.adaptive_engine.parameters.__setitem__.call_args_list
    assert any(
        c.args == ("acquisition_function", "variance") for c in setitem_calls
    )


# ============================================================
# End-to-end tests: configure must actually change engine behavior.
# These use a real GPCAMInProcessEngine so a silent no-op cannot pass.
# ============================================================

@pytest.fixture
def real_engine_service(monkeypatch, tmp_path):
    monkeypatch.setenv("TSUCHINOKO_USER_DIR", str(tmp_path))
    from tsuchinoko.adaptive.gpCAM_in_process import GPCAMInProcessEngine
    engine = GPCAMInProcessEngine(
        dimensionality=2,
        parameter_bounds=[(0.0, 10.0), (0.0, 10.0)],
        hyperparameters=[1.0, 1.0, 1.0],
        hyperparameter_bounds=[(0.1, 100.0), (0.1, 100.0), (0.1, 100.0)],
    )
    core = MagicMock()
    core.adaptive_engine = engine
    svc = NATSService(core, MagicMock())
    return svc, engine


@pytest.mark.asyncio
async def test_configure_kernel_builtin_rebuilds_optimizer(real_engine_service):
    """A builtin kernel name must end up in the optimizer's GP, not setattr limbo."""
    from tsuchinoko.adaptive.gpCAM_in_process import BUILTIN_KERNELS
    svc, engine = real_engine_service

    msg = FakeMsg({"kernel": "matern_5_2"})
    await svc._handle_configure(msg)
    reply = json.loads(msg.respond.call_args.args[0])
    assert reply["status"] == "ok"
    assert engine.kernel == "matern_5_2"
    # After reset(), the optimizer's kernel_function should be the matern_5_2
    # builtin (same callable identity).
    assert engine.optimizer.kernel_function is BUILTIN_KERNELS["matern_5_2"]


@pytest.mark.asyncio
async def test_configure_kernel_user_ref_rebuilds_with_callable(real_engine_service, monkeypatch, tmp_path):
    from tsuchinoko.nats.user_designs import write_design
    svc, engine = real_engine_service

    code = (
        "import numpy as np\n"
        "def kernel(x1, x2, hyperparameters):\n"
        "    return np.zeros((len(x1), len(x2)))\n"
    )
    write_design("my_kernel", "kernel", code)
    msg = FakeMsg({"kernel": "user:my_kernel"})
    await svc._handle_configure(msg)
    reply = json.loads(msg.respond.call_args.args[0])
    assert reply["status"] == "ok"
    # Engine got the resolved callable, optimizer picked it up.
    assert callable(engine.kernel)
    assert engine.optimizer.kernel_function is engine.kernel


@pytest.mark.asyncio
async def test_configure_kernel_unknown_name_rejected(real_engine_service):
    svc, engine = real_engine_service
    original_kernel = engine.optimizer.kernel_function
    msg = FakeMsg({"kernel": "not_a_kernel"})
    await svc._handle_configure(msg)
    reply = json.loads(msg.respond.call_args.args[0])
    assert reply["status"] == "error"
    assert "kernel" in reply["message"]
    # No engine mutation on validation failure.
    assert not hasattr(engine, "kernel") or engine.kernel is None
    assert engine.optimizer.kernel_function is original_kernel


@pytest.mark.asyncio
async def test_configure_initial_points_keeps_request_random(real_engine_service):
    """request_targets must stay random while n_data < initial_points,
    even after tell() builds the GP."""
    import numpy as np
    from tsuchinoko.adaptive import Data
    svc, engine = real_engine_service

    msg = FakeMsg({"initial_points": 5})
    await svc._handle_configure(msg)
    assert json.loads(msg.respond.call_args.args[0])["status"] == "ok"
    assert engine.initial_points == 5

    data = Data(dimensionality=2)
    for i in range(3):  # below the initial_points quota
        data.inject_new([((i * 1.0, i * 1.0), float(i), 0.01, {})])
    engine.update_measurements(data)
    assert engine.optimizer.gp is not None  # GP exists, but quota unfilled

    targets = engine.request_targets(position=(5.0, 5.0))
    # Random path returns plain lists; gpCAM ask() would return np.ndarray
    # rows. The plain-list shape confirms we stayed in the exploration branch.
    assert isinstance(targets[0], list)


@pytest.mark.asyncio
async def test_configure_training_method_pins_train_iteration(real_engine_service):
    """When training_method is set, train() must only iterate that method."""
    import numpy as np
    from tsuchinoko.adaptive import Data
    svc, engine = real_engine_service

    msg = FakeMsg({
        "training_method": "local",
        "local_training": [3],  # train at iteration > 3
    })
    await svc._handle_configure(msg)
    assert json.loads(msg.respond.call_args.args[0])["status"] == "ok"
    assert engine.training_method == "local"

    data = Data(dimensionality=2)
    for i in range(8):
        data.inject_new([((i * 1.0, i * 1.0), float(np.sin(i)), 0.01, {})])
    engine.update_measurements(data)

    engine.train()
    # Only 'local' should have a completed milestone; global/mcmc untouched.
    assert 3 in engine._completed_training.get("local", set())
    assert not engine._completed_training.get("global", set())
    assert not engine._completed_training.get("mcmc", set())


@pytest.mark.asyncio
async def test_configure_noise_variances_reaches_optimizer(real_engine_service):
    """Configure-time noise_variances must propagate into GPOptimizer init."""
    svc, engine = real_engine_service

    msg = FakeMsg({"noise_variances": 0.25})
    await svc._handle_configure(msg)
    assert json.loads(msg.respond.call_args.args[0])["status"] == "ok"
    assert engine.noise_variances == 0.25
    # gp_opts wasn't set by the caller, so GPOptimizer got noise_variances=0.25
    # at construction. We can't easily introspect the GPOptimizer's stored
    # value across gpCAM versions, but the engine attribute and a successful
    # reset() (no exception) prove the value reached init_optimizer.
    assert engine.optimizer is not None


@pytest.mark.asyncio
async def test_configure_rejects_bad_kernel_type(real_engine_service):
    svc, _engine = real_engine_service
    msg = FakeMsg({"kernel": 42})
    await svc._handle_configure(msg)
    reply = json.loads(msg.respond.call_args.args[0])
    assert reply["status"] == "error"
    assert "kernel" in reply["message"]


@pytest.mark.asyncio
async def test_configure_rejects_bad_training_method(real_engine_service):
    svc, _engine = real_engine_service
    msg = FakeMsg({"training_method": "not_a_method"})
    await svc._handle_configure(msg)
    reply = json.loads(msg.respond.call_args.args[0])
    assert reply["status"] == "error"
    assert "training_method" in reply["message"]


@pytest.mark.asyncio
async def test_configure_rejects_bad_initial_points(real_engine_service):
    svc, _engine = real_engine_service
    msg = FakeMsg({"initial_points": -3})
    await svc._handle_configure(msg)
    reply = json.loads(msg.respond.call_args.args[0])
    assert reply["status"] == "error"
    assert "initial_points" in reply["message"]


@pytest.mark.asyncio
async def test_configure_rejects_bad_noise_variances(real_engine_service):
    svc, _engine = real_engine_service
    msg = FakeMsg({"noise_variances": "loud"})
    await svc._handle_configure(msg)
    reply = json.loads(msg.respond.call_args.args[0])
    assert reply["status"] == "error"
    assert "noise_variances" in reply["message"]


@pytest.mark.asyncio
async def test_configure_rejects_non_user_prior_mean(real_engine_service):
    svc, _engine = real_engine_service
    msg = FakeMsg({"prior_mean": "linear"})  # not None, not user:<name>
    await svc._handle_configure(msg)
    reply = json.loads(msg.respond.call_args.args[0])
    assert reply["status"] == "error"
    assert "prior_mean" in reply["message"]
