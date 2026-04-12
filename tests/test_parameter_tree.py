"""Tests for the lightweight ParameterTree (pyqtgraph replacement)."""
import pytest



class TestParameter:
    def test_create_with_value(self):
        from tsuchinoko.parameters.tree import Parameter
        p = Parameter(name='foo', title='Foo', value=3.14, type='float')
        assert p.value() == 3.14

    def test_set_value(self):
        from tsuchinoko.parameters.tree import Parameter
        p = Parameter(name='foo', title='Foo', value=1, type='int')
        p.setValue(42)
        assert p.value() == 42

    def test_children_empty(self):
        from tsuchinoko.parameters.tree import Parameter
        p = Parameter(name='foo', title='Foo', value=0, type='int')
        assert p.children() == []

    def test_children_from_constructor(self):
        from tsuchinoko.parameters.tree import Parameter
        child1 = Parameter(name='a', title='A', value=1, type='int')
        child2 = Parameter(name='b', title='B', value=2, type='int')
        parent = Parameter(name='parent', title='Parent', children=[child1, child2])
        assert len(parent.children()) == 2

    def test_getitem_string_key(self):
        from tsuchinoko.parameters.tree import Parameter
        child = Parameter(name='x', title='X', value=99, type='int')
        parent = Parameter(name='top', children=[child])
        assert parent['x'] == 99

    def test_setitem_string_key(self):
        from tsuchinoko.parameters.tree import Parameter
        child = Parameter(name='x', title='X', value=0, type='int')
        parent = Parameter(name='top', children=[child])
        parent['x'] = 55
        assert parent['x'] == 55

    def test_getitem_tuple_key(self):
        from tsuchinoko.parameters.tree import Parameter
        grandchild = Parameter(name='val', title='Val', value=7, type='int')
        child = Parameter(name='group', title='Group', children=[grandchild])
        root = Parameter(name='top', children=[child])
        assert root[('group', 'val')] == 7

    def test_setitem_tuple_key(self):
        from tsuchinoko.parameters.tree import Parameter
        grandchild = Parameter(name='val', title='Val', value=0, type='int')
        child = Parameter(name='group', title='Group', children=[grandchild])
        root = Parameter(name='top', children=[child])
        root[('group', 'val')] = 123
        assert root[('group', 'val')] == 123

    def test_missing_key_raises_keyerror(self):
        from tsuchinoko.parameters.tree import Parameter
        p = Parameter(name='top')
        with pytest.raises(KeyError):
            _ = p['nonexistent']

    def test_missing_tuple_key_raises_keyerror(self):
        from tsuchinoko.parameters.tree import Parameter
        child = Parameter(name='group', title='Group')
        root = Parameter(name='top', children=[child])
        with pytest.raises(KeyError):
            _ = root[('group', 'missing')]

    def test_signal_fires_on_set(self):
        from tsuchinoko.parameters.tree import Parameter
        p = Parameter(name='foo', title='Foo', value=0, type='int')
        received = []
        p.sigValueChanged.connect(lambda param, val: received.append((param, val)))
        p.setValue(10)
        assert len(received) == 1
        assert received[0] == (p, 10)

    def test_block_signal_prevents_callback(self):
        from tsuchinoko.parameters.tree import Parameter
        p = Parameter(name='foo', title='Foo', value=0, type='int')
        received = []
        cb = lambda param, val: received.append(val)
        p.sigValueChanged.connect(cb)
        p.setValue(10, blockSignal=cb)
        assert received == []

    def test_block_signal_only_blocks_specified_callback(self):
        from tsuchinoko.parameters.tree import Parameter
        p = Parameter(name='foo', title='Foo', value=0, type='int')
        received_a = []
        received_b = []
        cb_a = lambda param, val: received_a.append(val)
        cb_b = lambda param, val: received_b.append(val)
        p.sigValueChanged.connect(cb_a)
        p.sigValueChanged.connect(cb_b)
        p.setValue(10, blockSignal=cb_a)
        assert received_a == []
        assert received_b == [10]

    def test_save_state(self):
        from tsuchinoko.parameters.tree import Parameter
        child = Parameter(name='x', title='X', value=5, type='int')
        parent = Parameter(name='top', title='Top', value=None, children=[child])
        state = parent.saveState()
        assert state['name'] == 'top'
        assert 'children' in state
        assert state['children'][0]['name'] == 'x'
        assert state['children'][0]['value'] == 5

    def test_add_child_from_dict(self):
        from tsuchinoko.parameters.tree import Parameter
        p = Parameter(name='top')
        p.addChild({'name': 'dynamic', 'value': 42, 'type': 'int'})
        assert p['dynamic'] == 42

    def test_add_child_from_parameter(self):
        from tsuchinoko.parameters.tree import Parameter
        p = Parameter(name='top')
        child = Parameter(name='child', value=7, type='int')
        p.addChild(child)
        assert p['child'] == 7

    def test_has_children_false(self):
        from tsuchinoko.parameters.tree import Parameter
        p = Parameter(name='top')
        assert p.hasChildren() is False

    def test_has_children_true(self):
        from tsuchinoko.parameters.tree import Parameter
        child = Parameter(name='child', value=1)
        p = Parameter(name='top', children=[child])
        assert p.hasChildren() is True

    def test_child_navigation(self):
        from tsuchinoko.parameters.tree import Parameter
        grandchild = Parameter(name='leaf', value=99)
        child = Parameter(name='group', children=[grandchild])
        root = Parameter(name='top', children=[child])
        assert root.child('group', 'leaf').value() == 99

    def test_child_single_level(self):
        from tsuchinoko.parameters.tree import Parameter
        child = Parameter(name='x', value=5)
        parent = Parameter(name='top', children=[child])
        assert parent.child('x').value() == 5

    def test_simpleparameter_is_alias(self):
        from tsuchinoko.parameters.tree import SimpleParameter, Parameter
        assert SimpleParameter is Parameter

    def test_groupparameter_is_alias(self):
        from tsuchinoko.parameters.tree import GroupParameter, Parameter
        assert GroupParameter is Parameter


class TestListParameter:
    def test_default_value_from_kwarg(self):
        from tsuchinoko.parameters.tree import ListParameter
        p = ListParameter(name='method', title='Method', limits=['a', 'b', 'c'], default='b')
        assert p.value() == 'b'

    def test_set_valid_value(self):
        from tsuchinoko.parameters.tree import ListParameter
        p = ListParameter(name='method', limits=['x', 'y', 'z'], default='x')
        p.setValue('y')
        assert p.value() == 'y'

    def test_set_invalid_value_raises(self):
        from tsuchinoko.parameters.tree import ListParameter
        p = ListParameter(name='method', limits=['a', 'b'], default='a')
        with pytest.raises(ValueError):
            p.setValue('not_in_list')

    def test_limits_stored(self):
        from tsuchinoko.parameters.tree import ListParameter
        limits = ['global', 'local', 'hgdl']
        p = ListParameter(name='method', limits=limits, default='global')
        assert p.limits == limits


class TestTrainingParameter:
    def test_initial_children(self):
        from tsuchinoko.parameters.tree import TrainingParameter, SimpleParameter
        import uuid
        children = [SimpleParameter(name=str(uuid.uuid4()), title='N=', value=N, type='int')
                    for N in (20, 50, 100)]
        tp = TrainingParameter(name='global_training', title='Train globally at...', addText='Add',
                               children=children)
        assert len(tp.children()) == 3

    def test_add_new_adds_child(self):
        from tsuchinoko.parameters.tree import TrainingParameter
        tp = TrainingParameter(name='global_training', title='Train globally at...', addText='Add')
        assert len(tp.children()) == 0
        tp.addNew()
        assert len(tp.children()) == 1

    def test_add_new_child_value_is_1(self):
        from tsuchinoko.parameters.tree import TrainingParameter
        tp = TrainingParameter(name='training', addText='Add')
        tp.addNew()
        child = tp.children()[0]
        assert child.value() == 1

    def test_add_new_child_title(self):
        from tsuchinoko.parameters.tree import TrainingParameter
        tp = TrainingParameter(name='training', addText='Add')
        tp.addNew()
        child = tp.children()[0]
        assert child.title == 'N='

    def test_iterate_children_values(self):
        from tsuchinoko.parameters.tree import TrainingParameter, SimpleParameter
        import uuid
        ns = (20, 50, 100, 400, 1000)
        children = [SimpleParameter(name=str(uuid.uuid4()), title='N=', value=N, type='int') for N in ns]
        tp = TrainingParameter(name='global_training', addText='Add', children=children)
        values = set(child.value() for child in tp.children())
        assert values == set(ns)


class TestEngineParameterPattern:
    """Integration test mimicking GPCAMInProcessEngine's exact parameter tree layout."""

    def _build_parameters(self):
        from tsuchinoko.parameters.tree import (
            SimpleParameter, GroupParameter, ListParameter, TrainingParameter
        )
        import uuid

        dimensionality = 2
        num_hyperparameters = 3
        default_retrain_globally_at = (20, 50, 100, 400, 1000)
        default_retrain_locally_at = (20, 40, 60, 80, 100, 200, 400, 1000)

        hyper_parameters = [
            SimpleParameter(title=f'Hyperparameter #{i + 1}', name=f'hyperparameter_{i}', type='float')
            for i in range(num_hyperparameters)
        ]
        hyper_parameters_bounds = [
            SimpleParameter(title=f'Hyperparameter #{i + 1} {edge}', name=f'hyperparameter_{i}_{edge}', type='float')
            for i in range(num_hyperparameters) for edge in ['min', 'max']
        ]
        bounds_parameters = [
            SimpleParameter(title=f'Axis #{i + 1} {edge}', name=f'axis_{i}_{edge}', type='float')
            for i in range(dimensionality) for edge in ['min', 'max']
        ]
        func_parameters = [
            ListParameter(title='Method', name='method', limits=['global', 'local', 'hgdl'], default='global'),
            ListParameter(title='Acquisition Function', name='acquisition_function',
                          limits=['variance', 'shannon_ig', 'ucb'], default='variance'),
            SimpleParameter(title='Queue Length', name='n', value=1, type='int'),
            SimpleParameter(title='Population Size', name='pop_size', value=20, type='int'),
            SimpleParameter(title='Tolerance', name='tol', value=1e-6, type='float'),
        ]

        global_train_parameter = TrainingParameter(
            title='Train globally at...', name='global_training', addText='Add',
            children=[SimpleParameter(title='N=', name=str(uuid.uuid4()), value=N, type='int')
                      for N in default_retrain_globally_at]
        )
        local_train_parameter = TrainingParameter(
            title='Train locally at...', name='local_training', addText='Add',
            children=[SimpleParameter(title='N=', name=str(uuid.uuid4()), value=N, type='int')
                      for N in default_retrain_locally_at]
        )

        parameters = func_parameters + [
            GroupParameter(name='bounds', title='Axis Bounds', children=bounds_parameters),
            GroupParameter(name='hyperparameters', title='Hyperparameter Bounds',
                           children=hyper_parameters + hyper_parameters_bounds),
            global_train_parameter,
            local_train_parameter,
        ]
        return GroupParameter(name='top', children=parameters), num_hyperparameters, dimensionality

    def test_string_key_access(self):
        params, num_hp, dim = self._build_parameters()
        # n has value=1
        assert params['n'] == 1

    def test_tuple_path_access(self):
        params, num_hp, dim = self._build_parameters()
        # bounds group has axis_0_min etc., all default None/0
        val = params[('bounds', 'axis_0_min')]
        # Just confirm it's accessible (default is None since no value given)
        assert val is None or isinstance(val, (int, float))

    def test_set_via_string_key(self):
        params, num_hp, dim = self._build_parameters()
        params['n'] = 5
        assert params['n'] == 5

    def test_set_via_tuple_key(self):
        params, num_hp, dim = self._build_parameters()
        params[('bounds', 'axis_0_min')] = 10.0
        assert params[('bounds', 'axis_0_min')] == 10.0

    def test_child_navigation_with_set_value(self):
        from tsuchinoko.parameters.tree import Parameter
        params, num_hp, dim = self._build_parameters()
        cb_called = []
        cb = lambda p, v: cb_called.append(v)
        params.child('hyperparameters', 'hyperparameter_0').sigValueChanged.connect(cb)
        params.child('hyperparameters', 'hyperparameter_0').setValue(255.0, blockSignal=None)
        assert params[('hyperparameters', 'hyperparameter_0')] == 255.0
        assert cb_called == [255.0]

    def test_child_set_value_block_signal(self):
        params, num_hp, dim = self._build_parameters()
        cb_called = []
        cb = lambda p, v: cb_called.append(v)
        node = params.child('hyperparameters', 'hyperparameter_0')
        node.sigValueChanged.connect(cb)
        node.setValue(255.0, blockSignal=cb)
        assert params[('hyperparameters', 'hyperparameter_0')] == 255.0
        assert cb_called == []

    def test_training_schedule_iteration(self):
        params, num_hp, dim = self._build_parameters()
        train_at = set(child.value() for child in params.child('global_training').children())
        assert train_at == {20, 50, 100, 400, 1000}

    def test_save_state(self):
        params, num_hp, dim = self._build_parameters()
        state = params.saveState()
        assert state['name'] == 'top'
        assert 'children' in state

    def test_has_children(self):
        params, num_hp, dim = self._build_parameters()
        assert params.hasChildren() is True

    def test_list_parameter_method_access(self):
        params, num_hp, dim = self._build_parameters()
        assert params['method'] == 'global'

    def test_set_bounds_in_loop(self):
        """Mimic engine __init__ loop: set bounds via tuple key."""
        params, num_hp, dim = self._build_parameters()
        parameter_bounds = [(0, 100), (0, 200)]
        for i in range(dim):
            for j, edge in enumerate(['min', 'max']):
                params[('bounds', f'axis_{i}_{edge}')] = parameter_bounds[i][j]
        assert params[('bounds', 'axis_0_min')] == 0
        assert params[('bounds', 'axis_0_max')] == 100
        assert params[('bounds', 'axis_1_min')] == 0
        assert params[('bounds', 'axis_1_max')] == 200
