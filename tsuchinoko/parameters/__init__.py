"""Parameter types for Tsuchinoko.

Qt-based parameter types (for the GUI) are registered when PySide6 is available.
Headless code should import from tsuchinoko.parameters.tree instead.
"""

try:
    import uuid

    from pyqtgraph.parametertree import registerParameterType
    from pyqtgraph.parametertree.parameterTypes import GroupParameter

    class TrainingParameter(GroupParameter):
        def __init__(self, **kwargs):
            kwargs['type'] = 'training'
            super(TrainingParameter, self).__init__(**kwargs)

        def addNew(self):
            self.addChild(dict(name=str(uuid.uuid4()), title='N=', type='int', value=1, removable=True, renamable=False))

    registerParameterType('training', TrainingParameter)
except ImportError:
    pass
