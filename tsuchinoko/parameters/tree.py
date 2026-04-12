"""Lightweight, pure-Python replacement for pyqtgraph's ParameterTree.

Provides the same API surface used by Tsuchinoko's adaptive engines so that
they can run without any Qt dependency.
"""
from __future__ import annotations

import uuid
from typing import Any, Callable, Dict, List, Optional, Sequence, Union


class Signal:
    """Minimal signal/slot mechanism mimicking pyqtgraph's sigValueChanged."""

    def __init__(self) -> None:
        self._callbacks: List[Callable] = []

    def connect(self, cb: Callable) -> None:
        if cb not in self._callbacks:
            self._callbacks.append(cb)

    def disconnect(self, cb: Optional[Callable] = None) -> None:
        if cb is None:
            self._callbacks.clear()
        elif cb in self._callbacks:
            self._callbacks.remove(cb)

    def emit(self, *args, exclude: Optional[Callable] = None) -> None:
        for cb in list(self._callbacks):
            if cb is not exclude:
                cb(*args)


class Parameter:
    """A single node in the parameter tree.

    Supports hierarchical navigation, value get/set with optional signal
    blocking, dict-like access by name string or tuple path, and serialisation
    via ``saveState()``.
    """

    def __init__(
        self,
        name: str = '',
        title: str = '',
        value: Any = None,
        type: str = '',
        children: Optional[Sequence[Union['Parameter', dict]]] = None,
        removable: bool = False,
        renamable: bool = False,
        **kwargs: Any,
    ) -> None:
        self.name: str = name
        self.title: str = title
        self.type: str = type
        self.removable: bool = removable
        self.renamable: bool = renamable

        self._value: Any = value
        self._children_map: Dict[str, 'Parameter'] = {}
        self._children_list: List['Parameter'] = []

        self.sigValueChanged: Signal = Signal()

        if children:
            for child in children:
                self.addChild(child)

    # ------------------------------------------------------------------
    # Value access
    # ------------------------------------------------------------------

    def value(self) -> Any:
        return self._value

    def setValue(self, value: Any, blockSignal: Optional[Callable] = None) -> None:
        self._value = value
        self.sigValueChanged.emit(self, value, exclude=blockSignal)

    # ------------------------------------------------------------------
    # Child management
    # ------------------------------------------------------------------

    def addChild(self, child_or_dict: Union['Parameter', dict]) -> 'Parameter':
        if isinstance(child_or_dict, dict):
            d = child_or_dict.copy()
            name = d.pop('name')
            title = d.pop('title', '')
            value = d.pop('value', None)
            type_ = d.pop('type', '')
            removable = d.pop('removable', False)
            renamable = d.pop('renamable', False)
            child = Parameter(
                name=name, title=title, value=value, type=type_,
                removable=removable, renamable=renamable, **d
            )
        else:
            child = child_or_dict

        self._children_map[child.name] = child
        self._children_list.append(child)
        return child

    def children(self) -> List['Parameter']:
        return list(self._children_list)

    def hasChildren(self) -> bool:
        return bool(self._children_list)

    def child(self, *path: str) -> 'Parameter':
        """Navigate to a descendant by successive name lookups."""
        node = self
        for name in path:
            try:
                node = node._children_map[name]
            except KeyError:
                raise KeyError(f"No child named {name!r} under {node.name!r}")
        return node

    # ------------------------------------------------------------------
    # Dict-like access
    # ------------------------------------------------------------------

    def __getitem__(self, key: Union[str, tuple]) -> Any:
        if isinstance(key, tuple):
            return self.child(*key[:-1])._children_map[key[-1]].value()
        try:
            return self._children_map[key].value()
        except KeyError:
            raise KeyError(key)

    def __setitem__(self, key: Union[str, tuple], value: Any) -> None:
        if isinstance(key, tuple):
            self.child(*key[:-1])._children_map[key[-1]].setValue(value)
        else:
            try:
                self._children_map[key].setValue(value)
            except KeyError:
                raise KeyError(key)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def saveState(self) -> dict:
        state: dict = {'name': self.name, 'title': self.title, 'value': self._value}
        if self._children_list:
            state['children'] = [child.saveState() for child in self._children_list]
        return state


# Aliases expected by existing engine code
SimpleParameter = Parameter
GroupParameter = Parameter


class ListParameter(Parameter):
    """A parameter whose value is restricted to a predefined list of options."""

    def __init__(
        self,
        name: str = '',
        title: str = '',
        limits: Optional[List[Any]] = None,
        default: Any = None,
        **kwargs: Any,
    ) -> None:
        self.limits: List[Any] = list(limits) if limits is not None else []
        # 'default' acts as the initial value
        super().__init__(name=name, title=title, value=default, **kwargs)

    def setValue(self, value: Any, blockSignal: Optional[Callable] = None) -> None:
        if value not in self.limits:
            raise ValueError(
                f"Value {value!r} is not in the allowed list {self.limits!r}"
            )
        super().setValue(value, blockSignal=blockSignal)


class TrainingParameter(Parameter):
    """A group parameter with an 'Add' button that appends integer schedule entries."""

    def __init__(self, addText: str = 'Add', **kwargs: Any) -> None:
        self.addText: str = addText
        super().__init__(**kwargs)

    def addNew(self) -> Parameter:
        """Append a new child with value=1, matching the original pyqtgraph behaviour."""
        return self.addChild({
            'name': str(uuid.uuid4()),
            'title': 'N=',
            'type': 'int',
            'value': 1,
            'removable': True,
            'renamable': False,
        })
